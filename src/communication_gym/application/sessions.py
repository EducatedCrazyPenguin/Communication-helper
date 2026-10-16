"""Durable session transitions, transcript revisioning and explicit acceptance."""
from __future__ import annotations

import copy

from communication_gym.domain.models import GymError, content_hash, new_id, utc_now
from communication_gym.storage.repository import Repository


class SessionService:
    def __init__(self, repo: Repository):
        self.repo = repo

    @staticmethod
    def _revision(session: dict, expected_revision: int | None) -> None:
        if expected_revision is not None and (type(expected_revision) is not int or expected_revision != session["revision"]):
            error = GymError("revision_conflict", "The session changed. Reload its current revision before continuing.", 409)
            error.current_revision = session["revision"]
            raise error

    @staticmethod
    def _stage(session: dict, *allowed: str) -> None:
        if session["stage"] not in allowed:
            raise GymError("invalid_transition", "This action is not available at the current session stage.", 409)

    @staticmethod
    def _operation_key(value: str | None) -> None:
        if value is not None and (not isinstance(value, str) or not 1 <= len(value) <= 200):
            raise GymError("validation_error", "Use a nonempty operation key of at most 200 characters.", 422)

    def create(self, data: dict) -> dict:
        payload = copy.deepcopy(data)
        payload.pop("expected_revision", None)
        allowed = {"language", "mode", "exercise", "prep_seconds", "planned_duration", "retry_duration", "equal_retry_duration_confirmed", "consent", "operation_key"}
        if set(payload) - allowed:
            raise GymError("validation_error", "Create a session with exercise and practice settings, not lifecycle-managed fields.", 422)
        self._operation_key(payload.get("operation_key"))
        payload["operation_hash"] = content_hash({key: value for key, value in payload.items() if key not in {"id", "operation_key"}})
        initial_duration = payload.get("planned_duration", 120)
        if type(initial_duration) is int:
            payload.setdefault("retry_duration", max(30, initial_duration // 2))
        return self.repo.create_session(payload)

    def prep(self, session_id: str, bullets: list[str], expected_revision: int | None) -> dict:
        with self.repo.transaction():
            session = self.repo.get_session(session_id)
            self._revision(session, expected_revision)
            self._stage(session, "topic", "prep")
            if not session.get("exercise"):
                raise GymError("exercise_required", "Choose a topic before preparation.", 422)
            if not isinstance(bullets, list) or len(bullets) > 5 or any(not isinstance(item, str) or not item.strip() or len(item) > 200 for item in bullets):
                raise GymError("validation_error", "Use at most five nonempty bullets, each at most 200 code points.", 422)
            return self.repo.update_session(session_id, {"prep_bullets": bullets, "stage": "prep"}, session["revision"])

    def create_attempt(self, session_id: str, role: str, question_id: str | None = None, expected_revision: int | None = None, operation_key: str | None = None) -> dict:
        self._operation_key(operation_key)
        if role not in ("initial", "cross_exam", "retry"):
            raise GymError("validation_error", "Choose initial, cross_exam or retry.", 422)
        with self.repo.transaction():
            session = self.repo.get_session(session_id)
            # A replay has priority over revision checking: it must return the durable
            # first attempt even if that request already advanced the revision.
            if operation_key:
                previous = self.repo.find("attempts", session_id=session_id, operation_key=operation_key)
                if previous:
                    attempt = previous[0]
                    if attempt["role"] != role or attempt.get("question_id") != question_id:
                        raise GymError("operation_conflict", "This operation key was used for a different attempt.", 409)
                    return self.repo.get_attempt(attempt["id"])
            self._revision(session, expected_revision)
            if not session.get("exercise"):
                raise GymError("exercise_required", "Choose a topic before recording.", 422)
            attempts = [item for item in session["attempts"] if item["role"] == role]
            if role == "initial":
                self._stage(session, "prep", "initial_recording")
            elif role == "retry":
                self._stage(session, "retry_recording")
                if not session.get("initial_analysis"):
                    raise GymError("invalid_transition", "Complete and review the first analysis before retrying.", 409)
                if not session.get("questions_finished"):
                    raise GymError("invalid_transition", "Finish or explicitly skip follow-up questions before retrying.", 409)
            else:
                self._stage(session, "cross_exam")
                if not question_id:
                    raise GymError("validation_error", "A spoken follow-up must reference its open question.", 422)
                question = self.repo.get("cross_exam_questions", question_id)
                if question["session_id"] != session_id or question["status"] != "open":
                    raise GymError("invalid_transition", "Answer the current open question in this session.", 409)
                attempts = [item for item in attempts if item.get("question_id") == question_id]
            if role != "cross_exam" and question_id is not None:
                raise GymError("validation_error", "Only cross-exam attempts may reference a question.", 422)
            if any(item.get("reviewed") or item["status"] not in {"failed", "cancelled", "interrupted"} for item in attempts):
                raise GymError("attempt_exists", "This attempt already exists. Resume it, or discard an incomplete recording before recording again.", 409)
            ordinal = max((item["ordinal"] for item in session["attempts"] if item["role"] == role), default=0) + 1
            planned_duration = session["planned_duration"] if role == "initial" else (session.get("retry_duration") or max(30, session["planned_duration"] // 2))
            if role == "cross_exam":
                planned_duration = 120
            attempt = self.repo.put("attempts", {"id": new_id(), "session_id": session_id, "role": role, "ordinal": ordinal, "status": "created", "question_id": question_id, "operation_key": operation_key, "generation": session.get("generation", 0), "planned_duration": planned_duration, "reviewed": False})
            next_stage = "cross_exam" if role == "cross_exam" else f"{role}_recording"
            self.repo.update_session(session_id, {"stage": next_stage}, session["revision"])
            return self.repo.get_attempt(attempt["id"])

    def accept_transcript(self, attempt_id: str, transcript_id: str, expected_revision: int | None = None) -> dict:
        with self.repo.transaction():
            attempt = self.repo.get("attempts", attempt_id)
            transcript = self.repo.get("transcripts", transcript_id)
            session = self.repo.get_session(attempt["session_id"])
            if transcript["attempt_id"] != attempt_id:
                raise GymError("ownership_conflict", "This transcript belongs to a different attempt.", 409)
            if attempt.get("reviewed") and attempt.get("active_transcript_id") == transcript_id:
                return self.repo.get_session(session["id"])
            self._revision(session, expected_revision)
            if not transcript["text"].strip():
                raise GymError("no_speech", "No speech was detected. Keep this result for review and record again.", 422)
            role = attempt["role"]
            if role == "cross_exam":
                self._stage(session, "cross_exam")
                question = self.repo.get("cross_exam_questions", attempt["question_id"])
                if question["status"] != "open":
                    raise GymError("invalid_transition", "This question is no longer open.", 409)
                stage = "cross_exam"
            else:
                self._stage(session, f"{role}_review")
                stage = f"{role}_analysis"
            self.repo.put("attempts", {**attempt, "active_transcript_id": transcript_id, "status": "accepted", "reviewed": True, "reviewed_at": utc_now()})
            return self.repo.update_session(session["id"], {"stage": stage}, session["revision"])

    def correct_transcript(self, attempt_id: str, segments: list[dict]) -> dict:
        with self.repo.transaction():
            attempt = self.repo.get("attempts", attempt_id)
            session = self.repo.get_session(attempt["session_id"])
            transcripts = self.repo.find("transcripts", attempt_id=attempt_id)
            parent = next((item for item in transcripts if item["id"] == attempt.get("active_transcript_id")), None)
            parent = parent or (max(transcripts, key=lambda row: row["revision"]) if transcripts else None)
            if parent is None:
                raise GymError("invalid_transition", "Transcribe this attempt before making a correction.", 409)
            if not isinstance(segments, list) or any(not isinstance(item, dict) or not isinstance(item.get("text"), str) for item in segments):
                raise GymError("validation_error", "A correction requires an ordered list of segment text objects.", 422)
            prior_segments = {item["id"]: item for item in parent.get("segments", [])}
            corrected_segments = []
            cursor = 0
            for index, submitted in enumerate(segments):
                original = prior_segments.get(submitted.get("id"))
                # Index fallback supports text-only editors, but never copies timing
                # from a different segment just because its text appears elsewhere.
                if original is None and "id" not in submitted and index < len(parent.get("segments", [])):
                    original = parent["segments"][index]
                text = submitted["text"]
                unchanged = original is not None and text == original["text"]
                if unchanged:
                    segment = copy.deepcopy(original)
                    offset_delta = cursor - original["start_char"]
                    for word in segment.get("words", []):
                        word["id"] = new_id()
                        for key in ("start_char", "end_char"):
                            if word.get(key) is not None:
                                word[key] += offset_delta
                else:
                    segment = {"text": text, "start": None, "end": None, "timing_kind": "none", "timing_origin": "human_corrected", "words": [], "uncertain": False}
                segment.update({"id": new_id(), "text": text, "start_char": cursor, "end_char": cursor + len(text)})
                corrected_segments.append(segment)
                cursor += len(text) + 1
            canonical = "\n".join(item["text"] for item in corrected_segments)
            if canonical == parent["text"]:
                return parent
            correction = self.repo.put("transcripts", {"id": new_id(), "session_id": session["id"], "attempt_id": attempt_id, "kind": "human_corrected", "parent_id": parent["id"], "text": canonical, "segments": corrected_segments, "provenance": {"source": "human", "parent_content_hash": parent["content_hash"]}, "diagnostics": {"human_corrected": True, "original_stitch_diagnostics": parent.get("diagnostics", {})}})
            self.repo.put("attempts", {**attempt, "active_transcript_id": correction["id"], "status": "review", "reviewed": False, "generation": attempt.get("generation", 0) + 1})
            # Do not rewrite historical analyses. Active selection is transcript-bound;
            # old results/comparisons remain inspectable but cannot enter progress.
            for job in self.repo.find("jobs", session_id=session["id"]):
                if job["status"] in ("queued", "running", "cancel_requested"):
                    self.repo.put("jobs", {**job, "status": "cancelled", "generation": job.get("generation", 0) + 1, "safe_error": "transcript_revised", "finished_at": utc_now()})
            changes = {"stage": "cross_exam" if attempt["role"] == "cross_exam" else f"{attempt['role']}_review", "generation": session.get("generation", 0) + 1, "active_comparison_id": None, "completed_at": None}
            if attempt["role"] in ("initial", "cross_exam"):
                changes.update({"questions_finished": False, "questions_skipped": False})
                for question in self.repo.find("cross_exam_questions", session_id=session["id"]):
                    if attempt["role"] == "initial" or (question["id"] != attempt.get("question_id") and question["status"] == "open"):
                        self.repo.put("cross_exam_questions", {**question, "status": "skipped", "skip_reason": "transcript_revised", "invalidated_by_transcript_id": correction["id"]})
                if attempt["role"] == "cross_exam":
                    question = self.repo.get("cross_exam_questions", attempt["question_id"])
                    history = question.get("resolution_history", [])
                    if question.get("resolution") or question.get("assessment"):
                        history = [*history, question.get("resolution") or question.get("assessment")]
                    self.repo.put("cross_exam_questions", {**question, "status": "open", "resolution": None, "assessment": None, "resolution_history": history, "answer_transcript_id": None})
                for downstream in session["attempts"]:
                    invalidated_role = downstream["role"] == "retry" or (attempt["role"] == "initial" and downstream["role"] == "cross_exam")
                    if downstream["id"] != attempt_id and invalidated_role:
                        self.repo.put("attempts", {**{key: value for key, value in downstream.items() if key not in {"transcripts", "transcript", "recording", "analyses", "analysis"}}, "status": "cancelled", "reviewed": False, "generation": downstream.get("generation", 0) + 1, "invalidated_by_transcript_id": correction["id"]})
            self.repo.invalidate_session_derivatives(session["id"])
            self.repo.update_session(session["id"], changes, session["revision"])
            return correction

    def finish_questions(self, session_id: str, expected_revision: int | None = None) -> dict:
        with self.repo.transaction():
            session = self.repo.get_session(session_id)
            if session["stage"] == "retry_recording" and session.get("questions_finished"):
                return session
            self._revision(session, expected_revision)
            self._stage(session, "breakdown", "cross_exam")
            if not session.get("initial_analysis"):
                raise GymError("invalid_transition", "Complete the first analysis before moving to retry.", 409)
            answered = [question for question in session["questions"] if question["status"] == "answered"]
            # The finish operation is the explicit decision to end/skip this phase;
            # skipped open questions are not rewritten as answered or resolved.
            for question in session["questions"]:
                if question["status"] == "open":
                    self.repo.put("cross_exam_questions", {**question, "status": "skipped", "skip_reason": "user_finished_questions"})
            return self.repo.update_session(session_id, {"stage": "retry_recording", "questions_finished": True, "questions_skipped": not bool(answered)}, session["revision"])

    def skip_question(self, question_id: str, expected_revision: int | None = None) -> dict:
        with self.repo.transaction():
            question = self.repo.get("cross_exam_questions", question_id)
            session = self.repo.get_session(question["session_id"])
            if question["status"] == "skipped":
                return question
            self._revision(session, expected_revision)
            self._stage(session, "cross_exam")
            if question["status"] != "open":
                raise GymError("invalid_transition", "An answered question cannot be marked skipped.", 409)
            result = self.repo.put("cross_exam_questions", {**question, "status": "skipped", "skip_reason": "user_skipped"})
            self.repo.update_session(session["id"], {}, session["revision"])
            return result

    def complete(self, session_id: str, expected_revision: int | None = None) -> dict:
        with self.repo.transaction():
            session = self.repo.get_session(session_id)
            if session["stage"] == "complete":
                return session
            self._revision(session, expected_revision)
            self._stage(session, "comparison")
            initial, retry, comparison = session.get("initial_analysis"), session.get("retry_analysis"), session.get("comparison")
            if not initial or not retry or not comparison or comparison.get("status") != "succeeded":
                raise GymError("invalid_transition", "Finish both analyses and their comparison before completing the session.", 409)
            if comparison["initial_analysis_id"] != initial["id"] or comparison["retry_analysis_id"] != retry["id"]:
                raise GymError("stale_comparison", "Recompute the comparison for the currently accepted transcripts.", 409)
            self.repo.invalidate_session_derivatives(session_id)
            return self.repo.update_session(session_id, {"stage": "complete", "completed_at": utc_now(), "active_comparison_id": comparison["id"]}, session["revision"])
