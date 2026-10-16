"""Offline lifecycle/storage behavior; all content and roots are synthetic."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from communication_gym.application.sessions import SessionService
from communication_gym.domain.models import GymError, content_hash, new_id
from communication_gym.storage.repository import Repository


@pytest.fixture
def repo(tmp_path):
    instance = Repository(tmp_path / "gym")
    instance.initialize()
    yield instance
    instance.close()


def create(service, **changes):
    return service.create({"language": "en", "mode": "compare", "exercise": {"title": "Battery A and B", "prompt": "Explain the roles of these two batteries.", "audience": "A colleague", "source": "curated"}, **changes})


def prepared(repo):
    service = SessionService(repo)
    session = create(service)
    return service, service.prep(session["id"], ["Name A and B"], session["revision"])


def transcript_for(repo, attempt, text="Battery A is the main battery."):
    transcript = repo.put("transcripts", {"attempt_id": attempt["id"], "kind": "asr_original", "text": text, "segments": [{"id": new_id(), "text": text, "start": 0.0, "end": 3.0, "words": [{"id": new_id(), "text": "Battery", "start_char": 0, "end_char": 7, "start": 0.0, "end": 0.5, "timing_origin": "aligned"}]}], "original": {"raw_segments": [{"text": text}]}})
    if attempt["role"] != "cross_exam":
        repo.advance_session(attempt["session_id"], f"{attempt['role']}_review")
    return transcript


def accepted(repo, service, session, role="initial", question_id=None):
    current = repo.get_session(session["id"])
    attempt = service.create_attempt(session["id"], role, question_id=question_id, expected_revision=current["revision"], operation_key=new_id())
    transcript = transcript_for(repo, attempt)
    service.accept_transcript(attempt["id"], transcript["id"], repo.get_session(session["id"])["revision"])
    return attempt, transcript


def analyze(repo, attempt, transcript):
    segment = transcript["segments"][0]
    evidence = {"transcript_id": transcript["id"], "segment_id": segment["id"], "start_char": 0, "end_char": 7, "quote": "Battery", "timing_kind": "estimated", "start_s": 0.0, "end_s": 3.0}
    issue = {"id": new_id(), "kind": "missing_context", "severity": "costly", "source": "likely_speaker_issue", "confidence": "medium", "evidence": [evidence], "reviewed_range": [], "listener_consequence": "The other role is not stated.", "corrective_pattern": "Name both roles.", "target": "B"}
    result = repo.put("analyses", {"attempt_id": attempt["id"], "transcript_id": transcript["id"], "critic": {"issues": [issue]}, "scores": {"rubric_version": "v1", "dimensions": {"clarity": 50}, "criteria": []}, "listener": {"clearly_understood": [{"text": "A is the main battery.", "evidence": [evidence]}]}, "feedback": [issue], "coverage": {"complete": True}, "provenance": {"rubric_version": "v1", "provider": "test", "model": "test"}})
    repo.put("attempts", {"id": attempt["id"], "status": "analyzed"})
    repo.advance_session(attempt["session_id"], "breakdown" if attempt["role"] == "initial" else "retry_analysis")
    return result


def comparable_session(repo):
    service, session = prepared(repo)
    initial, initial_text = accepted(repo, service, session)
    first = analyze(repo, initial, initial_text)
    session = service.finish_questions(session["id"], repo.get_session(session["id"])["revision"])
    retry, retry_text = accepted(repo, service, session, "retry")
    second = analyze(repo, retry, retry_text)
    comparison = repo.put("retry_comparisons", {"session_id": session["id"], "initial_analysis_id": first["id"], "retry_analysis_id": second["id"], "comparable": True, "issue_changes": []})
    session = repo.update_session(session["id"], {"stage": "comparison", "active_comparison_id": comparison["id"]})
    return service, session, initial, initial_text, retry, retry_text


def test_all_durable_stages_and_completion(repo):
    service, session, initial, initial_text, retry, retry_text = comparable_session(repo)
    assert session["initial_attempt"]["id"] == initial["id"]
    assert session["retry_attempt"]["id"] == retry["id"]
    assert session["initial_analysis"] and session["retry_analysis"]
    completed = service.complete(session["id"], session["revision"])
    assert completed["stage"] == "complete"
    assert completed["resume_stage"] == "complete"
    assert service.complete(session["id"], session["revision"])["revision"] == completed["revision"]
    root = repo.root
    repo.close()
    repo.initialize()
    restored = repo.get_session(session["id"])
    assert restored["stage"] == "complete"
    assert restored["initial_attempt"]["transcript"]["text"] == initial_text["text"]
    assert restored["retry_attempt"]["transcript"]["id"] == retry_text["id"]
    assert repo.root == root


@pytest.mark.parametrize("stage", ["topic", "prep", "initial_recording", "initial_transcribing", "initial_review", "initial_analysis", "breakdown", "cross_exam", "retry_recording", "retry_transcribing", "retry_review", "retry_analysis", "comparison"])
def test_resume_checkpoint_after_restart(repo, stage):
    session = create(SessionService(repo))
    repo.advance_session(session["id"], stage)
    repo.close()
    repo.initialize()
    assert repo.get_session(session["id"])["stage"] == stage


def test_prep_limits_and_strict_duration(repo):
    service = SessionService(repo)
    session = create(service)
    valid = service.prep(session["id"], [f"Point {index}" for index in range(5)], session["revision"])
    for invalid in (["x"] * 6, [""], [" "], ["अ" * 201], "a script", [42]):
        with pytest.raises(GymError) as error:
            service.prep(session["id"], invalid, valid["revision"])
        assert error.value.status == 422
    assert repo.get_session(session["id"])["prep_bullets"] == valid["prep_bullets"]
    with pytest.raises(GymError):
        create(service, planned_duration=90)
    with pytest.raises(GymError):
        create(service, retry_duration=120)
    assert create(service, retry_duration=120, equal_retry_duration_confirmed=True)["retry_duration"] == 120


def test_retry_and_acceptance_stage_guards(repo):
    service, session = prepared(repo)
    with pytest.raises(GymError, match="current session stage"):
        service.create_attempt(session["id"], "retry", expected_revision=session["revision"])
    attempt = service.create_attempt(session["id"], "initial", expected_revision=session["revision"])
    transcript = transcript_for(repo, attempt)
    repo.advance_session(session["id"], "initial_transcribing")
    with pytest.raises(GymError) as error:
        service.accept_transcript(attempt["id"], transcript["id"])
    assert error.value.code == "invalid_transition"


def test_creation_and_attempt_operations_are_idempotent(repo):
    service = SessionService(repo)
    key = new_id()
    first = create(service, operation_key=key)
    assert create(service, operation_key=key)["id"] == first["id"]
    with pytest.raises(GymError) as error:
        create(service, operation_key=key, language="hi")
    assert error.value.code == "operation_conflict"
    session = service.prep(first["id"], [], first["revision"])
    key = new_id()
    attempt = service.create_attempt(session["id"], "initial", expected_revision=session["revision"], operation_key=key)
    replay = service.create_attempt(session["id"], "initial", expected_revision=session["revision"], operation_key=key)
    assert replay["id"] == attempt["id"]
    assert len(repo.find("attempts", session_id=session["id"])) == 1
    with pytest.raises(GymError) as error:
        service.create_attempt(session["id"], "retry", operation_key=key)
    assert error.value.code == "operation_conflict"
    with pytest.raises(GymError):
        service.create_attempt(session["id"], "initial", operation_key=new_id())


def test_concurrent_revision_updates_have_one_winner(repo):
    session = create(SessionService(repo))
    barrier = Barrier(2)

    def mutate(value):
        barrier.wait()
        try:
            return repo.update_session(session["id"], {"prep_bullets": [value]}, session["revision"])["revision"]
        except GymError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(mutate, ["one", "two"]))
    assert sorted(result, key=str) == [1, "revision_conflict"]


def test_correction_keeps_original_and_invalidates_active_derivatives(repo):
    service, session, initial, original, retry, retry_text = comparable_session(repo)
    service.complete(session["id"], session["revision"])
    before = repo.get("transcripts", original["id"])
    repo.put("personal_patterns", {"kind": "missing_context", "evidence_session_ids": [session["id"]]})
    job = repo.put("jobs", {"session_id": session["id"], "attempt_id": initial["id"], "operation": "analyze", "status": "running", "generation": 0})
    corrected = service.correct_transcript(initial["id"], [{"id": original["segments"][0]["id"], "text": "Battery A is primary; Battery B is backup."}])
    assert corrected["kind"] == "human_corrected" and corrected["parent_id"] == original["id"]
    assert corrected["content_hash"] == content_hash(corrected["text"])
    assert corrected["segments"][0]["words"] == []
    assert corrected["segments"][0]["start"] is None
    assert repo.get("transcripts", original["id"]) == before
    current = repo.get_session(session["id"])
    assert current["stage"] == "initial_review"
    assert current["initial_analysis"] is None and current["comparison"] is None
    assert current["retry_attempt"] is None
    assert current["analyses"] and current["comparisons"]
    assert repo.get("jobs", job["id"])["status"] == "cancelled"
    assert repo.find("personal_patterns") == []
    with pytest.raises(GymError) as error:
        repo.put("transcripts", {"id": original["id"], "text": "Overwrite original"})
    assert error.value.code == "immutable_record"


def test_unchanged_segment_keeps_timings_but_changes_owner_ids(repo):
    service, session = prepared(repo)
    attempt, original = accepted(repo, service, session)
    source = original["segments"][0]
    corrected = service.correct_transcript(attempt["id"], [{"id": source["id"], "text": source["text"]}, {"text": "The second role is backup."}])
    assert corrected["segments"][0]["words"][0]["start"] == 0.0
    assert corrected["segments"][0]["words"][0]["segment_id"] == corrected["segments"][0]["id"]
    assert corrected["segments"][0]["transcript_id"] == corrected["id"]
    assert len(repo.find("transcript_segments", transcript_id=corrected["id"])) == 2
    assert len(repo.find("word_timings", transcript_id=corrected["id"])) == 1


def test_one_open_question_answer_and_skip_differ(repo):
    service, session = prepared(repo)
    initial, text = accepted(repo, service, session)
    analysis = analyze(repo, initial, text)
    repo.advance_session(session["id"], "cross_exam")
    question = repo.put("cross_exam_questions", {"session_id": session["id"], "analysis_id": analysis["id"], "question": "What is Battery B's role?", "target_ids": [analysis["critic"]["issues"][0]["id"]]})
    with pytest.raises(GymError) as error:
        repo.put("cross_exam_questions", {"session_id": session["id"], "question": "Another question?"})
    assert error.value.status == 409
    with pytest.raises(GymError):
        service.create_attempt(session["id"], "cross_exam")
    answer, answer_text = accepted(repo, service, session, "cross_exam", question["id"])
    assert answer["question_id"] == question["id"]
    assert repo.get("cross_exam_questions", question["id"])["status"] == "open"
    skipped = service.skip_question(question["id"], repo.get_session(session["id"])["revision"])
    assert skipped["status"] == "skipped" and "resolution" not in skipped
    next_stage = service.finish_questions(session["id"], repo.get_session(session["id"])["revision"])
    assert next_stage["stage"] == "retry_recording"
    assert next_stage["questions_skipped"] is True
    assert repo.get("transcripts", text["id"])["attempt_id"] == initial["id"]
    assert repo.get("transcripts", answer_text["id"])["attempt_id"] == answer["id"]


def test_indexed_child_lookups_and_rejected_ungrounded_evidence(repo):
    service, session = prepared(repo)
    attempt, text = accepted(repo, service, session)
    analysis = analyze(repo, attempt, text)
    for table in ("communication_issues", "issue_evidence", "score_snapshots", "listener_models", "coach_feedback"):
        rows = repo.find(table, analysis_id=analysis["id"])
        assert rows and all(row["session_id"] == session["id"] for row in rows)
    invalid = {**analysis, "id": new_id(), "critic": {"issues": [{**analysis["critic"]["issues"][0], "id": new_id(), "evidence": [{**analysis["critic"]["issues"][0]["evidence"][0], "quote": "invented"}]}]}}
    with pytest.raises(GymError) as error:
        repo.put("analyses", invalid)
    assert error.value.code == "invalid_evidence"
    assert len(repo.find("analyses", attempt_id=attempt["id"])) == 1


def test_cancelled_job_cannot_commit_late_artifacts(repo):
    service, session = prepared(repo)
    attempt = service.create_attempt(session["id"], "initial")
    job = repo.put("jobs", {"session_id": session["id"], "attempt_id": attempt["id"], "operation": "transcribe", "status": "running", "generation": 1})
    repo.put("jobs", {"id": job["id"], "status": "cancelled"})
    with pytest.raises(GymError) as error:
        repo.put("transcripts", {"attempt_id": attempt["id"], "text": "Late result", "job_id": job["id"], "generation": 1})
    assert error.value.code == "stale_generation"
    assert repo.find("transcripts", attempt_id=attempt["id"]) == []
    with pytest.raises(GymError):
        repo.put("jobs", {"id": job["id"], "status": "succeeded"})


def test_delete_cascades_and_retains_retryable_purge_tasks(repo):
    service, session, initial, original, retry, retry_text = comparable_session(repo)
    service.correct_transcript(initial["id"], [{"text": "Corrected synthetic sentence."}])
    recording_id = new_id()
    relative = f"data/sessions/{session['id']}/{recording_id}.wav"
    target = repo.path(relative)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"synthetic owned audio")
    repo.put("audio_recordings", {"id": recording_id, "attempt_id": initial["id"], "relative_path": relative, "status": "ready"})
    repo.delete_session(session["id"])
    with pytest.raises(GymError) as error:
        repo.get_session(session["id"])
    assert error.value.status == 404
    assert repo.find("transcripts", session_id=session["id"]) == []
    assert repo.find("analyses", session_id=session["id"]) == []
    assert repo.find("deletion_tasks", session_id=session["id"])
    assert target.exists()  # A purge runner, not a false deletion claim, owns this step.


@pytest.mark.parametrize("relative", ["../private.wav", "/absolute.wav", "data/../secret", "C:/outside.wav", "data/no-owner.wav", "data\\escape.wav", "data//bad.wav"])
def test_managed_paths_reject_unsafe_names(repo, relative):
    with pytest.raises(GymError) as error:
        repo.path(relative)
    assert error.value.code == "unsafe_path"


def test_missing_ids_unknown_collections_and_cross_session_links(repo):
    with pytest.raises(GymError) as error:
        repo.get("sessions", new_id())
    assert error.value.status == 404
    with pytest.raises(GymError):
        repo.find("arbitrary_table")
    service, first = prepared(repo)
    second = create(service)
    attempt = service.create_attempt(first["id"], "initial")
    with pytest.raises(GymError) as error:
        repo.put("transcripts", {"attempt_id": attempt["id"], "session_id": second["id"], "text": "Wrong owner"})
    assert error.value.code == "ownership_conflict"


def test_empty_transcript_is_reviewable_but_cannot_be_scored(repo):
    service, session = prepared(repo)
    attempt = service.create_attempt(session["id"], "initial")
    transcript = repo.put("transcripts", {"attempt_id": attempt["id"], "text": "", "segments": []})
    repo.advance_session(session["id"], "initial_review")
    with pytest.raises(GymError) as error:
        service.accept_transcript(attempt["id"], transcript["id"])
    assert error.value.code == "no_speech"
    assert repo.get("transcripts", transcript["id"])["text"] == ""


def test_creation_cannot_inject_completed_lifecycle_fields(repo):
    with pytest.raises(GymError) as error:
        create(SessionService(repo), stage="complete")
    assert error.value.status == 422
    assert repo.list_sessions() == []


def test_analysis_cache_is_session_scoped_and_version_sensitive(repo):
    service, session = prepared(repo)
    attempt, transcript = accepted(repo, service, session)
    original = analyze(repo, attempt, transcript)
    replay = repo.put("analyses", {**original, "id": new_id()})
    assert replay["id"] == original["id"]
    changed = repo.put("analyses", {**original, "id": new_id(), "critic": {"issues": []}, "feedback": [], "provenance": {**original["provenance"], "model": "different-test-model"}})
    assert changed["id"] != original["id"]
    assert changed["cache_key"] != original["cache_key"]


def test_model_answer_can_cite_both_owned_transcripts_without_replacing_analysis(repo):
    service, session, initial, original, retry, retry_text = comparable_session(repo)
    initial_analysis = session["initial_analysis"]
    retry_analysis = session["retry_analysis"]
    support = [initial_analysis["critic"]["issues"][0]["evidence"][0], retry_analysis["critic"]["issues"][0]["evidence"][0]]
    example = repo.put("analyses", {"attempt_id": retry["id"], "transcript_id": retry_text["id"], "subsystem": "model_answer", "payload": {"text": "A synthetic example", "supporting_source_spans": support}})
    assert example["subsystem"] == "model_answer"
    assert repo.get_session(session["id"])["retry_analysis"]["id"] == retry_analysis["id"]
    # The same cross-transcript source is forbidden for ordinary retry analysis.
    with pytest.raises(GymError) as error:
        repo.put("analyses", {"attempt_id": retry["id"], "transcript_id": retry_text["id"], "subsystem": "communication", "payload": {"evidence": support}})
    assert error.value.code == "invalid_evidence"


def test_answer_correction_reopens_question_and_retains_old_resolution(repo):
    service, session = prepared(repo)
    initial, initial_text = accepted(repo, service, session)
    analysis = analyze(repo, initial, initial_text)
    repo.advance_session(session["id"], "cross_exam")
    question = repo.put("cross_exam_questions", {"session_id": session["id"], "analysis_id": analysis["id"], "question": "What is B's role?"})
    answer, answer_text = accepted(repo, service, session, "cross_exam", question["id"])
    repo.put("cross_exam_questions", {"id": question["id"], "status": "answered", "resolution": {"status": "partly_resolved"}})
    correction = service.correct_transcript(answer["id"], [{"text": "Battery B is the backup battery."}])
    revised_question = repo.get("cross_exam_questions", question["id"])
    assert revised_question["status"] == "open" and revised_question["resolution"] is None
    assert revised_question["resolution_history"] == [{"status": "partly_resolved"}]
    assert repo.get("transcripts", answer_text["id"])["kind"] == "asr_original"
    assert repo.get("attempts", answer["id"])["active_transcript_id"] == correction["id"]
