"""Bounded independent model tasks with exact, server-resolved Unicode evidence."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
from pathlib import Path
import time
from uuid import uuid4

from communication_gym.domain.models import GymError
from communication_gym.providers.openai_compatible import validate_schema
from communication_gym.application.listener import LISTENER_FIELDS, listener_input, merge_listener
from communication_gym.application.heuristics import candidates, delivery_metrics
from communication_gym.application.comparison import comparison_metrics, comparable, score_deltas
from communication_gym.application.cross_exam import open_question, unresolved_targets, validate_resolution

VERSION = "1"
ROOT = Path(__file__).resolve().parents[1]
KINDS = ("missing_context", "ambiguous_reference", "unnamed_entity", "unclear_relationship", "implicit_comparison", "missing_cause", "buried_conclusion", "topic_jump", "unfinished_thought", "tangent", "unnecessary_detail", "repetition", "contradiction", "chronology", "ungrounded_term", "missing_consequence", "missing_transition", "question_not_answered", "obscuring_hedging", "information_overload", "listener_load", "word_retrieval", "obstructive_filler", "vague_quantity", "weak_call_to_action", "weak_proposal")
DIMENSIONS = {
    "context": ("subject_named", "purpose_stated", "audience_grounded", "scope_bounded"),
    "structure": ("opening_mainline", "logical_order", "transitions", "closure"),
    "clarity": ("entities_named", "references_resolve", "relations_explicit", "conclusion_explicit"),
    "directness": ("answers_question", "answer_early", "relevance", "concise_path"),
    "precision": ("distinctions", "qualifiers", "consistency", "consequences"),
    "listener_load": ("few_unresolved_entities", "grounded_terms", "closed_branches", "manageable_sequence"),
    "delivery": ("repetition_not_obstructive", "restarts_recovered", "retrieval_not_obscuring", "fillers_not_obstructive")}
SUPPLEMENTAL = {"compare": ("a_b_defined", "dimensions_named", "dimensionwise_comparison", "consequence_recommendation"),
                "proposal": ("problem_audience", "value_cost", "evidence_risks", "specific_ask"),
                "interview": ("answers_requested_parts", "grounds_answer", "resolves_gap", "states_limits")}


def string(maximum=1500, enum=None):
    schema = {"type": "string", "maxLength": maximum}
    if enum is not None:
        schema["enum"] = list(enum)
    return schema


def array(item, maximum=200):
    return {"type": "array", "items": item, "maxItems": maximum}


def obj(**properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


CITATION = obj(segment_id=string(100), quote=string(6000), occurrence={"type": "integer", "minimum": 0, "maximum": 1000})
REFS = array(CITATION, 12)
RANGE = array({"anyOf": [string(100), obj(segment_id=string(100), start_char={"type": "integer", "minimum": 0}, end_char={"type": "integer", "minimum": 0})]}, 200)
ISSUE_SCHEMA = obj(kind=string(enum=KINDS), severity=string(enum=("minor", "costly", "blocking")),
                   source=string(enum=("likely_speaker_issue", "possible_transcription_error")), evidence=REFS,
                   reviewed_range=RANGE, listener_consequence=string(600), corrective_pattern=string(600),
                   confidence=string(enum=("low", "medium", "high")), target=string(150))
CRITIC_SCHEMA = obj(issues=array(ISSUE_SCHEMA), structure_summary=string(1500))
LISTENER_ITEM = obj(text=string(700), evidence=REFS, reviewed_range=RANGE,
                    event=string(enum=("introduced", "defined", "related", "closed", "unresolved")))
LISTENER_SCHEMA = obj(**{field: array(LISTENER_ITEM, 100) for field in LISTENER_FIELDS})
CRITERION_SCHEMA = obj(id=string(100), status=string(enum=("assessed", "not_assessable")),
                       level={"anyOf": [{"type": "integer", "minimum": 0, "maximum": 2}, {"type": "null"}]},
                       explanation=string(600), evidence=REFS, reviewed_range=RANGE, reason=string(400), missing_evidence=string(400))


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_transcript(transcript: dict):
    text = transcript["text"]
    if text != "\n".join(segment["text"] for segment in transcript["segments"]):
        raise GymError("invalid_transcript", "Canonical transcript must retain every ordered segment joined with one newline.", status=422)
    ids = set()
    last = -1
    for segment in transcript["segments"]:
        start, end = segment["start_char"], segment["end_char"]
        if segment["id"] in ids or not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end <= len(text) or start < last or text[start:end] != segment["text"]:
            raise GymError("invalid_transcript", "Transcript segment offsets must exactly match canonical Unicode text.", status=422)
        ids.add(segment["id"])
        last = end


def resolve_evidence(citation: dict, transcript: dict) -> dict:
    """No model-supplied time or global offset is trusted. Occurrences overlap."""
    segment = next((s for s in transcript["segments"] if s["id"] == citation.get("segment_id")), None)
    quote, occurrence = citation.get("quote"), citation.get("occurrence")
    if segment is None or not isinstance(quote, str) or not quote or not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 0:
        raise ValueError("invalid citation fields")
    positions, position = [], 0
    while (position := segment["text"].find(quote, position)) >= 0:
        positions.append(position)
        position += 1
    if occurrence >= len(positions):
        raise ValueError("quote occurrence does not exist")
    start = segment["start_char"] + positions[occurrence]
    end = start + len(quote)
    if transcript["text"][start:end] != quote:
        raise ValueError("canonical quote mismatch")
    result = {"transcript_id": transcript["id"], "segment_id": segment["id"], "start_char": start, "end_char": end, "quote": quote, "occurrence": occurrence, "timing_kind": "none"}
    seg_start, seg_end = segment.get("start"), segment.get("end")
    valid_segment = _finite(seg_start) and _finite(seg_end) and 0 <= seg_start < seg_end
    duration = transcript.get("duration_seconds")
    if valid_segment and _finite(duration) and seg_end > duration:
        valid_segment = False
    words = [word for word in segment.get("words", []) if isinstance(word.get("start_char"), int) and isinstance(word.get("end_char"), int) and word["start_char"] < end and word["end_char"] > start]
    aligned = valid_segment and bool(words) and words[0]["start_char"] <= start and words[-1]["end_char"] >= end
    previous = seg_start
    for word in words:
        a, b = word.get("start"), word.get("end")
        if not (_finite(a) and _finite(b) and valid_segment and seg_start <= a < b <= seg_end and a >= previous and word.get("timing_origin", word.get("timing_kind")) == "aligned"):
            aligned = False
        if _finite(b):
            previous = b
    if aligned:
        result.update(start_s=words[0]["start"], end_s=words[-1]["end"], timing_kind="aligned")
    elif valid_segment:
        result.update(start_s=seg_start, end_s=seg_end, timing_kind="estimated")
    return result


def resolve_range(ids: list, transcript: dict) -> list[dict]:
    known = {s["id"]: s for s in transcript["segments"]}
    order = {s["id"]: index for index, s in enumerate(transcript["segments"])}
    if not ids:
        raise ValueError("absence requires an explicit reviewed range")
    resolved = []
    for item in ids:
        identifier = item if isinstance(item, str) else item["segment_id"]
        if identifier not in known:
            raise ValueError("unknown reviewed segment")
        segment = known[identifier]
        start = segment["start_char"] if isinstance(item, str) else item["start_char"]
        end = segment["end_char"] if isinstance(item, str) else item["end_char"]
        if not isinstance(start, int) or not isinstance(end, int) or not segment["start_char"] <= start < end <= segment["end_char"]:
            raise ValueError("reviewed range lies outside its source segment")
        resolved.append({"transcript_id": transcript["id"], "segment_id": identifier, "start_char": start, "end_char": end})
    resolved.sort(key=lambda ref: ref["start_char"])
    for left, right in zip(resolved, resolved[1:]):
        if left["segment_id"] == right["segment_id"]:
            if left["end_char"] != right["start_char"]:
                raise ValueError("reviewed range overlaps or has a gap")
        elif order[right["segment_id"]] != order[left["segment_id"]] + 1 or left["end_char"] != known[left["segment_id"]]["end_char"] or right["start_char"] != known[right["segment_id"]]["start_char"]:
            raise ValueError("reviewed range must be contiguous")
    return resolved


def uncertain_evidence(evidence: list[dict], transcript: dict) -> bool:
    segments = {s["id"]: s for s in transcript["segments"]}
    # Prefer adapter-mapped affected segments; only unmapped legacy edits make
    # the whole generation uncertain rather than guessing the removed region.
    diagnostics = transcript.get("diagnostics") or transcript.get("original", {}).get("diagnostics", {})
    if diagnostics.get("chunk_stitch_edits") and not any(s.get("stitch_edited") for s in segments.values()):
        return True
    for ref in evidence:
        segment = segments[ref["segment_id"]]
        if segment.get("uncertain") or segment.get("stitch_edited") or segment.get("possible_transcription_error"):
            return True
        for word in segment.get("words", []):
            if word.get("uncertain") and (word.get("start_char") is None or word["start_char"] < ref["end_char"] and word.get("end_char", ref["end_char"]) > ref["start_char"]):
                return True
    return False


def _resolve_item(item, transcript, *, require_evidence=False):
    item = dict(item)
    item["evidence"] = [resolve_evidence(ref, transcript) for ref in item["evidence"]]
    item["reviewed_range"] = resolve_range(item["reviewed_range"], transcript) if item.get("reviewed_range") else []
    if not item["evidence"] and (require_evidence or not item["reviewed_range"]):
        raise ValueError("claim requires spoken evidence or reviewed absence range")
    if item.get("kind") in {"missing_context", "unnamed_entity", "missing_cause", "missing_consequence", "missing_transition", "question_not_answered", "weak_call_to_action", "weak_proposal", "implicit_comparison", "ungrounded_term"} and not item["reviewed_range"]:
        raise ValueError("absence finding requires explicit reviewed coverage")
    return item


def build_windows(transcript: dict, maximum: int = 6000, overlap: int = 600) -> list[dict]:
    validate_transcript(transcript)
    if maximum < 64 or maximum > 6000 or not 0 <= overlap <= 600 or overlap >= maximum:
        raise ValueError("invalid window bounds")
    # Split overlong segments on lexical boundaries, retaining original IDs and
    # global offsets; evidence is always resolved against the whole revision.
    pieces = []
    for segment in transcript["segments"]:
        start, end = segment["start_char"], segment["end_char"]
        while end - start > maximum:
            boundary = transcript["text"].rfind(" ", start + maximum // 2, start + maximum)
            if boundary < start:
                boundary = start + maximum
            else:
                boundary += 1
            piece = dict(segment, text=transcript["text"][start:boundary], start_char=start, end_char=boundary)
            pieces.append(piece)
            start = boundary
        if start < end:
            pieces.append(dict(segment, text=transcript["text"][start:end], start_char=start, end_char=end))
    windows, index = [], 0
    while index < len(pieces):
        stop = index
        while stop < len(pieces) and pieces[stop]["end_char"] - pieces[index]["start_char"] <= maximum:
            stop += 1
        selected = pieces[index:stop]
        windows.append({"transcript_id": transcript["id"], "index": len(windows), "start_char": selected[0]["start_char"], "end_char": selected[-1]["end_char"], "segments": selected})
        if stop == len(pieces):
            break
        following = stop
        while following > index + 1 and pieces[stop - 1]["end_char"] - pieces[following - 1]["start_char"] <= overlap:
            following -= 1
        index = following
    return windows


def _window_citations(result: dict, window: dict, transcript: dict):
    """Restrict citations/ranges to the actual window, not unseen transcript."""
    def visit(value):
        if isinstance(value, dict):
            if {"segment_id", "quote", "occurrence"} <= value.keys():
                ref = resolve_evidence(value, transcript)
                if not window["start_char"] <= ref["start_char"] < ref["end_char"] <= window["end_char"]:
                    raise ValueError("citation outside reviewed window")
            for key, child in value.items():
                if key == "reviewed_range" and child:
                    for ref in resolve_range(child, transcript):
                        if not window["start_char"] <= ref["start_char"] < ref["end_char"] <= window["end_char"]:
                            raise ValueError("absence range was not fully reviewed")
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(result)


def validate_assessment(criterion: dict):
    if criterion["status"] == "not_assessable":
        if criterion["level"] is not None or not criterion.get("reason") or not criterion.get("missing_evidence"):
            raise ValueError("unassessable criteria need reason and missing evidence, not a numeric value")
    elif criterion["level"] not in (0, 1, 2) or isinstance(criterion["level"], bool) or not criterion["explanation"] or not (criterion["evidence"] or criterion["reviewed_range"]):
        raise ValueError("assessed criterion lacks grounded level")
    if criterion["status"] == "assessed" and criterion["level"] == 2 and (criterion["id"] in DIMENSIONS["delivery"] or criterion["id"] in {"few_unresolved_entities", "closed_branches"}) and not criterion["reviewed_range"]:
        raise ValueError("absence-of-obstruction assessment requires reviewed coverage")


def score_snapshot(criteria: list[dict], mode: str) -> dict:
    expected = {criterion: dimension for dimension, names in DIMENSIONS.items() for criterion in names}
    expected.update({criterion: "supplemental" for criterion in SUPPLEMENTAL.get(mode, ())})
    if len(criteria) != len(expected) or {criterion["id"] for criterion in criteria} != expected.keys():
        raise ValueError("exactly one assessment is required for every rubric criterion")
    for criterion in criteria:
        criterion["dimension"] = expected[criterion["id"]]
        validate_assessment(criterion)
    dimensions = {}
    for dimension in (*DIMENSIONS, *(("supplemental",) if mode in SUPPLEMENTAL else ())):
        levels = [criterion["level"] for criterion in criteria if criterion["dimension"] == dimension]
        dimensions[dimension] = 5 * math.floor((sum(levels) / 8 * 100 / 5) + .5) if len(levels) == 4 and all(level is not None for level in levels) else None
    return {"rubric_version": VERSION, "criteria": criteria, "dimensions": dimensions, "overall_score": None}


def select_feedback(issues: list[dict], listener: dict | None = None) -> list[dict]:
    severity = {"blocking": 0, "costly": 1, "minor": 2}
    structural = {"missing_context", "ambiguous_reference", "unnamed_entity", "unclear_relationship", "implicit_comparison", "buried_conclusion", "topic_jump", "missing_transition", "question_not_answered"}
    supported = [issue for issue in issues if issue["source"] == "likely_speaker_issue" and issue["confidence"] in {"medium", "high"} and (issue["evidence"] or issue.get("reviewed_range"))]
    supported.sort(key=lambda issue: (severity[issue["severity"]], 0 if issue["kind"] in structural else 2 if issue["kind"] in {"word_retrieval", "obstructive_filler", "repetition"} else 1, min((r["start_char"] for r in issue["evidence"] or issue["reviewed_range"]), default=0)))
    result, seen = [], set()
    for issue in supported:
        key = (issue["kind"], issue.get("target", "").casefold())
        if key not in seen:
            result.append(issue)
            seen.add(key)
        if len(result) == 3:
            break
    return result


def compact_model_input(value):
    """Remove redundant storage/timing metadata, never transcript or ledger facts."""
    if isinstance(value, list):
        return [compact_model_input(item) for item in value]
    if not isinstance(value, dict):
        return value
    if {"transcript_id", "segment_id", "quote", "start_char", "end_char"} <= value.keys():
        return {key: value[key] for key in ("transcript_id", "segment_id", "quote", "start_char", "end_char", "occurrence") if key in value}
    result = {}
    for key, item in value.items():
        if key in {"original", "words", "window_artifacts", "local_issues", "delivery_metrics", "input_hash", "created_at", "updated_at"}:
            continue
        if key == "text" and "segments" in value:
            continue
        if key == "reviewed_range" and isinstance(item, list):
            result[key] = [{name: ref[name] for name in ("segment_id", "start_char", "end_char")} if isinstance(ref, dict) else ref for ref in item]
        else:
            result[key] = compact_model_input(item)
    if result.get("first_window") == result.get("last_window") and "last_window" in result:
        result["last_window"] = {"same_as": "first_window"}
    return result


def share_citations(payload: dict) -> dict:
    """Encode repeated exact citations once, without shortening ledger facts."""
    citations, counts = {}, {}
    def collect(value):
        if isinstance(value, dict):
            if {"segment_id", "quote", "occurrence"} <= value.keys():
                key = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                citations[key] = value
                counts[key] = counts.get(key, 0) + 1
            else:
                for child in value.values():
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    collect(payload)
    names = {key: f"c{index}" for index, key in enumerate(key for key, count in counts.items() if count > 1)}
    if not names:
        return payload
    def replace(value):
        if isinstance(value, dict):
            key = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            if key in names:
                return {"citation_ref": names[key]}
            return {name: replace(child) for name, child in value.items()}
        if isinstance(value, list):
            return [replace(child) for child in value]
        return value
    return dict(replace(payload), citation_refs={name: citations[key] for key, name in names.items()})


class AnalysisEngine:
    def __init__(self, provider):
        self.provider = provider
        self.window_artifacts = []
        self._task_cache = {}
        self.last_failure = None

    def clear_cache(self):
        self._task_cache.clear()
        self.window_artifacts = []

    async def _call(self, task: str, payload: dict, schema: dict, validator=None):
        self.last_failure = {"task": task, "category": "pending"}
        payload = share_citations(compact_model_input(payload))
        prompt = (ROOT / "prompts" / "v1" / f"{task}.md").read_text(encoding="utf-8")
        if "citation_refs" in payload:
            prompt += "\nInput citation_ref aliases refer to the exact source citations in citation_refs. Read every referenced citation; return expanded {segment_id,quote,occurrence} evidence, never aliases. No fact or reviewed coverage has been omitted."
        original = payload
        profile = {"provider_class": type(self.provider).__name__, "model": getattr(self.provider, "model", None), "config": getattr(self.provider, "config", {}), "versions": VERSION}
        cache_key = hashlib.sha256(json.dumps([task, prompt, payload, schema, profile], sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        if task in {"communication_critic", "listener_simulator"} and cache_key in self._task_cache:
            cached = json.loads(self._task_cache[cache_key])
            validated = validator(cached) if validator else cached
            self.last_failure = None
            return validated
        for attempt in range(2):
            try:
                result = await self.provider.complete(task, prompt, payload, schema)
                validate_schema(result, schema)
                serialized = json.dumps(result, ensure_ascii=False, allow_nan=False)
                validated = validator(result) if validator else result
                if task in {"communication_critic", "listener_simulator"}:
                    if len(self._task_cache) >= 256:
                        self._task_cache.pop(next(iter(self._task_cache)))
                    self._task_cache[cache_key] = serialized
                self.last_failure = None
                return validated
            except GymError as error:
                self.last_failure = {"task": task, "category": error.code, "attempt": attempt + 1}
                if error.code != "invalid_model_output":
                    raise
            except (ValueError, KeyError, TypeError, IndexError) as error:
                # Only fixed validator messages, never model content, are persisted.
                safe_reasons = {"claim requires spoken evidence or reviewed absence range", "absence finding requires explicit reviewed coverage",
                                "missing listener relation requires reviewed coverage", "citation outside reviewed window",
                                "absence range was not fully reviewed", "quote occurrence does not exist", "invalid citation fields",
                                "unknown reviewed segment", "reviewed range lies outside its source segment",
                                "reviewed range overlaps or has a gap", "reviewed range must be contiguous",
                                "exactly one assessment is required for every rubric criterion", "score batch requires every requested criterion exactly once",
                                "assessed criterion lacks grounded level", "absence-of-obstruction assessment requires reviewed coverage",
                                "unassessable criteria need reason and missing evidence, not a numeric value"}
                reason = str(error)
                self.last_failure = {"task": task, "category": "evidence_or_schema_validation", "attempt": attempt + 1,
                                     "reason": reason if reason in safe_reasons else type(error).__name__}
            if attempt == 0:
                # Only safe fixed errors; never replay provider body, exception or
                # previous invalid model text into the repair request.
                payload = {"original_task": original, "schema_repair": "Previous response violated schema, exact citations, coverage or grounding. Recheck every field and cite only supplied exact spans.",
                           "validation_failure": self.last_failure}
        raise GymError("invalid_model_output", "Coach output failed schema/evidence validation twice. Saved transcript remains available.", status=502, retryable=True)

    def _provenance(self):
        metadata = getattr(self.provider, "last_metadata", {})
        config = getattr(self.provider, "config", {})
        return {"provider": metadata.get("provider", type(self.provider).__name__), "model": metadata.get("model", getattr(self.provider, "model", "unknown")),
                "rubric_version": VERSION, "prompt_version": VERSION, "schema_version": VERSION,
                "context_tokens": config.get("context_tokens", 16384), "temperature": 0, "maximum_output_tokens": 4096}

    async def analyze(self, transcript: dict, exercise: dict, language: str, progress=None) -> dict:
        try:
            return await self._analyze(transcript, exercise, language, progress)
        except TimeoutError:
            raise GymError("analysis_deadline", "The analysis exceeded its 30-minute deadline. Successful windows are retained for an explicit retry.", status=503, retryable=True) from None

    async def _analyze(self, transcript: dict, exercise: dict, language: str, progress=None) -> dict:
        if not transcript.get("text", "").strip():
            raise GymError("no_speech", "No speech detected; record again. Empty transcripts are not scored.", status=422)
        self.window_artifacts = []
        self.last_failure = None
        # Hindi/code-mixed windows may be shorter to respect a conservative
        # byte-token context bound; never drop the remainder.
        config = getattr(self.provider, "config", {})
        context = int(config.get("context_tokens", 16384))
        window_size = min(6000, max(128, (context - 9500) // 4))
        windows = build_windows(transcript, window_size, min(600, window_size // 5))
        issues, listener, structure = [], {}, []
        started = time.monotonic()
        async with asyncio.timeout(1800):
            for index, window in enumerate(windows):
                if progress:
                    response = progress("analysis_windows", index / len(windows))
                    if inspect.isawaitable(response):
                        await response
                def critic_validator(raw):
                    _window_citations(raw, window, transcript)
                    resolved = []
                    for issue in raw["issues"]:
                        issue = _resolve_item(issue, transcript)
                        issue["id"] = str(uuid4())
                        if uncertain_evidence(issue["evidence"] or issue["reviewed_range"], transcript):
                            issue["source"] = "possible_transcription_error"
                        resolved.append(issue)
                    return {"issues": resolved, "structure_summary": raw["structure_summary"]}
                critic = await self._call("communication_critic", {"pass": "local_window", "exercise": exercise, "output_language": language, "window": window, "heuristic_candidates": [c for c in candidates(transcript) if window["start_char"] <= c["start_char"] < window["end_char"]]}, CRITIC_SCHEMA, critic_validator)
                issues.extend(critic["issues"])
                structure.append({"window_index": index, "reviewed_start_char": window["start_char"], "reviewed_end_char": window["end_char"], "summary": critic["structure_summary"]})
                artifact = {"window_index": index, "start_char": window["start_char"], "end_char": window["end_char"], "critic": critic, "listener_status": "pending"}
                self.window_artifacts.append(artifact)
                def listener_validator(raw):
                    # Prior ledger facts can be returned with earlier citations;
                    # new facts must have exact supplied transcript evidence.
                    resolved = {}
                    for field in LISTENER_FIELDS:
                        resolved[field] = []
                        for item in raw[field]:
                            resolved_item = _resolve_item(item, transcript, require_evidence=True)
                            if field in {"missing_relationships", "ungrounded_terms"} and not resolved_item["reviewed_range"]:
                                raise ValueError("missing listener relation requires reviewed coverage")
                            known = any(old["text"] == resolved_item["text"] and old["evidence"] == resolved_item["evidence"] for old in listener.get(field, []))
                            if not known:
                                _window_citations({"item": item}, window, transcript)
                            resolved_item["id"] = str(uuid4())
                            if field in {"probably_understood", "required_assumptions"}:
                                resolved_item["inference"] = True
                            resolved[field].append(resolved_item)
                    return resolved
                listener_window = await self._call("listener_simulator", listener_input(window, listener, language), LISTENER_SCHEMA, listener_validator)
                listener = merge_listener(listener, listener_window)
                artifact.update(listener=listener_window, listener_status="succeeded")
                if len(issues) + len(structure) > 200:
                    raise GymError("analysis_capacity_exceeded", "Critic ledger exceeds 200 entries; all completed window artifacts remain available.", status=422)
            # A separate whole-session pass sees evidence ledger and boundaries,
            # not a silently truncated middle transcript.
            def global_critic_validator(raw):
                result = []
                for issue in raw["issues"]:
                    resolved = _resolve_item(issue, transcript)
                    resolved["id"] = str(uuid4())
                    if uncertain_evidence(resolved["evidence"] or resolved["reviewed_range"], transcript):
                        resolved["source"] = "possible_transcription_error"
                    result.append(resolved)
                return {"issues": result, "structure_summary": raw["structure_summary"]}
            global_critic = await self._call("communication_critic", {"pass": "session_structure_and_listener_confirmation", "exercise": exercise, "output_language": language, "complete_window_ledger": structure,
                "cited_issues": issues, "independent_listener": listener, "first_window": windows[0], "last_window": windows[-1]}, CRITIC_SCHEMA, global_critic_validator)
            # The global pass explicitly confirms local findings against listener
            # consequences, so only its validated findings feed feedback/scoring.
            confirmed = global_critic["issues"]
            if len(confirmed) > 200:
                raise GymError("analysis_capacity_exceeded", "Session event ledger exceeds capacity.", status=422)
            rubric = json.loads((ROOT / "rubrics" / "v1.json").read_text(encoding="utf-8"))
            mode = exercise.get("mode", "explain")
            def score_validator(raw):
                resolved = []
                for item in raw["criteria"]:
                    if item["status"] == "not_assessable":
                        item = dict(item, evidence=[], reviewed_range=[])
                    else:
                        item = _resolve_item(item, transcript)
                        if uncertain_evidence(item["evidence"] or item["reviewed_range"], transcript) and item["level"] < 2:
                            item.update(status="not_assessable", level=None, reason="Possible transcription error affects this assessment.", missing_evidence="Review the affected transcript/audio before scoring.")
                    validate_assessment(item)
                    resolved.append(item)
                return {"criteria": resolved, "answer_evidence": [resolve_evidence(ref, transcript) for ref in raw["answer_evidence"]]}
            required = [c for names in DIMENSIONS.values() for c in names] + list(SUPPLEMENTAL.get(mode, ()))
            all_criteria, answer_evidence = [], []
            # Bounded responses: 28/32 detailed assessments do not reliably fit
            # 4096 output tokens, especially in Hindi. No criterion is dropped.
            for offset in range(0, len(required), 8):
                batch = required[offset:offset + 8]
                batch_rubric = {key: value for key, value in rubric.items() if key not in {"dimensions", "supplemental"}}
                batch_rubric["criteria"] = {name: description for group in (*rubric["dimensions"].values(), rubric["supplemental"].get(mode, {}))
                                            for name, description in group.items() if name in batch}
                def batch_validator(raw):
                    ids = [item["id"] for item in raw["criteria"]]
                    if len(ids) != len(batch) or set(ids) != set(batch):
                        raise ValueError("score batch requires every requested criterion exactly once")
                    return score_validator(raw)
                batch_criterion = dict(CRITERION_SCHEMA, properties=dict(CRITERION_SCHEMA["properties"], id=string(100, batch)))
                batch_schema = obj(criteria=dict(array(batch_criterion, len(batch)), minItems=len(batch)), answer_evidence=REFS)
                result = await self._call("score_evaluator", {"exercise": exercise, "output_language": language, "rubric": batch_rubric,
                    "required_criterion_ids": batch, "complete_coverage": structure,
                    "confirmed_critic": [issue for issue in confirmed if issue["source"] == "likely_speaker_issue" and issue["confidence"] in {"medium", "high"}],
                    "listener": listener, "first_window": windows[0], "last_window": windows[-1]}, batch_schema, batch_validator)
                all_criteria.extend(result["criteria"])
                for ref in result["answer_evidence"]:
                    if ref not in answer_evidence:
                        answer_evidence.append(ref)
            scores = score_snapshot(all_criteria, mode)
            scores["answer_evidence"] = answer_evidence
        provenance = self._provenance()
        key = {"transcript_hash": hashlib.sha256(transcript["text"].encode()).hexdigest(), "transcript_id": transcript["id"], "exercise": exercise, "language": language, "provenance": provenance}
        return {"id": str(uuid4()), "transcript_id": transcript["id"], "status": "succeeded", "critic": {"issues": confirmed, "local_issues": issues, "structure_summary": global_critic["structure_summary"]},
                "listener": listener, "scores": scores, "feedback": select_feedback(confirmed, listener), "provenance": provenance,
                "coverage": {"complete": True, "windows": [{k: window[k] for k in ("index", "start_char", "end_char")} for window in windows], "character_count": len(transcript["text"])},
                "window_artifacts": self.window_artifacts, "delivery_metrics": delivery_metrics(transcript), "seconds": time.monotonic() - started,
                "input_hash": hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}

    async def generate_topic(self, mode: str, language: str) -> dict:
        schema = obj(title=string(120), prompt=string(1500), audience=string(200))
        def validator(result):
            if any(not result[key].strip() for key in ("title", "prompt", "audience")):
                raise ValueError("topic fields must be nonempty")
            return result
        result = await self._call("topic_generator", {"mode": mode, "language": language}, schema, validator)
        return dict(result, id=str(uuid4()), source="generated", mode=mode, language=language, provenance=self._provenance())

    async def question(self, original_analysis: dict, transcript: dict, exercise: dict, history: list) -> dict:
        if existing := open_question(history):
            return existing
        if len(history) >= 10:
            raise GymError("question_limit", "Ten follow-ups completed. Continue to the retry.", status=409)
        targets = unresolved_targets(original_analysis, history)
        if not targets:
            return {"available": False, "reason": "No unresolved grounded gaps remain; continue to retry."}
        schema = obj(question=string(500), target_ids=array(string(100), 5), expected_resolution=string(800))
        def validator(result):
            if not result["question"].strip() or not result["target_ids"] or set(result["target_ids"]) - set(targets):
                raise ValueError("question needs unresolved known targets")
            return result
        result = await self._call("question_generator", {"original_transcript": transcript, "exercise": exercise, "unresolved_target_ids": targets, "issues": original_analysis["critic"], "listener_gaps": {field: original_analysis["listener"].get(field, []) for field in ("ambiguities", "missing_relationships", "ungrounded_terms", "confusion_events")}, "previous_questions": history}, schema, validator)
        return dict(result, id=str(uuid4()), status="open", ordinal=len(history) + 1, available=True)

    async def evaluate_answer(self, question: dict, original: dict, answer: dict) -> dict:
        schema = obj(status=string(enum=("resolved", "partly_resolved", "unresolved", "uncertain_asr")), target_ids=array(string(100), 5), answer_evidence=REFS, new_gaps=array(string(500), 10), explanation=string(900), conversation_update=array(LISTENER_ITEM, 30))
        def validator(result):
            result["answer_evidence"] = [resolve_evidence(ref, answer) for ref in result["answer_evidence"]]
            if uncertain_evidence(result["answer_evidence"], answer):
                result["status"] = "uncertain_asr"
            validate_resolution(result, question)
            result["conversation_update"] = [_resolve_item(item, answer, require_evidence=True) for item in result["conversation_update"]]
            result["conversation_updated_listener"] = {"label": "Conversation-updated listener; original listener unchanged", "answer_transcript_id": answer["id"], "facts": result.pop("conversation_update")}
            return result
        return await self._call("answer_evaluator", {"original_transcript": original, "question": question, "answer_transcript": answer}, schema, validator)

    async def compare(self, initial_analysis: dict, retry_analysis: dict, initial: dict, retry: dict) -> dict:
        metrics = comparison_metrics(dict(initial, answer_evidence=initial_analysis.get("scores", {}).get("answer_evidence", [])), dict(retry, answer_evidence=retry_analysis.get("scores", {}).get("answer_evidence", [])))
        deltas = score_deltas(initial_analysis, retry_analysis)
        if not comparable(initial_analysis, retry_analysis):
            return {"id": str(uuid4()), "comparable": False, "metrics": metrics, "score_deltas": deltas, "issue_changes": [], "reason": deltas["reason"]}
        change = obj(issue_id=string(100), status=string(enum=("fixed", "improved", "still_present", "not_comparable")), initial_evidence=REFS, retry_evidence=REFS, explanation=string(700))
        schema = obj(issue_changes=array(change, 200), new_issue_ids=array(string(100), 200), listener_change=string(1200), remaining_focus=string(700))
        def validator(result):
            known = {issue["id"] for issue in initial_analysis["critic"]["issues"] if issue["source"] == "likely_speaker_issue"}
            returned = [item["issue_id"] for item in result["issue_changes"]]
            if len(returned) != len(set(returned)) or set(returned) != known:
                raise ValueError("every original supported issue needs one comparison")
            if set(result["new_issue_ids"]) - {issue["id"] for issue in retry_analysis["critic"]["issues"]}:
                raise ValueError("unknown retry issue")
            for item in result["issue_changes"]:
                item["initial_evidence"] = [resolve_evidence(ref, initial) for ref in item["initial_evidence"]]
                item["retry_evidence"] = [resolve_evidence(ref, retry) for ref in item["retry_evidence"]]
                if item["status"] != "not_comparable" and (not item["initial_evidence"] or not item["retry_evidence"]):
                    raise ValueError("comparison needs evidence from both attempts")
            return result
        result = await self._call("retry_evaluator", {"initial": initial, "retry": retry, "initial_analysis": initial_analysis, "retry_analysis": retry_analysis, "deterministic_metrics": metrics, "score_deltas": deltas}, schema, validator)
        return dict(result, id=str(uuid4()), comparable=True, metrics=metrics, score_deltas=deltas, initial_analysis_id=initial_analysis["id"], retry_analysis_id=retry_analysis["id"])

    async def model_answer(self, transcripts: list, language: str) -> dict:
        reference = obj(transcript_id=string(100), segment_id=string(100), quote=string(6000), occurrence={"type": "integer", "minimum": 0})
        statement = obj(text=string(1000), supporting_sources=array(reference, 12))
        schema = obj(statements=array(statement, 40), unresolved_information=array(string(500), 30))
        known = {transcript["id"]: transcript for transcript in transcripts}
        def validator(result):
            for item in result["statements"]:
                if not item["supporting_sources"]:
                    raise ValueError("example statement requires spoken source")
                item["supporting_sources"] = [resolve_evidence(ref, known[ref["transcript_id"]]) for ref in item["supporting_sources"]]
            result["text"] = " ".join(item["text"] for item in result["statements"])
            result["language"] = language
            return result
        return dict(await self._call("model_answer_generator", {"transcripts": transcripts, "output_language": language}, schema, validator), id=str(uuid4()), provenance=self._provenance())

    async def profile_summary(self, aggregates: dict) -> dict:
        schema = obj(statements=array(obj(text=string(500), aggregate_ids=array(string(150), 10)), 10), micro_drill=string(500), drill_aggregate_id=string(150))
        ids = {item["id"] for group in aggregates.get("groups", []) for item in group.get("patterns", []) + group.get("strengths", [])}
        def validator(result):
            for statement in result["statements"]:
                if not statement["aggregate_ids"] or set(statement["aggregate_ids"]) - ids:
                    raise ValueError("profile prose requires supplied aggregate IDs")
            if result["drill_aggregate_id"] not in ids:
                raise ValueError("micro-drill requires an aggregate")
            return result
        if not ids:
            return {"statements": [], "micro_drill": "", "drill_aggregate_id": "", "computed_only": True}
        return await self._call("personal_profile_updater", {"computed_aggregates": aggregates}, schema, validator)
