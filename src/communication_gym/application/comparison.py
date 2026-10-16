"""Descriptive compression and criterion deltas, never an invented overall grade."""
from __future__ import annotations

import math
import re


def _duration(attempt: dict):
    value = attempt.get("duration_seconds", attempt.get("captured_duration"))
    return value if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0 else None


def answer_position(transcript: dict) -> dict:
    words = list(re.finditer(r"\S+", transcript.get("text", "")))
    spans = transcript.get("answer_evidence", [])
    spans = [span for span in spans if span.get("transcript_id") == transcript.get("id") and isinstance(span.get("start_char"), int)
             and transcript.get("text", "")[span["start_char"]:span.get("end_char", -1)] == span.get("quote")]
    if not words or not spans:
        return {"word_fraction": None, "seconds": None, "timing_kind": "unavailable"}
    first = min(spans, key=lambda span: span["start_char"])
    index = sum(word.end() <= first["start_char"] for word in words)
    return {"word_fraction": index / len(words), "seconds": first.get("start_s") if first.get("timing_kind") == "aligned" else None,
            "timing_kind": "aligned" if first.get("timing_kind") == "aligned" else "text_only"}


def comparison_metrics(initial: dict, retry: dict) -> dict:
    first_count, retry_count = len(initial.get("text", "").split()), len(retry.get("text", "").split())
    first_duration, retry_duration = _duration(initial), _duration(retry)
    first_position, retry_position = answer_position(initial), answer_position(retry)
    return {"initial_duration_seconds": first_duration, "retry_duration_seconds": retry_duration,
            "duration_ratio": retry_duration / first_duration if first_duration and retry_duration else None,
            "initial_word_count": first_count, "retry_word_count": retry_count,
            "word_reduction_fraction": 1 - retry_count / first_count if first_count and retry_count else None,
            "initial_approximate_wpm": first_count * 60 / first_duration if first_duration and first_count else None,
            "retry_approximate_wpm": retry_count * 60 / retry_duration if retry_duration and retry_count else None,
            "initial_answer_position": first_position, "retry_answer_position": retry_position,
            "answer_position_shift": retry_position["word_fraction"] - first_position["word_fraction"] if first_position["word_fraction"] is not None and retry_position["word_fraction"] is not None else None,
            "approximate": True, "interpretation": "Compression is descriptive; improvement requires retained listener clarity."}


def comparable(initial_analysis: dict, retry_analysis: dict) -> bool:
    keys = ("provider", "model", "rubric_version", "prompt_version", "schema_version", "context_tokens")
    first, second = initial_analysis.get("provenance", {}), retry_analysis.get("provenance", {})
    return all(first.get(key) is not None and first.get(key) == second.get(key) for key in keys)


def score_deltas(initial_analysis: dict, retry_analysis: dict) -> dict:
    if not comparable(initial_analysis, retry_analysis):
        return {"comparable": False, "dimensions": {}, "criteria": {}, "reason": "Reanalyze both attempts under the same complete analysis profile."}
    first, second = initial_analysis["scores"], retry_analysis["scores"]
    dims = {key: second["dimensions"][key] - value if value is not None and second["dimensions"].get(key) is not None else None for key, value in first["dimensions"].items()}
    old = {item["id"]: item.get("level") for item in first["criteria"]}
    criteria = {item["id"]: item["level"] - old[item["id"]] if item.get("level") is not None and old.get(item["id"]) is not None else None for item in second["criteria"]}
    return {"comparable": True, "dimensions": dims, "criteria": criteria}
