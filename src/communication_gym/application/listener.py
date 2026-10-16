"""Listener input deliberately excludes exercise, preparation, critic and profile."""
from __future__ import annotations

LISTENER_FIELDS = ("clearly_understood", "probably_understood", "ambiguities", "missing_relationships", "ungrounded_terms", "required_assumptions", "confusion_events", "entities", "relationships")


def listener_input(window: dict, previous_ledger: dict | None, language: str) -> dict:
    return {"output_language": language, "transcript": {"id": window["transcript_id"], "segments": [
        {"id": s["id"], "text": s["text"], "start_char": s["start_char"], "end_char": s["end_char"]}
        for s in window["segments"]]}, "previous_cited_listener_ledger": previous_ledger or {}}


def merge_listener(previous: dict, current: dict) -> dict:
    result = {}
    for field in LISTENER_FIELDS:
        items = list(previous.get(field, []))
        for item in current.get(field, []):
            signature = (item["text"], tuple((ref["transcript_id"], ref["start_char"], ref["end_char"]) for ref in item["evidence"]))
            if not any((old["text"], tuple((ref["transcript_id"], ref["start_char"], ref["end_char"]) for ref in old["evidence"])) == signature for old in items):
                items.append(item)
        result[field] = items
    if sum(map(len, result.values())) > 200:
        from communication_gym.domain.models import GymError
        raise GymError("analysis_capacity_exceeded", "Listener ledger exceeds 200 grounded entries; retained windows remain available.", status=422)
    return result
