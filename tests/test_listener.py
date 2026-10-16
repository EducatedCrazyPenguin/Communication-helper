import copy
import pytest

from communication_gym.application.analysis import _resolve_item
from communication_gym.application.listener import LISTENER_FIELDS, listener_input, merge_listener


def test_listener_request_cannot_leak_hidden_prep_exercise_or_critic():
    segment = {"id": "s", "text": "Battery A exists.", "start_char": 0, "end_char": 17}
    window = {"transcript_id": "t", "segments": [segment], "prep": ["A is backup"], "exercise": {"expected": "A is backup"}, "critic": {"secret": "A is backup"}, "profile": {"secret": "A is backup"}}
    first = listener_input(window, {}, "en")
    window.update(prep=["A is main"], exercise={"expected": "A is main"}, critic={"secret": "A is main"})
    assert listener_input(window, {}, "en") == first
    assert set(first) == {"output_language", "transcript", "previous_cited_listener_ledger"}
    assert set(first["transcript"]) == {"id", "segments"}
    assert "backup" not in str(first)


def test_clear_understanding_requires_spoken_citation_not_reviewed_absence():
    transcript = {"id": "t", "text": "Battery A exists.", "segments": [{"id": "s", "text": "Battery A exists.", "start_char": 0, "end_char": 17}]}
    assertion = {"text": "Battery A is backup", "evidence": [], "reviewed_range": ["s"], "event": "defined"}
    with pytest.raises(ValueError):
        _resolve_item(assertion, transcript, require_evidence=True)
    assertion["evidence"] = [{"segment_id": "s", "quote": "Battery A is backup", "occurrence": 0}]
    with pytest.raises(ValueError):
        _resolve_item(assertion, transcript, require_evidence=True)


def test_incremental_ledger_deduplicates_overlap_without_losing_evidence():
    ref = {"transcript_id": "t", "segment_id": "s", "start_char": 0, "end_char": 9, "quote": "Battery A"}
    fact = {"id": "first", "text": "Battery A is introduced", "evidence": [ref], "event": "introduced"}
    earlier = {field: [] for field in LISTENER_FIELDS}
    earlier["entities"] = [fact]
    current = copy.deepcopy(earlier)
    current["entities"][0]["id"] = "second"
    merged = merge_listener(earlier, current)
    assert merged["entities"] == [fact]
    assert earlier["entities"] == [fact]


def test_capacity_failure_retains_input_ledger_without_silent_dropping():
    earlier = {field: [] for field in LISTENER_FIELDS}
    earlier["entities"] = [{"text": str(i), "evidence": [{"transcript_id": "t", "start_char": i, "end_char": i+1}]} for i in range(200)]
    with pytest.raises(Exception) as error:
        merge_listener(earlier, {"entities": [{"text": "overflow", "evidence": [{"transcript_id": "t", "start_char": 200, "end_char": 201}]}]})
    assert error.value.code == "analysis_capacity_exceeded"
    assert len(earlier["entities"]) == 200
