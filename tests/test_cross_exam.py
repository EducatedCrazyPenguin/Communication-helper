import copy
import pytest

from communication_gym.application.analysis import AnalysisEngine
from communication_gym.application.cross_exam import open_question, unresolved_targets, validate_resolution


def test_one_open_question_and_replay_returns_same_record():
    history = [{"id": "q", "status": "open"}]
    assert open_question(history) is history[0]
    with pytest.raises(Exception) as error:
        open_question(history * 2)
    assert error.value.code == "invalid_question_state"


@pytest.mark.asyncio
async def test_question_replay_does_not_dispatch_provider():
    class NeverProvider:
        async def complete(self, *args):
            raise AssertionError("A replay must not trigger model work")
    question = {"id": "q", "status": "open"}
    assert await AnalysisEngine(NeverProvider()).question({}, {}, {}, [question]) == question


def test_partial_unresolved_skipped_do_not_clear_target():
    analysis = {"feedback": [{"id": "gap", "source": "likely_speaker_issue"}], "listener": {}}
    for status, resolution in (("answered", "partly_resolved"), ("answered", "unresolved"), ("skipped", "resolved")):
        history = [{"status": status, "resolution": {"status": resolution, "target_ids": ["gap"]}}]
        assert unresolved_targets(analysis, history) == ["gap"]
    assert unresolved_targets(analysis, [{"status": "answered", "resolution": {"status": "resolved", "target_ids": ["gap"]}}]) == []


def test_resolution_requires_affirmative_answer_and_known_targets():
    question = {"target_ids": ["gap"]}
    with pytest.raises(ValueError):
        validate_resolution({"status": "resolved", "target_ids": ["gap"], "answer_evidence": []}, question)
    with pytest.raises(ValueError):
        validate_resolution({"status": "resolved", "target_ids": ["other"], "answer_evidence": [{}]}, question)


@pytest.mark.asyncio
async def test_answer_cannot_use_original_evidence_and_original_remains_unchanged():
    class MisgroundedProvider:
        async def complete(self, *args):
            return {"status": "resolved", "target_ids": ["gap"], "answer_evidence": [{"segment_id": "original-segment", "quote": "Battery A", "occurrence": 0}], "new_gaps": [], "explanation": "Resolved", "conversation_update": []}
    original = {"id": "o", "text": "Battery A", "segments": [{"id": "original-segment", "text": "Battery A", "start_char": 0, "end_char": 9}]}
    answer = {"id": "a", "text": "Yes", "segments": [{"id": "answer-segment", "text": "Yes", "start_char": 0, "end_char": 3}]}
    saved = copy.deepcopy(original)
    with pytest.raises(Exception) as error:
        await AnalysisEngine(MisgroundedProvider()).evaluate_answer({"target_ids": ["gap"]}, original, answer)
    assert error.value.code == "invalid_model_output"
    assert original == saved

@pytest.mark.parametrize("quote", ["Yes.", "हाँ।", "Haan", "जी"])
def test_acknowledgement_alone_is_not_gap_resolution(quote):
    with pytest.raises(ValueError):
        validate_resolution({"status": "resolved", "target_ids": ["gap"], "answer_evidence": [{"quote": quote}]}, {"target_ids": ["gap"]})
