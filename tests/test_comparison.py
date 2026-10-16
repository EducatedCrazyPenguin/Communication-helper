import copy
import pytest

from communication_gym.application.analysis import AnalysisEngine
from communication_gym.application.comparison import answer_position, comparison_metrics, score_deltas


def provenance():
    return {"provider": "local", "model": "model", "rubric_version": "1", "prompt_version": "1", "schema_version": "1", "context_tokens": 16384}


def test_compression_can_accompany_clear_regression():
    metrics = comparison_metrics({"text": "a b c d", "duration_seconds": 60}, {"text": "a b", "duration_seconds": 30})
    assert metrics["duration_ratio"] == .5
    assert metrics["word_reduction_fraction"] == .5
    first = {"provenance": provenance(), "scores": {"dimensions": {"clarity": 100}, "criteria": [{"id": "relations_explicit", "level": 2}]}}
    second = {"provenance": provenance(), "scores": {"dimensions": {"clarity": 25}, "criteria": [{"id": "relations_explicit", "level": 0}]}}
    assert score_deltas(first, second)["dimensions"]["clarity"] == -75
    assert "improvement" not in metrics


def test_empty_or_zero_duration_metrics_are_null():
    metrics = comparison_metrics({"text": "", "duration_seconds": 0}, {"text": "a", "duration_seconds": float("inf")})
    assert metrics["duration_ratio"] is None
    assert metrics["word_reduction_fraction"] is None
    assert metrics["initial_approximate_wpm"] is None
    assert metrics["retry_approximate_wpm"] is None


def test_answer_position_requires_exact_span_and_trustworthy_timing():
    transcript = {"id": "t", "text": "detail first answer"}
    transcript["answer_evidence"] = [{"transcript_id": "t", "quote": "answer", "start_char": 13, "end_char": 19, "start_s": 8, "timing_kind": "estimated"}]
    result = answer_position(transcript)
    assert result["word_fraction"] == 2 / 3
    assert result["seconds"] is None
    transcript["answer_evidence"][0]["quote"] = "invented"
    assert answer_position(transcript)["word_fraction"] is None


@pytest.mark.asyncio
async def test_fixed_claim_requires_both_attempts_not_cross_exam_evidence():
    class InvalidComparison:
        async def complete(self, *args):
            return {"issue_changes": [{"issue_id": "gap", "status": "fixed", "initial_evidence": [], "retry_evidence": [], "explanation": "Shorter"}], "new_issue_ids": [], "listener_change": "", "remaining_focus": ""}
    analysis = {"id": "analysis", "provenance": provenance(), "critic": {"issues": [{"id": "gap", "source": "likely_speaker_issue"}]}, "scores": {"dimensions": {}, "criteria": []}}
    transcript = {"id": "t", "text": "a", "segments": [{"id": "s", "text": "a", "start_char": 0, "end_char": 1}]}
    with pytest.raises(Exception) as error:
        await AnalysisEngine(InvalidComparison()).compare(analysis, copy.deepcopy(analysis), transcript, transcript)
    assert error.value.code == "invalid_model_output"


@pytest.mark.asyncio
async def test_model_answer_rejects_unsourced_statement():
    class Unsourced:
        async def complete(self, *args):
            return {"statements": [{"text": "The capacity is 5000 mAh.", "supporting_sources": []}], "unresolved_information": []}
    with pytest.raises(Exception) as error:
        await AnalysisEngine(Unsourced()).model_answer([], "en")
    assert error.value.code == "invalid_model_output"
