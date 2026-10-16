import pytest

from communication_gym.application.analysis import AnalysisEngine, select_feedback, uncertain_evidence
from communication_gym.domain.models import GymError


def issue(kind, severity="costly", **changes):
    result = {"id": kind, "kind": kind, "severity": severity, "confidence": "high", "source": "likely_speaker_issue", "evidence": [{"start_char": 5, "end_char": 8}], "reviewed_range": [], "target": kind}
    result.update(changes)
    return result


def test_no_more_than_three_distinct_supported_patterns():
    values = [issue("obstructive_filler", "minor"), issue("missing_context", "blocking"), issue("unclear_relationship"), issue("ambiguous_reference"), issue("missing_context", "blocking")]
    feedback = select_feedback(values)
    assert len(feedback) == 3
    assert feedback[0]["kind"] == "missing_context"
    assert "obstructive_filler" not in {value["kind"] for value in feedback}


def test_uncertainty_low_confidence_and_empty_result_never_invent_criticism():
    assert select_feedback([]) == []
    assert select_feedback([issue("repetition", source="possible_transcription_error"), issue("missing_context", confidence="low")]) == []
    transcript = {"segments": [{"id": "s", "uncertain": True}], "diagnostics": {}}
    assert uncertain_evidence([{"segment_id": "s"}], transcript)


@pytest.mark.asyncio
async def test_one_semantic_repair_then_durable_invalid_output():
    class InvalidProvider:
        def __init__(self):
            self.calls = []
        async def complete(self, task, system, payload, schema):
            self.calls.append((system, payload))
            return {"title": "injection", "prompt": "p", "audience": "a", "extra": "not allowed"}
    provider = InvalidProvider()
    with pytest.raises(GymError) as caught:
        await AnalysisEngine(provider).generate_topic("compare", "en")
    assert caught.value.code == "invalid_model_output"
    assert len(provider.calls) == 2
    assert "schema_repair" in provider.calls[1][1]
    assert "extra" not in str(provider.calls[1][1])


@pytest.mark.asyncio
async def test_transport_failure_is_not_replayed_or_billed_twice():
    class BrokenProvider:
        calls = 0
        async def complete(self, *args):
            self.calls += 1
            raise GymError("coach_transport_error", "Unavailable", status=503)
    provider = BrokenProvider()
    with pytest.raises(GymError):
        await AnalysisEngine(provider).generate_topic("compare", "en")
    assert provider.calls == 1

@pytest.mark.asyncio
async def test_window_cache_requires_exact_transcript_and_analysis_profile():
    class CountingProvider:
        model = "model"
        config = {"context_tokens": 16384}
        calls = 0
        async def complete(self, *args):
            self.calls += 1
            return {"ok": True}
    provider = CountingProvider()
    engine = AnalysisEngine(provider)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
    payload = {"transcript_id": "first", "text": "same private text"}
    await engine._call("communication_critic", payload, schema)
    await engine._call("communication_critic", payload, schema)
    assert provider.calls == 1
    await engine._call("communication_critic", dict(payload, transcript_id="second"), schema)
    assert provider.calls == 2
    provider.model = "changed"
    await engine._call("communication_critic", payload, schema)
    assert provider.calls == 3
    provider.config = {"context_tokens": 32768}
    await engine._call("communication_critic", payload, schema)
    assert provider.calls == 4
    engine.clear_cache()
    assert engine.window_artifacts == []
    await engine._call("communication_critic", payload, schema)
    assert provider.calls == 5
