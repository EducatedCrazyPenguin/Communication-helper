import copy
import pytest

from communication_gym.application.analysis import DIMENSIONS, SUPPLEMENTAL, score_snapshot
from communication_gym.application.comparison import score_deltas
from communication_gym.providers.openai_compatible import validate_schema


def assessments(level=2, mode="explain"):
    ids = [name for names in DIMENSIONS.values() for name in names] + list(SUPPLEMENTAL.get(mode, ()))
    return [{"id": name, "status": "assessed", "level": level, "explanation": "Explicit in reviewed speech.", "evidence": [], "reviewed_range": [{"segment_id": "s", "start_char": 0, "end_char": 10}], "reason": "", "missing_evidence": ""} for name in ids]


@pytest.mark.parametrize("levels,expected", [([0, 0, 0, 0], 0), ([2, 2, 2, 2], 100), ([1, 1, 1, 1], 50), ([0, 0, 0, 1], 15), ([2, 2, 2, 1], 90)])
def test_coarse_score_formula(levels, expected):
    values = assessments()
    for value, level in zip(values[:4], levels):
        value["level"] = level
    snapshot = score_snapshot(values, "explain")
    assert snapshot["dimensions"]["context"] == expected
    assert snapshot["overall_score"] is None
    assert snapshot["dimensions"]["delivery"] == 100


def test_unavailable_is_not_zero_and_requires_reason():
    values = assessments(0)
    values[0].update(status="not_assessable", level=None, reason="ASR uncertainty", missing_evidence="Reviewed audio")
    result = score_snapshot(values, "explain")
    assert result["dimensions"]["context"] is None
    assert result["dimensions"]["clarity"] == 0
    values[0]["missing_evidence"] = ""
    with pytest.raises(ValueError):
        score_snapshot(values, "explain")


def test_every_criterion_exactly_once_and_supplemental_separate():
    values = assessments(mode="compare")
    assert score_snapshot(values, "compare")["dimensions"]["supplemental"] == 100
    values[-1] = copy.deepcopy(values[0])
    with pytest.raises(ValueError):
        score_snapshot(values, "compare")


def test_nonfinite_extra_field_and_boolean_levels_rejected():
    schema = {"type": "object", "properties": {"score": {"type": "number", "minimum": 0, "maximum": 100}}, "required": ["score"], "additionalProperties": False}
    for value in ({"score": float("nan")}, {"score": float("inf")}, {"score": True}, {"score": 5, "injected": "ignore schema"}):
        with pytest.raises(ValueError):
            validate_schema(value, schema)
    values = assessments()
    values[0]["level"] = True
    with pytest.raises(ValueError):
        score_snapshot(values, "explain")


def test_version_mismatch_refuses_deltas():
    provenance = {"provider": "a", "model": "b", "rubric_version": "1", "prompt_version": "1", "schema_version": "1", "context_tokens": 16384}
    first = {"scores": score_snapshot(assessments(1), "explain"), "provenance": provenance}
    second = copy.deepcopy(first)
    second["provenance"]["model"] = "other"
    assert score_deltas(first, second)["comparable"] is False

@pytest.mark.asyncio
async def test_managed_coach_capabilities_never_start_models_or_server(monkeypatch):
    from communication_gym.providers.ollama import OllamaProvider
    from communication_gym.providers.processes import OwnedProcess
    async def forbidden(*args, **kwargs):
        raise AssertionError("Settings must not launch a model server")
    monkeypatch.setattr(OwnedProcess, "start", forbidden)
    item = OllamaProvider({"managed": True})
    try:
        result = await item.capabilities()
        assert result["available"] is False
        assert result["schema_probe_passed"] is False
        assert result["real_calibration_passed"] is False
    finally:
        await item.close()


@pytest.mark.asyncio
async def test_missing_local_model_never_pulls_or_dispatches_inference():
    import httpx
    from communication_gym.providers.ollama import OllamaProvider
    calls = []
    def transport(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"version": "synthetic"} if request.url.path == "/api/version" else {"models": []})
    item = OllamaProvider({"managed": False, "model": "not-cached"})
    await item._client.aclose()
    item._client = httpx.AsyncClient(transport=httpx.MockTransport(transport), trust_env=False)
    try:
        with pytest.raises(Exception) as error:
            await item.complete("task", "system", {}, {"type": "object"})
        assert error.value.code == "model_not_cached"
        assert calls == ["/api/version", "/api/tags"]
    finally:
        await item.close()


def test_listener_wire_schema_reuses_nodes_without_relaxing_constraints():
    import json
    from communication_gym.application.analysis import LISTENER_SCHEMA
    from communication_gym.application.listener import LISTENER_FIELDS
    from communication_gym.providers.openai_compatible import compact_schema
    wire = compact_schema(LISTENER_SCHEMA)
    assert len(json.dumps(wire)) < len(json.dumps(LISTENER_SCHEMA)) // 2
    good = {field: [] for field in LISTENER_FIELDS}
    good["entities"] = [{"text": "A", "evidence": [{"segment_id": "s", "quote": "A", "occurrence": 0}],
                         "reviewed_range": ["s"], "event": "introduced"}]
    for schema in (wire, LISTENER_SCHEMA):
        validate_schema(good, schema)
        for change in ({"event": "invented"}, {"extra": True}, {"evidence": [{"segment_id": "s", "quote": "A", "occurrence": True}]}):
            bad = copy.deepcopy(good)
            bad["entities"][0].update(change)
            with pytest.raises(ValueError):
                validate_schema(bad, schema)


@pytest.mark.asyncio
async def test_scoring_batches_preserve_every_criterion_and_full_coverage():
    from communication_gym.application.analysis import AnalysisEngine
    from communication_gym.application.listener import LISTENER_FIELDS
    calls = []
    class Provider:
        config = {"context_tokens": 16384}
        async def complete(self, task, system, payload, schema):
            calls.append((task, payload))
            if task == "communication_critic":
                return {"issues": [], "structure_summary": "A is defined."}
            if task == "listener_simulator":
                return {field: [] for field in LISTENER_FIELDS}
            assert task == "score_evaluator"
            return {"criteria": [{"id": name, "status": "assessed", "level": 2, "explanation": "Explicit in reviewed speech.",
                                  "evidence": [{"segment_id": "s", "quote": "A is defined.", "occurrence": 0}],
                                  "reviewed_range": ["s"], "reason": "", "missing_evidence": ""}
                                 for name in payload["required_criterion_ids"]], "answer_evidence": []}
    source = {"id": "t", "text": "A is defined.", "segments": [{"id": "s", "text": "A is defined.", "start_char": 0, "end_char": 13}]}
    result = await AnalysisEngine(Provider()).analyze(source, {"mode": "compare"}, "hi")
    batches = [payload for task, payload in calls if task == "score_evaluator"]
    requested = [name for payload in batches for name in payload["required_criterion_ids"]]
    expected = [name for names in DIMENSIONS.values() for name in names] + list(SUPPLEMENTAL["compare"])
    assert requested == expected
    assert all(0 < len(payload["required_criterion_ids"]) <= 8 for payload in batches)
    assert all(payload["complete_coverage"] and payload["first_window"]["segments"][0]["text"] == source["text"] for payload in batches)
    assert [item["id"] for item in result["scores"]["criteria"]] == expected
    assert result["coverage"]["complete"] is True


@pytest.mark.asyncio
async def test_real_listener_request_schema_fits_16384_context_without_truncation():
    import json
    from pathlib import Path
    import httpx
    from communication_gym.application.analysis import LISTENER_SCHEMA
    from communication_gym.application.listener import LISTENER_FIELDS
    from communication_gym.providers.ollama import OllamaProvider
    responses = {field: [] for field in LISTENER_FIELDS}
    requests = []
    def transport(request):
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "synthetic"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "cached"}]})
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": json.dumps(responses)}, "done_reason": "stop", "prompt_eval_count": 100, "eval_count": 20})
    item = OllamaProvider({"managed": False, "model": "cached", "context_tokens": 16384})
    await item._client.aclose()
    item._client = httpx.AsyncClient(transport=httpx.MockTransport(transport), trust_env=False)
    prompt = (Path(__file__).parents[1] / "src/communication_gym/prompts/v1/listener_simulator.md").read_text(encoding="utf-8")
    payload = {"output_language": "hi", "transcript": {"id": "t", "segments": [{"id": "s", "text": "दो बैटरियां हैं।", "start_char": 0, "end_char": len("दो बैटरियां हैं।")}]},
               "previous_cited_listener_ledger": {}}
    try:
        assert await item.complete("listener_simulator", prompt, payload, LISTENER_SCHEMA) == responses
        assert requests[0]["options"]["num_ctx"] == 16384
        assert requests[0]["options"]["num_predict"] == 4096
        assert json.loads(requests[0]["messages"][1]["content"]) == payload
        assert "$defs" in requests[0]["format"]
        assert item.last_metadata["input_token_upper_bound"] <= 16384 - 4096
    finally:
        await item.close()
