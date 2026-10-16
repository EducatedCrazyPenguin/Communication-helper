import copy
from pathlib import Path
from uuid import uuid4

import pytest

from communication_gym.domain.models import GymError
from communication_gym.providers.stt import SpeechProvider, _scrub_environment, build_transcript


def payload():
    return {"model": "Qwen/Qwen3-ASR-1.7B", "device": "cpu", "language": "mixed", "duration_seconds": 6,
            "diagnostics": {}, "segments": [
                {"start": 0, "end": 3, "text": "🙂  हाँ मतलब मतलब", "words": [
                    {"text": "🙂", "start": 0, "end": 0.2, "timing_origin": "estimated", "confidence_kind": "unavailable"},
                    {"text": "हाँ", "start": 0.2, "end": 1, "timing_origin": "aligned", "confidence_kind": "alignment_score"},
                    {"text": "मतलब", "start": 1, "end": 2, "timing_origin": "estimated"},
                    {"text": "मतलब", "start": 2, "end": 3, "timing_origin": "estimated"}]},
                {"start": 3, "end": 6, "text": "  I— I mean um um  ", "words": []}],
            "raw_segments": [{"start": 0, "end": 3, "text": "  🙂  हाँ मतलब मतलब  ", "words": []}]}


def test_unicode_offsets_preserve_repeated_words_and_whitespace():
    result = payload()
    original = copy.deepcopy(result)
    transcript = build_transcript(result, str(uuid4()))
    assert transcript["text"] == "🙂  हाँ मतलब मतलब\n  I— I mean um um  "
    assert transcript["original"] == original == result
    for segment in transcript["segments"]:
        assert transcript["text"][segment["start_char"]:segment["end_char"]] == segment["text"]
        for word in segment["words"]:
            assert transcript["text"][word["start_char"]:word["end_char"]] == word["text"]
    repeated = transcript["segments"][0]["words"][-2:]
    assert repeated[0]["end_char"] < repeated[1]["start_char"]
    assert transcript["segments"][0]["words"][1]["timing_kind"] == "aligned"
    result["raw_segments"][0]["text"] = "changed caller object"
    assert transcript["original"]["raw_segments"][0]["text"] == original["raw_segments"][0]["text"]


@pytest.mark.parametrize("start,end", [(float("nan"), 1), (0, float("inf")), (-1, 1), (1, 1), (0, 9)])
def test_invalid_word_timing_is_unavailable_without_rewriting_original(start, end):
    result = payload()
    result["segments"][0]["words"][0].update(start=start, end=end)
    transcript = build_transcript(result, str(uuid4()))
    word = transcript["segments"][0]["words"][0]
    assert word["start"] is None and word["end"] is None
    assert word["timing_kind"] == "none"
    assert result["segments"][0]["words"][0]["end"] == end


def test_backwards_and_unmatched_words_do_not_gain_precise_evidence():
    result = payload()
    result["segments"][0]["words"][1].update(start=0.1, end=0.5)
    result["segments"][0]["words"][2]["text"] = "invented"
    transcript = build_transcript(result, str(uuid4()))
    words = transcript["segments"][0]["words"]
    assert words[1]["start"] is None
    assert "start_char" not in words[2] and "end_char" not in words[2]


def test_stitch_edits_are_visible_uncertainty_not_recognition_confidence():
    result = payload()
    result["diagnostics"]["chunk_stitch_edits"] = [{"source_chunk_index": 0, "removed_word_count": 2,
        "removed_prefix_text": "हाँ मतलब ", "overlap_start_s": 0, "overlap_end_s": 1}]
    transcript = build_transcript(result, str(uuid4()))
    assert transcript["segments"][0]["stitch_edited"] is True
    assert transcript["segments"][0]["uncertain"] is True
    assert transcript["original"]["raw_segments"][0]["text"] == result["raw_segments"][0]["text"]


def test_unknown_provenance_is_not_promoted_to_alignment():
    result = payload()
    del result["segments"][0]["words"][0]["timing_origin"]
    word = build_transcript(result, str(uuid4()))["segments"][0]["words"][0]
    assert word["timing_origin"] == "unknown" and word["timing_kind"] == "none"


def test_environment_excludes_tokens_and_python_hooks(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "synthetic-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-secret")
    monkeypatch.setenv("PYTHONPATH", "untrusted")
    monkeypatch.setenv("HTTP_PROXY", "http://untrusted.invalid")
    environment = _scrub_environment()
    assert not {"HF_TOKEN", "GEMINI_API_KEY", "PYTHONPATH", "HTTP_PROXY"} & environment.keys()
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"


def test_storage_paths_cannot_escape_root(tmp_path):
    provider = SpeechProvider({}, tmp_path)
    with pytest.raises(GymError, match="Gym-owned storage"):
        provider._path(tmp_path.parent / "outside.wav")


@pytest.mark.asyncio
async def test_missing_configuration_is_truthfully_unavailable(tmp_path):
    provider = SpeechProvider({}, tmp_path)
    capabilities = await provider.capabilities()
    assert capabilities["available"] is False
    assert provider.pid is None
    await provider.close()


@pytest.mark.parametrize("index", [None, '{"weight_map": []}',
    '{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}'])
def test_worker_preflight_rejects_incomplete_shards(tmp_path, index):
    from communication_gym.worker.main import _complete_qwen
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json",
                 "model-00001-of-00002.safetensors"):
        (tmp_path / name).touch()
    if index is not None:
        (tmp_path / "model.safetensors.index.json").write_text(index, encoding="utf-8")
    assert _complete_qwen(tmp_path) is False
