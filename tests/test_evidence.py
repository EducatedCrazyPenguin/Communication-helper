import pytest

from communication_gym.application.analysis import build_windows, resolve_evidence, resolve_range, validate_transcript


def transcript(text="🙂 यह यह é बात है।"):
    return {"id": "transcript", "text": text, "segments": [{"id": "segment", "text": text, "start_char": 0, "end_char": len(text), "start": 0, "end": 5, "words": []}]}


def test_exact_occurrence_uses_unicode_code_points():
    source = transcript()
    result = resolve_evidence({"segment_id": "segment", "quote": "यह", "occurrence": 1}, source)
    assert result["start_char"] == 5
    assert source["text"][result["start_char"]:result["end_char"]] == "यह"
    assert result["timing_kind"] == "estimated"
    from communication_gym.application.analysis import compact_model_input
    assert result["occurrence"] == 1
    assert compact_model_input(result)["occurrence"] == 1
    combined = resolve_evidence({"segment_id": "segment", "quote": "é", "occurrence": 0}, source)
    assert combined["end_char"] - combined["start_char"] == 2


@pytest.mark.parametrize("citation", [
    {"segment_id": "missing", "quote": "यह", "occurrence": 0},
    {"segment_id": "segment", "quote": "यह", "occurrence": 2},
    {"segment_id": "segment", "quote": "ये", "occurrence": 0},
    {"segment_id": "segment", "quote": "", "occurrence": 0},
    {"segment_id": "segment", "quote": "यह", "occurrence": -1},
    {"segment_id": "segment", "quote": "यह", "occurrence": True},
])
def test_invalid_ids_quotes_and_occurrences_rejected(citation):
    with pytest.raises(ValueError):
        resolve_evidence(citation, transcript())


def test_model_timestamps_never_become_evidence_timestamps():
    source = transcript()
    citation = {"segment_id": "segment", "quote": "यह", "occurrence": 0, "start_s": 999}
    assert resolve_evidence(citation, source)["start_s"] == 0
    source["segments"][0].update(start=float("nan"), end=0)
    result = resolve_evidence(citation, source)
    assert result["timing_kind"] == "none"
    assert "start_s" not in result


def test_only_exact_aligned_word_coverage_is_precise():
    source = transcript("one two")
    source["segments"][0]["words"] = [{"start_char": 0, "end_char": 3, "start": 0, "end": 1, "timing_origin": "aligned"}, {"start_char": 4, "end_char": 7, "start": 1, "end": 2, "timing_origin": "estimated"}]
    first = resolve_evidence({"segment_id": "segment", "quote": "one", "occurrence": 0}, source)
    both = resolve_evidence({"segment_id": "segment", "quote": "one two", "occurrence": 0}, source)
    assert first["timing_kind"] == "aligned"
    assert both["timing_kind"] == "estimated"
    source["segments"][0]["words"][0]["end"] = 100
    assert resolve_evidence({"segment_id": "segment", "quote": "one", "occurrence": 0}, source)["timing_kind"] == "estimated"


def test_absence_requires_explicit_contiguous_review():
    source = {"id": "t", "text": "a\nb\nc", "segments": [{"id": str(i), "text": char, "start_char": i * 2, "end_char": i * 2 + 1} for i, char in enumerate("abc")]}
    assert len(resolve_range(["0", "1"], source)) == 2
    for ids in ([], ["missing"], ["0", "2"], ["0", "0"]):
        with pytest.raises(ValueError):
            resolve_range(ids, source)


def test_windows_cover_long_unicode_transcript_without_truncation():
    source = transcript(("बैटरी A और B 🙂 " * 1000).rstrip())
    windows = build_windows(source)
    covered = set()
    for window in windows:
        assert window["end_char"] - window["start_char"] <= 6000
        covered.update(range(window["start_char"], window["end_char"]))
        for segment in window["segments"]:
            assert source["text"][segment["start_char"]:segment["end_char"]] == segment["text"]
    assert covered == set(range(len(source["text"])))


def test_invalid_canonical_offsets_rejected():
    source = transcript()
    source["segments"][0]["start_char"] = 1
    with pytest.raises(Exception) as error:
        validate_transcript(source)
    assert error.value.code == "invalid_transcript"

def test_absence_claim_cannot_hide_behind_an_unrelated_quote():
    from communication_gym.application.analysis import _resolve_item
    source = transcript("A exists.")
    with pytest.raises(ValueError):
        _resolve_item({"kind": "missing_context", "evidence": [{"segment_id": "segment", "quote": "A exists.", "occurrence": 0}], "reviewed_range": []}, source)


def test_partial_segment_range_cannot_claim_unseen_window_coverage():
    from communication_gym.application.analysis import _window_citations
    source = transcript("one two three")
    window = {"start_char": 0, "end_char": 7}
    _window_citations({"reviewed_range": [{"segment_id": "segment", "start_char": 0, "end_char": 7}]}, window, source)
    with pytest.raises(ValueError):
        _window_citations({"reviewed_range": ["segment"]}, window, source)


@pytest.mark.asyncio
async def test_denied_remote_coach_sends_no_http_request(monkeypatch):
    import httpx
    import socket
    from communication_gym.providers.openai_compatible import OpenAICompatibleProvider
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.1", 443))])
    calls = []
    item = OpenAICompatibleProvider({"base_url": "https://coach.invalid", "model": "model", "allow_cloud": False})
    await item._client.aclose()
    item._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: calls.append(request)), trust_env=False)
    try:
        with pytest.raises(Exception) as error:
            await item.complete("task", "system", {}, {"type": "object"})
        assert error.value.code == "cloud_consent_required"
        assert calls == []
    finally:
        await item.close()


@pytest.mark.asyncio
async def test_changed_model_invalidates_cloud_session_consent(monkeypatch):
    import socket
    from communication_gym.providers.openai_compatible import OpenAICompatibleProvider
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.1", 443))])
    item = OpenAICompatibleProvider({"base_url": "https://coach.invalid", "model": "new", "allow_cloud": True, "session_consent": {"approved": True, "endpoint": "https://coach.invalid", "model": "old"}})
    try:
        with pytest.raises(Exception) as error:
            await item.complete("task", "system", {}, {"type": "object"})
        assert error.value.code == "cloud_consent_required"
    finally:
        await item.close()


def test_shared_citation_encoding_retains_all_facts_and_reviewed_coverage():
    from communication_gym.application.analysis import share_citations
    ref = {"transcript_id": "t", "segment_id": "s", "quote": "यह यह", "occurrence": 1, "start_char": 4, "end_char": 9}
    payload = {"issues": [{"kind": "ambiguous_reference", "evidence": [ref], "reviewed_range": ["s"]}],
               "listener": {"ambiguities": [{"text": "Which one?", "evidence": [dict(ref)], "reviewed_range": ["s"]}]}}
    encoded = share_citations(payload)
    assert len(encoded["citation_refs"]) == 1
    def expand(value):
        if isinstance(value, dict):
            if set(value) == {"citation_ref"}:
                return encoded["citation_refs"][value["citation_ref"]]
            return {key: expand(child) for key, child in value.items()}
        if isinstance(value, list):
            return [expand(child) for child in value]
        return value
    assert expand({key: value for key, value in encoded.items() if key != "citation_refs"}) == payload
    assert payload["issues"][0]["evidence"] == [ref]
