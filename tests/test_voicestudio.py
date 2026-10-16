import asyncio
import io
import wave

import httpx
import pytest

from communication_gym.domain.models import GymError
from communication_gym.providers.voicestudio import VoiceStudioProvider


def api_document():
    return {"info": {"title": "VoiceStudio", "version": "1.2"}, "paths": {"/profiles": {"get": {}}, "/v1/audio/speech": {"post": {"requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"model": {"type": "string", "enum": ["tts-1"]}, "input": {"type": "string"}, "voice": {"type": "string"}, "response_format": {"type": "string", "enum": ["wav"]}, "language": {"type": "string"}, "speed": {"type": "number"}}}}}}}}}}


def wav_bytes():
    stream = io.BytesIO()
    with wave.open(stream, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 1600)
    return stream.getvalue()


def handler(request):
    if request.url.path == "/health":
        return httpx.Response(200, json={"service": "VoiceStudio", "version": "1.2", "status": "ready"})
    if request.url.path == "/openapi.json":
        return httpx.Response(200, json=api_document())
    if request.url.path == "/profiles":
        return httpx.Response(200, json=[{"id": "voice", "name": "Owned voice", "kind": "clone"}])
    if request.url.path == "/v1/audio/speech":
        return httpx.Response(200, content=wav_bytes(), headers={"content-type": "audio/wav"})
    raise AssertionError("Unexpected request")


async def provider(tmp_path, callback=handler):
    result = VoiceStudioProvider({"enabled": True, "authorized_profiles": {"voice": "1.2"}}, tmp_path)
    await result._client.aclose()
    result._client = httpx.AsyncClient(transport=httpx.MockTransport(callback), trust_env=False)
    return result


@pytest.mark.asyncio
async def test_absent_or_starting_service_never_claims_available(tmp_path):
    item = await provider(tmp_path, lambda request: httpx.Response(503, json={"status": "starting"}))
    try:
        result = await item.probe()
        assert result["status"] == "unavailable"
        assert result["available"] is False
    finally:
        await item.close()


@pytest.mark.asyncio
async def test_unverified_api_or_alias_is_incompatible(tmp_path):
    def incompatible(request):
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json={"info": {"title": "VoiceStudio", "version": "0.5"}, "paths": {}})
        return handler(request)
    item = await provider(tmp_path, incompatible)
    try:
        assert (await item.probe())["status"] == "incompatible"
    finally:
        await item.close()


@pytest.mark.asyncio
async def test_explicit_profile_permission_and_input_limit_before_synthesis(tmp_path):
    requested = []
    def recording(request):
        requested.append(request.url.path)
        return handler(request)
    item = await provider(tmp_path, recording)
    try:
        await item.probe()
        item.config["authorized_profiles"] = {}
        with pytest.raises(GymError) as error:
            await item.synthesize({"text": "Hello", "language": "en", "profile_id": "voice"})
        assert error.value.code == "voice_permission_required"
        item.config["authorized_profiles"] = {"voice": "1.2"}
        with pytest.raises(GymError) as error:
            await item.synthesize({"text": "x" * 4097, "language": "en", "profile_id": "voice"})
        assert error.value.code == "tts_text_limit"
        assert "/v1/audio/speech" not in requested
    finally:
        await item.close()


@pytest.mark.asyncio
async def test_valid_audio_is_managed_and_hinglish_is_not_translated(tmp_path):
    import json
    payloads = []
    def recording(request):
        if request.url.path == "/v1/audio/speech":
            payloads.append(json.loads(request.content))
        return handler(request)
    item = await provider(tmp_path, recording)
    try:
        assert (await item.probe())["available"]
        result = await item.synthesize({"text": "Battery A main hai.", "language": "hinglish", "profile_id": "voice"})
        assert (tmp_path / result["path"]).read_bytes() == wav_bytes()
        assert not (tmp_path / result["path"]).with_suffix(".partial").exists()
        assert result["duration_seconds"] == .1
        assert payloads[0]["input"] == "Battery A main hai."
        assert "language" not in payloads[0]
        assert result["hinglish_verified"] is False
    finally:
        await item.close()


@pytest.mark.asyncio
async def test_invalid_wav_bytes_never_become_saved_playback(tmp_path):
    def invalid(request):
        if request.url.path == "/v1/audio/speech":
            return httpx.Response(200, content=b"not really audio", headers={"content-type": "audio/wav"})
        return handler(request)
    item = await provider(tmp_path, invalid)
    try:
        await item.probe()
        with pytest.raises(GymError) as error:
            await item.synthesize({"text": "Hello", "language": "en", "profile_id": "voice"})
        assert error.value.code == "invalid_tts_audio"
        assert list(tmp_path.rglob("*.wav")) == []
        assert list(tmp_path.rglob("*.partial")) == []
    finally:
        await item.close()


@pytest.mark.asyncio
async def test_client_cancellation_truthfully_blocks_gpu_switch_until_idle(tmp_path):
    reached = asyncio.Event()
    async def delayed(request):
        if request.url.path == "/v1/audio/speech":
            reached.set()
            await asyncio.Event().wait()
        return handler(request)
    item = await provider(tmp_path, delayed)
    try:
        await item.probe()
        task = asyncio.create_task(item.synthesize({"text": "Hello", "language": "en", "profile_id": "voice"}))
        await reached.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(GymError) as error:
            await item.release()
        assert error.value.code == "tts_draining"
        assert (await item.capabilities())["cancellation"] == "client_only"
        await item.confirm_idle()
        await item.release()
    finally:
        await item.close()

@pytest.mark.asyncio
async def test_generation_timeout_is_safe_and_marks_external_service_draining(tmp_path):
    def timed_out(request):
        if request.url.path == "/v1/audio/speech":
            raise httpx.ReadTimeout("synthetic provider body must not be logged", request=request)
        return handler(request)
    item = await provider(tmp_path, timed_out)
    try:
        await item.probe()
        with pytest.raises(GymError) as error:
            await item.synthesize({"text": "Hello", "language": "en", "profile_id": "voice"})
        assert error.value.code == "tts_timeout"
        assert "synthetic provider body" not in str(error.value)
        assert (await item.capabilities())["status"] == "draining"
    finally:
        await item.close()


@pytest.mark.asyncio
async def test_audio_size_bound_removes_partial_artifact(tmp_path, monkeypatch):
    import communication_gym.providers.voicestudio as module
    monkeypatch.setattr(module, "MAX_AUDIO", 20)
    item = await provider(tmp_path)
    try:
        await item.probe()
        with pytest.raises(GymError) as error:
            await item.synthesize({"text": "Hello", "language": "en", "profile_id": "voice"})
        assert error.value.code == "tts_audio_too_large"
        assert not list(tmp_path.rglob("*.wav"))
        assert not list(tmp_path.rglob("*.partial"))
    finally:
        await item.close()

@pytest.mark.asyncio
async def test_backend_version_change_requires_renewed_voice_permission(tmp_path):
    version = ["1.2"]
    speech_calls = []
    def changing(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"service": "VoiceStudio", "version": version[0], "status": "ready"})
        if request.url.path == "/v1/audio/speech":
            speech_calls.append(request)
        return handler(request)
    item = await provider(tmp_path, changing)
    try:
        await item.probe()
        version[0] = "1.3"
        with pytest.raises(GymError) as error:
            await item.synthesize({"text": "Hello", "language": "en", "profile_id": "voice"})
        assert error.value.code == "voice_permission_required"
        assert speech_calls == []
    finally:
        await item.close()
