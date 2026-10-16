"""Optional external VoiceStudio HTTP; no model installation or voice cloning."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import struct
from uuid import uuid4

import httpx

from communication_gym.domain.models import GymError
from communication_gym.providers.openai_compatible import checked_endpoint, read_json

MAX_AUDIO = 256 * 1024 * 1024


class VoiceStudioProvider:
    def __init__(self, config: dict, root: Path | None = None):
        self.config = dict(config)
        self.base_url = config.get("base_url", "http://127.0.0.1:3900")
        self.root = Path(root).resolve() if root is not None else None
        self._client = httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=httpx.Timeout(180, connect=2))
        self._verified = None
        self._voices = []
        self._draining = False

    async def capabilities(self) -> dict:
        return {"provider_id": "voicestudio", "model_id": "tts-1", "available": bool(self._verified) and not self._draining,
                "locality": "local", "languages": ["en", "hi"], "hinglish_verified": False, "structured_output": False,
                "word_timing": "none", "word_confidence": False, "cancellation": "client_only", "release_supported": False,
                "backend_version": self._verified, "status": "draining" if self._draining else "verified" if self._verified else "not_probed",
                "unavailable_reason": "External synthesis may still be draining; explicitly confirm service idle." if self._draining else None if self._verified else "Enable and explicitly probe the optional external backend."}

    @staticmethod
    def _resolve(schema: dict, document: dict) -> dict:
        for _ in range(12):
            if "$ref" not in schema:
                return schema
            if not schema["$ref"].startswith("#/"):
                raise ValueError("external schema reference")
            value = document
            for key in schema["$ref"][2:].split("/"):
                value = value[key.replace("~1", "/").replace("~0", "~")]
            schema = value
        raise ValueError("recursive schema")

    async def probe(self) -> dict:
        self._verified = None
        if not self.config.get("enabled", False):
            return dict(await self.capabilities(), status="disabled")
        self.base_url, _ = checked_endpoint(self.base_url)
        try:
            async with self._client.stream("GET", self.base_url + "/health", timeout=2) as response:
                headers = dict(response.headers)
                health = await read_json(response)
            async with self._client.stream("GET", self.base_url + "/openapi.json", timeout=2) as response:
                api = await read_json(response)
            identity = " ".join(str(value) for value in (health.get("service", ""), health.get("name", ""), api.get("info", {}).get("title", ""), headers.get("server", ""))).lower()
            version = health.get("version") or headers.get("x-voicestudio-version") or api.get("info", {}).get("version")
            status = health.get("status")
            ready = health.get("ready") is True or status in {"ok", "healthy", "ready"}
            if "voicestudio" not in identity or not version or not ready:
                raise ValueError("identity or readiness")
            paths = api["paths"]
            paths["/profiles"]["get"]
            operation = paths["/v1/audio/speech"]["post"]
            request = self._resolve(operation["requestBody"]["content"]["application/json"]["schema"], api)
            properties = request["properties"]
            required = {"model", "input", "voice", "response_format", "language", "speed"}
            if not required <= properties.keys():
                raise ValueError("missing speech request fields")
            model = self._resolve(properties["model"], api)
            response_format = self._resolve(properties["response_format"], api)
            # A string field without an enum does not prove support for the alias.
            documented = json.dumps(model).lower() + str(operation.get("description", "")).lower()
            if "tts-1" not in documented or "enum" in response_format and "wav" not in response_format["enum"]:
                raise ValueError("unsupported OpenAI alias or WAV")
            self._verified = str(version)
            await self.list_voices()
            return await self.capabilities()
        except (httpx.HTTPError, GymError):
            self._verified = None
            return dict(await self.capabilities(), status="unavailable", unavailable_reason="VoiceStudio is absent, starting, or unavailable. Training remains usable.")
        except (ValueError, KeyError, TypeError):
            self._verified = None
            return dict(await self.capabilities(), status="incompatible", unavailable_reason="Backend identity/version or required speech API could not be verified.")

    async def list_voices(self) -> list[dict]:
        if not self._verified:
            raise GymError("tts_not_verified", "Probe the configured VoiceStudio backend first.", status=503)
        try:
            async with self._client.stream("GET", self.base_url + "/profiles", timeout=2) as response:
                profiles = await read_json(response, expected_type=list)
            self._voices = [{"id": str(item["id"]), "name": str(item.get("name", item["id"])), "kind": item.get("kind", "unknown"), "languages": item.get("languages", []),
                             "authorized": self.config.get("authorized_profiles", {}).get(str(item["id"])) == self._verified}
                            for item in profiles if isinstance(item, dict) and item.get("id")]
            return self._voices
        except (httpx.HTTPError, GymError, ValueError, KeyError, TypeError):
            raise GymError("tts_profiles_unavailable", "Voice profiles could not be read. No voice was selected automatically.", status=503, retryable=True) from None

    async def synthesize(self, request: dict) -> dict:
        if not self._verified or not self.config.get("enabled"):
            raise GymError("tts_not_verified", "Enable and probe the optional VoiceStudio backend.", status=503)
        if self._draining:
            raise GymError("tts_draining", "Confirm that external synthesis is idle before more GPU work.", status=409)
        text = request.get("text", "")
        if not isinstance(text, str) or not text.strip() or len(text) > 4096:
            raise GymError("tts_text_limit", "Select a nonempty excerpt of at most 4096 characters for playback.", status=422)
        # Reconfirm backend identity/version before using saved voice permission.
        # This is health/schema work only, never implicit speech generation.
        if not (await self.probe())["available"]:
            raise GymError("tts_service_unavailable", "Optional playback backend is no longer ready or compatible. Training is unchanged.", status=503, retryable=True)
        profile = request.get("profile_id")
        if not profile or self.config.get("authorized_profiles", {}).get(profile) != self._verified:
            raise GymError("voice_permission_required", "Explicitly confirm that this is your voice or you have permission for this profile and backend version.", status=403)
        if profile not in {voice["id"] for voice in self._voices}:
            raise GymError("invalid_voice", "Selected authorized voice is no longer available.", status=422)
        if self.root is None:
            raise GymError("storage_unavailable", "Configure the Gym-owned storage root for playback.", status=503)
        payload = {"model": "tts-1", "input": text, "voice": profile, "response_format": "wav", "speed": 1.0}
        if request.get("language") in {"en", "hi"}:
            payload["language"] = request["language"]
        elif request.get("language") != "hinglish":
            raise GymError("invalid_language", "Select English, Hindi or Hinglish.", status=422)
        identifier = str(uuid4())
        directory = self.root / "data" / "tts"
        if not directory.resolve().is_relative_to(self.root):
            raise GymError("unsafe_path", "Playback storage escapes the owned root.", status=403)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / f"{identifier}.partial"
        destination = directory / f"{identifier}.wav"
        sent = False
        completed = False
        try:
            async with asyncio.timeout(180):
                sent = True
                async with self._client.stream("POST", self.base_url + "/v1/audio/speech", json=payload) as response:
                    response.raise_for_status()
                    if response.headers.get("content-type", "").split(";")[0].lower() not in {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}:
                        raise GymError("invalid_tts_audio", "Backend did not return WAV audio.", status=502)
                    size, digest, header = 0, hashlib.sha256(), bytearray()
                    with temporary.open("xb") as handle:
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > MAX_AUDIO:
                                raise GymError("tts_audio_too_large", "Playback exceeds the 256 MiB limit.", status=502)
                            if len(header) < 44:
                                header.extend(chunk[:44 - len(header)])
                            digest.update(chunk)
                            handle.write(chunk)
            completed = True
            if size < 44 or header[:4] != b"RIFF" or header[8:12] != b"WAVE" or struct.unpack("<I", header[4:8])[0] + 8 > size:
                raise GymError("invalid_tts_audio", "Backend audio has an invalid WAV container.", status=502)
            import wave
            try:
                with wave.open(str(temporary), "rb") as wav:
                    if wav.getnframes() <= 0 or wav.getframerate() <= 0:
                        raise ValueError()
                    duration = wav.getnframes() / wav.getframerate()
                    sample_rate = wav.getframerate()
                    expected_bytes = wav.getnframes() * wav.getnchannels() * wav.getsampwidth()
                    if len(wav.readframes(wav.getnframes())) != expected_bytes:
                        raise ValueError()
            except (wave.Error, EOFError, ValueError):
                raise GymError("invalid_tts_audio", "Backend returned an invalid or empty PCM WAV.", status=502) from None
            temporary.replace(destination)
            return {"id": identifier, "path": destination.relative_to(self.root).as_posix(), "media_type": "audio/wav", "sha256": digest.hexdigest(), "size": size,
                    "duration_seconds": duration, "sample_rate": sample_rate, "provenance": "tts", "text": text, "profile_id": profile, "backend_version": self._verified,
                    "language": request["language"], "hinglish_verified": False}
        except asyncio.CancelledError:
            if sent:
                self._draining = True
            raise
        except (TimeoutError, httpx.TimeoutException):
            if sent:
                self._draining = True
            raise GymError("tts_timeout", "Playback timed out; external synthesis may still be draining. Training is unchanged.", status=503, retryable=True) from None
        except GymError:
            if sent and not completed:
                self._draining = True
            raise
        except httpx.HTTPError:
            if sent:
                self._draining = True
            raise GymError("tts_service_error", "External playback failed; synthesis may still be draining. Training is unchanged.", status=503, retryable=True) from None
        finally:
            temporary.unlink(missing_ok=True)

    async def release(self):
        if self._draining:
            raise GymError("tts_draining", "External synthesis may still be running. Explicitly confirm service idle before switching GPU work, or use CPU.", status=409)

    async def confirm_idle(self):
        self._draining = False

    async def close(self):
        await self._client.aclose()
