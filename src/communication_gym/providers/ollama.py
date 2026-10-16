"""Lazy, contained Ollama ownership using existing cached weights only."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import time

import httpx

from communication_gym.domain.models import GymError
from communication_gym.providers.openai_compatible import checked_endpoint, compact_schema, read_json, validate_schema
from communication_gym.providers.processes import OwnedProcess


class OllamaProvider:
    def __init__(self, config: dict):
        self.config = dict(config)
        self.model = config.get("model", "qwen3.5:9b-q8_0")
        self.base_url = config.get("base_url", "http://127.0.0.1:11435")
        self.managed = config.get("managed", True)
        self._owner = None
        self._start_lock = asyncio.Lock()
        self._client = httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=httpx.Timeout(300, connect=2))
        self.last_metadata = {}
        self._schema_passed = False
        self._think_supported = False
        self._release_failed = False

    async def _ensure_server(self):
        self.base_url, local = checked_endpoint(self.base_url)
        if not local:
            raise GymError("invalid_endpoint", "Ollama must use a loopback endpoint.", status=422)
        if not self.managed:
            return
        async with self._start_lock:
            if self._owner and self._owner.returncode is None:
                return
            executable = self.config.get("executable") or shutil.which("ollama")
            if not executable:
                raise GymError("ollama_not_installed", "Install or configure the existing Ollama executable; no models will be downloaded.", status=503)
            from urllib.parse import urlsplit
            parsed = urlsplit(self.base_url)
            port = parsed.port or 11435
            # A busy port belongs to someone else. Never attach to or stop it.
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError:
                    probe.bind(("127.0.0.1", 0))
                    port = probe.getsockname()[1]
            self.base_url = f"http://127.0.0.1:{port}"
            environment = {k: v for k, v in os.environ.items() if not any(secret in k.upper() for secret in ("TOKEN", "API_KEY", "SECRET", "PASSWORD"))}
            environment.update(OLLAMA_HOST=f"127.0.0.1:{port}", OLLAMA_NO_CLOUD="1", OLLAMA_MAX_LOADED_MODELS="1", OLLAMA_NUM_PARALLEL="1")
            self._owner = await OwnedProcess.start([str(executable), "serve"], env=environment, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            for _ in range(100):
                if self._owner.returncode is not None:
                    break
                try:
                    response = await self._client.get(self.base_url + "/api/version", timeout=1)
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(.1)
            await self._owner.terminate()
            self._owner = None
            raise GymError("ollama_start_failed", "Owned Ollama server did not become ready.", status=503, retryable=True)

    async def capabilities(self) -> dict:
        # Passive: opening Settings must not start a model or model-server process.
        result = {"provider_id": "ollama", "model_id": self.model, "available": False, "locality": "local",
                  "languages": ["en", "hi", "hinglish"], "structured_output": self._schema_passed,
                  "word_timing": "none", "word_confidence": False, "cancellation": "terminate_owned" if self.managed else "client_only",
                  "release_supported": True, "endpoint_reachable": False, "model_installed": False,
                  "schema_probe_passed": self._schema_passed, "real_calibration_passed": False, "actual_endpoint": self.base_url}
        try:
            checked_endpoint(self.base_url)
            # Never query another service occupying our configured managed port.
            if self.managed and self._owner is None:
                result["unavailable_reason"] = "Managed Ollama is configured and starts on explicit coach work. Cached model is checked before inference."
                return result
            async with self._client.stream("GET", self.base_url + "/api/version", timeout=2) as response:
                version = await read_json(response)
            async with self._client.stream("GET", self.base_url + "/api/tags", timeout=2) as response:
                tags = await read_json(response)
            result.update(endpoint_reachable=True, backend_version=version.get("version"))
            result["model_installed"] = any((item.get("name") == self.model or item.get("model") == self.model) and not item.get("remote_host") and not item.get("remote_model") and "cloud" not in self.model.lower() for item in tags.get("models", []))
            result["available"] = result["model_installed"]
            if not result["available"]:
                result["unavailable_reason"] = "Selected model is not cached. No automatic pull is allowed."
        except (httpx.HTTPError, GymError):
            result["unavailable_reason"] = "Local Ollama endpoint is unavailable."
        return result

    async def probe(self) -> dict:
        await self._ensure_server()
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
        self._think_supported = False
        self._schema_passed = False
        if not (await self.capabilities())["model_installed"]:
            raise GymError("model_not_cached", "Selected model is not cached; probe did not attempt a download.", status=503)
        try:
            # Probe the optional flag explicitly before relying on it in real
            # tasks. An unsupported-field response is not an inference retry.
            async with self._client.stream("POST", self.base_url + "/api/chat", json={"model": self.model, "stream": False, "think": False, "format": schema, "messages": [{"role": "user", "content": "Return {\"ok\":true}"}], "options": {"temperature": 0, "num_ctx": int(self.config.get("context_tokens", 16384)), "num_predict": 128}}) as response:
                if response.status_code in {400, 422}:
                    unsupported = True
                    body = None
                else:
                    unsupported = False
                    body = await read_json(response)
            if unsupported:
                value = await self.complete("schema_probe", "Return the requested JSON object.", {"ok": True}, schema)
            else:
                value = json.loads(body["message"]["content"])
                validate_schema(value, schema)
                self._think_supported = value == {"ok": True}
            self._schema_passed = value == {"ok": True}
        except asyncio.CancelledError:
            if self._owner:
                await self._owner.terminate()
                self._owner = None
            raise
        except (httpx.HTTPError, GymError, ValueError, KeyError, TypeError, RecursionError):
            raise GymError("schema_probe_failed", "Explicit local model schema/optional-flag probe failed.", status=503, retryable=True) from None
        return await self.capabilities()

    async def complete(self, task: str, system: str, payload: dict, schema: dict) -> dict:
        if self._release_failed:
            raise GymError("model_release_failed", "Close the owned coach process before starting new GPU work.", status=409)
        await self._ensure_server()
        capabilities = await self.capabilities()
        if not capabilities["model_installed"]:
            raise GymError("model_not_cached", "Selected Ollama model is not present locally. No download was attempted.", status=503)
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        context = int(self.config.get("context_tokens", 16384))
        wire_schema = compact_schema(schema)
        input_bytes = len((system + encoded + json.dumps(wire_schema, ensure_ascii=False, separators=(",", ":"))).encode("utf-8")) + 256
        if input_bytes > context - 4096:
            self.last_metadata = {"provider": "ollama", "model": self.model, "task": task, "input_token_upper_bound": input_bytes, "context_tokens": context}
            raise GymError("context_capacity_exceeded", "Coach input exceeds the configured safe context capacity.", status=422)
        request = {"model": self.model, "stream": False, "format": wire_schema, "keep_alive": "10m",
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": encoded}],
                   "options": {"temperature": 0, "num_ctx": context, "num_predict": 4096}}
        if self._think_supported:
            request["think"] = False
        started = time.monotonic()
        try:
            async with asyncio.timeout(300):
                async with self._client.stream("POST", self.base_url + "/api/chat", json=request) as response:
                    body = await read_json(response)
            self.last_metadata = {"provider": "ollama", "model": self.model, "task": task, "seconds": time.monotonic() - started,
                                  "input_token_upper_bound": input_bytes, "prompt_eval_count": body.get("prompt_eval_count"),
                                  "eval_count": body.get("eval_count"), "done_reason": body.get("done_reason"), "think_disabled": self._think_supported}
            if body.get("done_reason") == "length":
                raise GymError("invalid_model_output", "Local coach exhausted its bounded output before completing the schema.", status=502, retryable=True)
            value = json.loads(body["message"]["content"])
            validate_schema(value, schema)
        except asyncio.CancelledError:
            if self._owner:
                await self._owner.terminate()
                self._owner = None
            raise
        except (ValueError, TypeError, KeyError, RecursionError):
            raise GymError("invalid_model_output", "Local coach returned invalid structured output.", status=502, retryable=True) from None
        except (httpx.HTTPError, TimeoutError):
            raise GymError("coach_transport_error", "Local coach timed out or failed. Retry explicitly.", status=503, retryable=True) from None
        return value

    async def release(self):
        if self.managed and not self._owner:
            return
        try:
            async with self._client.stream("POST", self.base_url + "/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=10) as response:
                await read_json(response)
            async with self._client.stream("GET", self.base_url + "/api/ps", timeout=5) as response:
                state = await read_json(response)
            if any(item.get("name") == self.model or item.get("model") == self.model for item in state.get("models", [])):
                raise ValueError()
        except (httpx.HTTPError, GymError, ValueError):
            if self._owner:
                await self._owner.terminate()
                self._owner = None
            else:
                self._release_failed = True
                raise GymError("model_release_failed", "External coach model release could not be verified. Confirm idle before other GPU work.", status=409) from None

    async def close(self):
        if self._owner:
            await self._owner.terminate()
            self._owner = None
        await self._client.aclose()
