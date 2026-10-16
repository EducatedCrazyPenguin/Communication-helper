"""Explicitly configured structured coach transport; never sends audio or retries HTTP."""
from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import socket
import time
from urllib.parse import urlsplit

import httpx

from communication_gym.domain.models import GymError

MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def compact_schema(schema: dict) -> dict:
    """Factor repeated schema nodes, preserving every validation constraint."""
    counts, nodes = {}, {}
    def collect(value):
        if isinstance(value, dict):
            key = json.dumps(value, sort_keys=True, separators=(",", ":"))
            if len(key) >= 200 and ("type" in value or "anyOf" in value):
                counts[key] = counts.get(key, 0) + 1
                nodes[key] = value
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    collect(schema)
    names = {key: f"shared_{index}" for index, key in enumerate(key for key, count in counts.items() if count > 1)}
    def replace(value, *, root=False):
        if isinstance(value, dict):
            key = json.dumps(value, sort_keys=True, separators=(",", ":"))
            if not root and key in names:
                return {"$ref": f"#/$defs/{names[key]}"}
            return {key: replace(child) for key, child in value.items()}
        if isinstance(value, list):
            return [replace(child) for child in value]
        return value
    result = replace(schema, root=True)
    if names:
        result["$defs"] = {name: replace(nodes[key], root=True) for key, name in names.items()}
    return result


def validate_schema(value, schema: dict, path: str = "$", *, depth: int = 0, root: dict | None = None):
    """Validate the deliberately small JSON-schema vocabulary emitted by Gym."""
    if depth > 32:
        raise ValueError("schema_depth")
    root = schema if root is None else root
    if "$ref" in schema:
        reference = schema["$ref"]
        if not reference.startswith("#/$defs/") or reference[8:] not in root.get("$defs", {}):
            raise ValueError("unsupported schema reference")
        return validate_schema(value, root["$defs"][reference[8:]], path, depth=depth + 1, root=root)
    if "anyOf" in schema:
        for choice in schema["anyOf"]:
            try:
                return validate_schema(value, choice, path, depth=depth + 1, root=root)
            except ValueError:
                pass
        raise ValueError(f"{path}: invalid alternative")
    kind = schema.get("type")
    valid = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str), "boolean": isinstance(value, bool),
             "integer": isinstance(value, int) and not isinstance(value, bool),
             "number": isinstance(value, (int, float)) and not isinstance(value, bool),
             "null": value is None}
    if kind and not valid.get(kind, False):
        raise ValueError(f"{path}: expected {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: invalid enum")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        if set(schema.get("required", [])) - value.keys():
            raise ValueError(f"{path}: missing fields")
        if schema.get("additionalProperties") is False and value.keys() - props.keys():
            raise ValueError(f"{path}: unexpected fields")
        for key, child in value.items():
            if key in props:
                validate_schema(child, props[key], f"{path}.{key}", depth=depth + 1, root=root)
    elif isinstance(value, list):
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 10000):
            raise ValueError(f"{path}: invalid item count")
        for index, child in enumerate(value):
            validate_schema(child, schema.get("items", {}), f"{path}[{index}]", depth=depth + 1, root=root)
    elif isinstance(value, str):
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 100000):
            raise ValueError(f"{path}: invalid string length")
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        if not math.isfinite(value) or not schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf):
            raise ValueError(f"{path}: invalid number")
    return value


def checked_endpoint(url: str, *, allow_cloud: bool = False) -> tuple[str, bool]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise GymError("invalid_endpoint", "Configure an HTTP endpoint without credentials or query parameters.", status=422)
    try:
        addresses = {ipaddress.ip_address(item[4][0]) for item in socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)}
    except (OSError, ValueError):
        raise GymError("endpoint_unavailable", "The provider hostname could not be resolved.", status=503, retryable=True) from None
    local = bool(addresses) and all(address.is_loopback for address in addresses)
    if not local and (not allow_cloud or parsed.scheme != "https"):
        raise GymError("cloud_consent_required", "Remote coaching requires HTTPS, cloud permission and current session consent.", status=403)
    # Pin local endpoints to their numeric loopback address to avoid DNS rebinding.
    if local and parsed.hostname not in {"127.0.0.1", "::1"}:
        host = "[::1]" if all(a.version == 6 for a in addresses) else "127.0.0.1"
        url = f"{parsed.scheme}://{host}" + (f":{parsed.port}" if parsed.port else "") + parsed.path
    return url.rstrip("/"), local


async def read_json(response: httpx.Response, expected_type=dict):
    response.raise_for_status()
    data = bytearray()
    async for chunk in response.aiter_bytes():
        data.extend(chunk)
        if len(data) > MAX_RESPONSE_BYTES:
            raise GymError("provider_response_too_large", "Provider response exceeds the safe limit.", status=502)
    try:
        value = json.loads(data)
        if not isinstance(value, expected_type):
            raise ValueError()
        return value
    except (ValueError, UnicodeError, RecursionError):
        raise GymError("invalid_model_output", "Provider returned invalid JSON.", status=502, retryable=True) from None


class OpenAICompatibleProvider:
    def __init__(self, config: dict, api_key: str | None = None):
        self.config = dict(config)
        self.model = config.get("model", "")
        self.base_url = config.get("base_url", "http://127.0.0.1:1234")
        self._api_key = api_key
        self._local = True
        self._client = httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=httpx.Timeout(300, connect=5))
        self.last_metadata = {}
        self._schema_passed = False
        self._probe_requested = False
        self._active_local = False

    def _endpoint(self):
        consent = self.config.get("session_consent")
        approved = isinstance(consent, dict) and consent.get("approved") is True and isinstance(consent.get("endpoint"), str) and consent["endpoint"].rstrip("/") == self.base_url.rstrip("/") and consent.get("model") == self.model
        self.base_url, self._local = checked_endpoint(self.base_url, allow_cloud=bool(self.config.get("allow_cloud") and approved))
        return self.base_url.removesuffix("/v1")

    def _headers(self):
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    async def capabilities(self) -> dict:
        result = {"provider_id": "openai_compatible", "model_id": self.model, "available": False,
                  "locality": "local", "languages": ["en", "hi", "hinglish"], "structured_output": False,
                  "word_timing": "none", "word_confidence": False, "cancellation": "client_only",
                  "release_supported": False, "endpoint_reachable": False, "model_installed": False,
                  "schema_probe_passed": self._schema_passed, "real_calibration_passed": False}
        try:
            endpoint = self._endpoint()
            result["locality"] = "local" if self._local else "cloud"
            async with self._client.stream("GET", endpoint + "/v1/models", headers=self._headers()) as response:
                value = await read_json(response)
            result["endpoint_reachable"] = True
            result["model_installed"] = any(item.get("id") == self.model for item in value.get("data", []))
            result["available"] = result["model_installed"]
            result["structured_output"] = result["schema_probe_passed"]
            if not result["available"]:
                result["unavailable_reason"] = "Selected model is not advertised by the configured endpoint."
        except (GymError, httpx.HTTPError):
            result["unavailable_reason"] = "Configured endpoint is unavailable or not permitted."
        return result

    async def probe(self) -> dict:
        """Explicit synthetic strict-output probe, never startup inference."""
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
        self._probe_requested = True
        self._schema_passed = False
        try:
            result = await self.complete("schema_probe", "Return {\"ok\":true}.", {"ok": True}, schema)
            self._schema_passed = result == {"ok": True}
        finally:
            self._probe_requested = False
        return await self.capabilities()

    async def complete(self, task: str, system: str, payload: dict, schema: dict) -> dict:
        endpoint = self._endpoint()
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        wire_schema = compact_schema(schema)
        schema_text = json.dumps(wire_schema, ensure_ascii=False, separators=(",", ":"))
        # UTF-8 bytes upper-bound tokens conservatively; reject rather than truncate.
        budget = int(self.config.get("context_tokens", 16384)) - 4096
        if len((system + serialized + schema_text).encode("utf-8")) + 256 > budget:
            raise GymError("context_capacity_exceeded", "This request exceeds the configured coach context. Increase the context capacity.", status=422)
        request = {"model": self.model, "temperature": 0, "max_tokens": 4096, "stream": False,
                   "messages": [{"role": "system", "content": system + "\nReturn only JSON matching this schema:\n" + schema_text},
                                {"role": "user", "content": serialized}]}
        if self._schema_passed or self._probe_requested:
            request["response_format"] = {"type": "json_schema", "json_schema": {"name": task, "strict": True, "schema": wire_schema}}
        started = time.monotonic()
        self._active_local = self._local and self.config.get("device") != "cpu"
        try:
            async with asyncio.timeout(300):
                async with self._client.stream("POST", endpoint + "/v1/chat/completions", headers=self._headers(), json=request) as response:
                    body = await read_json(response)
            raw = body["choices"][0]["message"]["content"]
            value = json.loads(raw)
            validate_schema(value, schema)
        except (ValueError, KeyError, IndexError, TypeError, RecursionError):
            raise GymError("invalid_model_output", "Coach response did not match the required schema.", status=502, retryable=True) from None
        except (httpx.HTTPError, TimeoutError):
            raise GymError("coach_transport_error", "Coach request failed or timed out. Retry explicitly.", status=503, retryable=True) from None
        self.last_metadata = {"provider": "openai_compatible", "model": self.model, "seconds": time.monotonic() - started}
        return value

    async def release(self):
        if self._active_local:
            raise GymError("external_model_release_required", "Unload the externally managed coach model before switching GPU work.", status=409)

    async def confirm_idle(self):
        self._active_local = False

    async def close(self):
        await self._client.aclose()
