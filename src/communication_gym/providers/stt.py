"""Async adapter to the original recognizer, isolated in its existing interpreter."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import math
import os
from pathlib import Path
from typing import Callable

from communication_gym.domain.models import GymError, new_id, utc_now
from communication_gym.providers.processes import OwnedProcess
from communication_gym.worker.protocol import VERSION, ProtocolError, encode_frame, read_frame_async


_SAFE_ERRORS = {
    "model_not_cached": "The complete Qwen model is not cached. Downloads require your approval.",
    "silero_not_cached": "The local Silero VAD cache is missing. No download was attempted.",
    "speech_dependencies_missing": "The existing speech interpreter is missing required packages.",
    "incompatible_stt_package": "The configured original STT package needs the Gym protocol capability changes.",
    "gpu_out_of_memory": "Speech ran out of GPU memory. Close other heavy applications or choose CPU, then retry.",
    "invalid_audio": "This file does not contain decodable audio.",
    "empty_audio": "This recording has no audio samples. Record again.",
    "audio_too_long": "Recordings must be no longer than 15 minutes, plus a two-second recorder tolerance.",
    "immutable_audio": "The prepared audio destination already exists. Start a new generation.",
    "invalid_path": "Speech artifacts must remain inside Gym-owned storage.",
    "invalid_language": "Choose English, Hindi or Hinglish.",
    "worker_busy": "The speech worker is busy. Wait for the current operation.",
}


def _error(code: str) -> GymError:
    return GymError(code, _SAFE_ERRORS.get(code, "Speech processing failed. Your recording is preserved; retry the operation."),
                    status=422 if code in {"invalid_audio", "empty_audio", "audio_too_long", "invalid_language"} else 503,
                    retryable=code not in {"invalid_path", "invalid_language", "audio_too_long"})


def _scrub_environment() -> dict[str, str]:
    safe = {}
    # Do not inherit arbitrary app/agent credentials, PYTHONPATH or startup hooks.
    allowed = {"SYSTEMROOT", "WINDIR", "PATH", "PATHEXT", "TEMP", "TMP", "USERPROFILE", "HOME", "LOCALAPPDATA",
               "APPDATA", "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "COMMONPROGRAMFILES", "COMSPEC",
               "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME", "XDG_CACHE_HOME", "CUDA_PATH",
               "CUDA_VISIBLE_DEVICES", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE"}
    for key, value in os.environ.items():
        if key.upper() in allowed:
            safe[key] = value
    safe.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                HF_HUB_DISABLE_IMPLICIT_TOKEN="1", DO_NOT_TRACK="1", PYTHONNOUSERSITE="1",
                PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", TOKENIZERS_PARALLELISM="false")
    return safe


class SpeechProvider:
    def __init__(self, config: dict, root: Path) -> None:
        self.config = dict(config)
        self.root = Path(root).resolve()
        self._owned: OwnedProcess | None = None
        self._worker_pid: int | None = None
        self._reader: asyncio.Task | None = None
        self._stderr: asyncio.Task | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._progress: dict[str, Callable] = {}
        self._operation_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._active: str | None = None
        self._closing = False

    @property
    def pid(self) -> int | None:
        """PID of the protocol-speaking worker, not its optional interpreter launcher."""
        return self._worker_pid if self._owned and self._owned.returncode is None else None

    def _path(self, path: Path) -> str:
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(self.root) or resolved == self.root:
            raise _error("invalid_path")
        return str(resolved)

    async def _send(self, request_id: str, op: str, payload: dict) -> None:
        owned = self._owned
        if owned is None or owned.returncode is not None or owned.process.stdin is None:
            raise _error("speech_worker_stopped")
        async with self._write_lock:
            owned.process.stdin.write(encode_frame({"version": VERSION, "request_id": request_id, "op": op, "payload": payload}))
            await owned.process.stdin.drain()

    async def _read_responses(self, owned: OwnedProcess) -> None:
        try:
            while True:
                message = await read_frame_async(owned.process.stdout)
                request_id, op, payload = message["request_id"], message["op"], message["payload"]
                future = self._pending.get(request_id)
                if future is None or future.done():
                    continue  # Cancelled generations cannot accept a late result.
                if op == "progress":
                    callback = self._progress.get(request_id)
                    if callback:
                        returned = callback(payload.get("phase", "transcribing"), payload.get("progress"))
                        if inspect.isawaitable(returned):
                            await returned
                elif op in {"ready", "result"}:
                    future.set_result(payload)
                elif op == "cancelled":
                    future.set_exception(GymError("cancelled", "Speech operation cancelled.", status=409, retryable=True))
                elif op == "error":
                    future.set_exception(_error(payload.get("code", "speech_operation_failed")))
                else:
                    raise ProtocolError("Unexpected response operation")
        except asyncio.CancelledError:
            raise
        except Exception:
            await owned.terminate()
            for future in tuple(self._pending.values()):
                if not future.done():
                    future.set_exception(_error("speech_worker_stopped"))

    async def _drain_stderr(self, owned: OwnedProcess) -> None:
        # Worker sends redacted bounded diagnostics. Never retain third-party text.
        while await owned.process.stderr.read(4096):
            pass

    async def _ensure_started(self) -> None:
        if self._closing:
            raise _error("speech_worker_stopped")
        if self.pid:
            return
        await self._dispose()
        python = Path(self.config.get("python_path", ""))
        repo = Path(self.config.get("repo_path", ""))
        if not python.is_file() or not (repo / "voice_to_text" / "__init__.py").is_file():
            raise _error("speech_not_configured")
        if self.config.get("candidate", "qwen3") != "qwen3":
            raise _error("incompatible_stt_package")
        startup = asyncio.get_running_loop().create_future()
        self._pending["startup"] = startup
        try:
            self._owned = await OwnedProcess.start(
                [str(python.resolve()), "-B", "-u", str(Path(__file__).resolve().parents[1] / "worker" / "main.py"),
                 "--repo", str(repo.resolve()), "--root", str(self.root)], env=_scrub_environment(), cwd=repo.resolve())
            self._reader = asyncio.create_task(self._read_responses(self._owned))
            self._stderr = asyncio.create_task(self._drain_stderr(self._owned))
            ready = await asyncio.wait_for(asyncio.shield(startup), timeout=15)
            if ready.get("protocol_version") != VERSION:
                raise _error("incompatible_stt_package")
            worker_pid = ready.get("pid")
            if not self._owned.owns_pid(worker_pid):
                raise _error("incompatible_stt_package")
            self._worker_pid = worker_pid
        except BaseException as exc:
            await self._dispose()
            if isinstance(exc, (asyncio.CancelledError, GymError)):
                raise
            raise _error("speech_worker_stopped") from None
        finally:
            self._pending.pop("startup", None)
            if not startup.done():
                startup.cancel()
            elif not startup.cancelled():
                startup.exception()

    async def _rpc(self, op: str, payload: dict, progress=None, timeout: float = 1800) -> dict:
        async with self._operation_lock:
            await self._ensure_started()
            request_id = new_id()
            future = asyncio.get_running_loop().create_future()
            self._pending[request_id] = future
            if progress:
                self._progress[request_id] = progress
            self._active = request_id
            try:
                await self._send(request_id, op, payload)
                return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
            except asyncio.CancelledError:
                await self.cancel()
                raise
            except TimeoutError:
                await self.cancel()
                raise GymError("speech_timeout", "Speech processing exceeded its deadline. Retry from the saved audio.", 503, True) from None
            except (BrokenPipeError, ConnectionError, OSError):
                await self._dispose()
                raise _error("speech_worker_stopped") from None
            finally:
                self._active = None
                self._pending.pop(request_id, None)
                self._progress.pop(request_id, None)
                if not future.done():
                    future.cancel()
                elif not future.cancelled():
                    future.exception()  # Consume errors from separately cancelled callers.

    async def capabilities(self) -> dict:
        try:
            return await self._rpc("capabilities", {}, timeout=20)
        except GymError as exc:
            return {"available": False, "provider_id": "original-stt-worker", "model_id": "Qwen/Qwen3-ASR-1.7B",
                    "locality": "local", "languages": ["en", "hi", "hinglish"], "structured_output": True,
                    "word_timing": "estimated", "word_confidence": False, "cancellation": "terminate_owned",
                    "release_supported": True, "unavailable_reason": exc.code, "configured": dict(self.config),
                    "verified": {"inference": False}, "protocol_version": VERSION}

    async def prepare_audio(self, source: Path, destination: Path) -> dict:
        return await self._rpc("prepare_audio", {"source": self._path(source), "destination": self._path(destination)}, timeout=180)

    async def transcribe(self, source: Path, language: str, checkpoint_dir: Path, progress=None) -> dict:
        if language not in {"en", "hi", "hinglish"}:
            raise _error("invalid_language")
        return await self._rpc("transcribe", {"source": self._path(source), "language": language,
                                               "checkpoint_dir": self._path(checkpoint_dir),
                                               "device": self.config.get("device", "auto"),
                                               "alignment": self.config.get("alignment", "available")}, progress=progress)

    async def cancel(self) -> None:
        request_id = self._active
        if request_id is None or self._owned is None:
            return
        future = self._pending.get(request_id)
        try:
            await self._send(new_id(), "cancel", {"request_id": request_id})
            if future and not future.done():
                await asyncio.wait_for(asyncio.shield(future), timeout=5)
        except Exception:
            if future is not None and future.done():
                return
            await self._dispose()

    async def release(self) -> None:
        if self.pid:
            await self._rpc("release", {}, timeout=15)

    async def _dispose(self) -> None:
        owned, self._owned = self._owned, None
        self._worker_pid = None
        if owned:
            await owned.terminate()
        current = asyncio.current_task()
        for task in (self._reader, self._stderr):
            if task and task is not current:
                task.cancel()
        tasks = [task for task in (self._reader, self._stderr) if task and task is not current]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._reader = self._stderr = None
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(_error("speech_worker_stopped"))

    async def close(self) -> None:
        if self._closing:
            return
        try:
            async with asyncio.timeout(8):
                await self.cancel()
                if self.pid:
                    try:
                        await self._rpc("shutdown", {}, timeout=2)
                    except GymError:
                        pass
                    if self._owned:
                        await self._owned.close(grace_seconds=1)
        except TimeoutError:
            pass
        finally:
            self._closing = True
            await self._dispose()


def _finite(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def build_transcript(result: dict, attempt_id: str) -> dict:
    """Build a conservative analysis view; the recognizer payload is never rewritten."""
    transcript_id = new_id()
    duration = _finite(result.get("duration_seconds"))
    duration = duration if duration is not None and duration > 0 else None
    segments = []
    pieces = []
    offset = 0
    edits = result.get("diagnostics", {}).get("chunk_stitch_edits", [])
    raw_segments = result.get("raw_segments", [])
    for index, source in enumerate(result.get("segments", [])):
        text = str(source.get("text", ""))
        start, end = _finite(source.get("start")), _finite(source.get("end"))
        valid_segment = (start is not None and end is not None and duration is not None
                         and 0 <= start < end <= duration)
        if not valid_segment:
            start = end = None
        stitch_edited = False
        for edit in edits:
            chunk_index = edit.get("source_chunk_index")
            raw = raw_segments[chunk_index] if isinstance(chunk_index, int) and 0 <= chunk_index < len(raw_segments) else {}
            if raw and raw.get("end") == source.get("end"):
                stitch_edited = True
            left, right = _finite(edit.get("overlap_start_s")), _finite(edit.get("overlap_end_s"))
            if valid_segment and left is not None and right is not None and start <= right and end >= left:
                stitch_edited = True
        segment_id = new_id()
        words = []
        word_cursor = 0
        offsets_valid = True
        last_word_end = start
        for word in source.get("words", []):
            word_text = str(word.get("text", ""))
            position = text.find(word_text, word_cursor) if word_text else -1
            exact = offsets_valid and position >= 0 and not text[word_cursor:position].strip()
            offsets_valid = exact
            word_start, word_end = _finite(word.get("start")), _finite(word.get("end"))
            valid_word = (valid_segment and word_start is not None and word_end is not None
                          and start <= word_start < word_end <= end
                          and (last_word_end is None or word_start >= last_word_end))
            if valid_word:
                last_word_end = word_end
            else:
                word_start = word_end = None
            timing_origin = word.get("timing_origin", "unknown")
            timing_kind = timing_origin if valid_word and timing_origin in {"estimated", "aligned"} else "none"
            confidence = _finite(word.get("confidence"))
            if confidence is not None and not 0 <= confidence <= 1:
                confidence = None
            item = {**word, "id": new_id(), "segment_id": segment_id, "text": word_text,
                    "start": word_start, "end": word_end, "confidence": confidence,
                    "timing_origin": timing_origin, "timing_kind": timing_kind,
                    "confidence_kind": word.get("confidence_kind", "unknown")}
            item.pop("start_char", None)
            item.pop("end_char", None)
            if exact:
                item.update(start_char=offset + position, end_char=offset + position + len(word_text))
                word_cursor = position + len(word_text)
            words.append(item)
        segments.append({"id": segment_id, "transcript_id": transcript_id, "text": text,
                         "start_char": offset, "end_char": offset + len(text), "start": start, "end": end,
                         "timing_kind": "estimated" if valid_segment else "none", "words": words,
                         "speaker": source.get("speaker"), "overlap": bool(source.get("overlap", False)),
                         "uncertain": bool(source.get("uncertain", False) or stitch_edited), "stitch_edited": stitch_edited})
        pieces.append(text)
        offset += len(text) + 1
    canonical = "\n".join(pieces)
    return {"id": transcript_id, "attempt_id": attempt_id, "schema_version": 1, "revision": 1,
            "kind": "asr_original", "parent_id": None, "text": canonical, "segments": segments,
            "content_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "duration_seconds": duration, "diagnostics": copy.deepcopy(result.get("diagnostics", {})),
            "provenance": {"provider": "original-stt-worker", "model": result.get("model"),
                           "device": result.get("device"), "language": result.get("language"),
                           "details": copy.deepcopy(result.get("provenance", []))},
            "label": "ASR verbatim — chunk-stitched", "original": copy.deepcopy(result),
            "created_at": utc_now(), "updated_at": utc_now()}
