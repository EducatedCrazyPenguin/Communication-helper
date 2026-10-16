"""Owned speech entry point. Only stdlib is imported until an operation needs STT."""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import queue
import re
import socket
import sys
import threading
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from communication_gym.worker.protocol import VERSION, ProtocolError, encode_frame, read_frame


class WorkerCancelled(BaseException):
    pass


class WorkerFailure(Exception):
    def __init__(self, code: str) -> None:
        self.code = code


def install_offline_guard() -> None:
    """The guard is process-local and installed before importing model libraries."""
    original = socket.socket

    class OfflineSocket(original):
        def connect(self, address):
            raise OSError("Speech worker networking is disabled")

        def connect_ex(self, address):
            raise OSError("Speech worker networking is disabled")

    def denied(*args, **kwargs):
        raise OSError("Speech worker networking is disabled")

    socket.socket = OfflineSocket
    socket.create_connection = denied
    socket.getaddrinfo = denied
    for name in tuple(os.environ):
        upper = name.upper()
        if any(part in upper for part in ("TOKEN", "SECRET", "API_KEY", "APIKEY", "PASSWORD", "PROXY")):
            os.environ.pop(name, None)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      DO_NOT_TRACK="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")


def _cache_snapshot(model_id: str) -> Path | None:
    cache = Path(os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE") or
                 str(Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"))
    directory = cache / ("models--" + model_id.replace("/", "--"))
    snapshots = directory / "snapshots"
    candidates = []
    reference = directory / "refs" / "main"
    if reference.is_file():
        revision = reference.read_text(encoding="utf-8").strip()
        if revision and Path(revision).name == revision:
            candidates.append(snapshots / revision)
    if snapshots.is_dir():
        candidates.extend(sorted((item for item in snapshots.iterdir() if item.is_dir()),
                                 key=lambda item: item.stat().st_mtime, reverse=True))
    for snapshot in candidates:
        if snapshot.is_dir():
            return snapshot
    return None


def _complete_qwen(snapshot: Path | None) -> bool:
    if snapshot is None or not all((snapshot / name).is_file() for name in ("config.json", "tokenizer_config.json")):
        return False
    if not (snapshot / "tokenizer.json").is_file() and not all((snapshot / name).is_file() for name in ("vocab.json", "merges.txt")):
        return False
    indexes = tuple(snapshot.glob("*.safetensors.index.json"))
    if indexes:
        try:
            for index in indexes:
                weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
                if not isinstance(weight_map, dict):
                    return False
                names = tuple(weight_map.values())
                if names and all(isinstance(name, str) and Path(name).name == name
                                 and (snapshot / name).is_file() for name in names):
                    return True
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return False
    return any(path.is_file() and not re.search(r"-\d+-of-\d+\.", path.name)
               for path in snapshot.glob("*.safetensors"))


def preflight(repo: Path) -> dict:
    packages = {}
    missing = []
    for distribution in ("torch", "torchaudio", "qwen-asr", "transformers", "onnx-asr", "av", "numpy"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing.append(distribution)
    compatible = False
    try:
        tree = ast.parse((repo / "voice_to_text" / "models.py").read_text(encoding="utf-8-sig"))
        runtime = any(isinstance(node, ast.ClassDef) and node.name == "QwenRuntime" for node in tree.body)
        dispatcher = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "transcribe_candidate")
        compatible = runtime and any(arg.arg == "qwen_runtime" for arg in dispatcher.args.kwonlyargs)
    except (OSError, SyntaxError, StopIteration):
        pass
    snapshot = _cache_snapshot("Qwen/Qwen3-ASR-1.7B")
    qwen_cached = _complete_qwen(snapshot)
    silero_snapshot = _cache_snapshot("istupakov/silero-vad-onnx")
    silero_cached = bool(silero_snapshot and (silero_snapshot / "silero_vad.onnx").is_file())
    torch_home = Path(os.environ.get("TORCH_HOME", Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "torch"))
    aligner_cached = (torch_home / "hub" / "checkpoints" / "model.pt").is_file()
    alignment_available = aligner_cached and importlib.util.find_spec("uroman") is not None
    reason = ("speech_dependencies_missing" if missing else "incompatible_stt_package" if not compatible
              else "model_not_cached" if not qwen_cached else "silero_not_cached" if not silero_cached else None)
    revision = None
    try:
        git = repo / ".git"
        head = (git / "HEAD").read_text(encoding="ascii").strip()
        revision = ((git / head[5:]).read_text(encoding="ascii").strip() if head.startswith("ref: ") else head)
    except OSError:
        pass
    return {"available": reason is None, "provider_id": "original-stt-worker", "model_id": "Qwen/Qwen3-ASR-1.7B",
            "locality": "local", "languages": ["en", "hi", "hinglish"], "structured_output": True,
            "word_timing": "estimated", "word_confidence": False, "cancellation": "terminate_owned",
            "release_supported": True, "unavailable_reason": reason, "packages": packages,
            "missing_packages": missing, "model_cached": qwen_cached, "silero_cached": silero_cached,
            "alignment_available": alignment_available, "protocol_version": VERSION, "stt_revision": revision,
            "configured": {"candidate": "qwen3", "downloads": False},
            "verified": {"package_compatible": compatible, "model_cached": qwen_cached, "inference": False},
            "inference_verified": False, "pid": os.getpid()}


class SpeechWorker:
    def __init__(self, repo: Path, root: Path, send) -> None:
        self.repo = repo.resolve()
        self.root = root.resolve()
        self.send = send
        self.runtime = None
        self.active_id = None
        self.events: dict[str, threading.Event] = {}
        self.lock = threading.Lock()

    def path(self, value: str) -> Path:
        if not isinstance(value, str):
            raise WorkerFailure("invalid_path")
        path = Path(value).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise WorkerFailure("invalid_path")
        return path

    def _package(self) -> None:
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        import voice_to_text
        if not Path(voice_to_text.__file__).resolve().is_relative_to(self.repo):
            raise WorkerFailure("incompatible_stt_package")

    def release(self) -> None:
        if self.runtime is not None:
            self.runtime.close()

    def run(self, message: dict) -> None:
        request_id, op, payload = message["request_id"], message["op"], message["payload"]
        with self.lock:
            event = self.events[request_id]
            self.active_id = request_id

        def check() -> None:
            if event.is_set():
                raise WorkerCancelled()

        def progress(phase: str, fraction: float | None = None) -> None:
            check()
            # Never forward third-party progress strings, which can contain user text.
            safe_phase = ("alignment" if "align" in phase.lower() else "loading" if "load" in phase.lower()
                          else "transcribing" if "chunk" in phase.lower() or "qwen" in phase.lower() else "preparing")
            self.send(request_id, "progress", {"phase": safe_phase, "progress": fraction})

        started = time.monotonic()
        try:
            check()
            if op == "capabilities":
                result = preflight(self.repo)
            elif op == "release":
                self.release()
                result = {"released": True}
            elif op == "shutdown":
                self.release()
                result = {"shutdown": True}
            elif op in {"prepare_audio", "transcribe"}:
                self._package()
                from voice_to_text.core import Transcriber, TranscriptionResult
                from voice_to_text.audio import analyze_audio, write_pcm16_wav
                source = self.path(payload.get("source"))
                if not source.is_file() or source.stat().st_size > 256 * 1024 * 1024:
                    raise WorkerFailure("invalid_audio")
                progress("preparing", 0.01)
                try:
                    waveform = Transcriber._decode_audio(source, max_duration_seconds=902)
                except Exception as exc:
                    code = ("audio_too_long" if str(exc).startswith("audio_too_long:") else
                            "empty_audio" if str(exc).startswith("No audio samples found") else "invalid_audio")
                    raise WorkerFailure(code) from exc
                check()
                duration = waveform.size / 16_000
                if not 0 < duration <= 902:
                    raise WorkerFailure("audio_too_long" if duration > 902 else "empty_audio")
                diagnostics = analyze_audio(waveform).to_dict()
                check()
                if op == "prepare_audio":
                    destination = self.path(payload.get("destination"))
                    if destination == source or destination.exists():
                        raise WorkerFailure("immutable_audio")
                    temporary = destination.with_name(destination.name + ".partial")
                    try:
                        write_pcm16_wav(temporary, waveform)
                        check()
                        temporary.replace(destination)
                    finally:
                        temporary.unlink(missing_ok=True)
                    result = {"duration_seconds": duration, "sample_rate": 16_000, "diagnostics": diagnostics}
                else:
                    from voice_to_text.accuracy import AccuracyPipeline, AccuracySettings
                    from voice_to_text.models import QwenRuntime
                    from voice_to_text.exports import result_to_dict
                    capabilities = preflight(self.repo)
                    if not capabilities["available"]:
                        raise WorkerFailure(capabilities["unavailable_reason"])
                    language = payload.get("language")
                    if language not in {"en", "hi", "hinglish"}:
                        raise WorkerFailure("invalid_language")
                    if self.runtime is None:
                        self.runtime = QwenRuntime(allow_downloads=False)
                    alignment = bool(language != "en" and payload.get("alignment", "available") != "none"
                                     and capabilities["alignment_available"])
                    settings = AccuracySettings(device=payload.get("device", "auto"),
                                                language={"en": "en", "hi": "hi", "hinglish": None}[language],
                                                candidates=("qwen3",), audio_mode="original", use_consensus=False,
                                                enhanced_recovery=False, diarize=False, generate_readable=False,
                                                use_readable_model=False, glossary=(), hf_token="",
                                                allow_model_downloads=False, force_align_words=alignment)
                    if float(abs(waveform).max()) < 0.0001:
                        asr = TranscriptionResult(source, "Qwen/Qwen3-ASR-1.7B", settings.device,
                                                  settings.language or "unknown", 0.0, duration, (),
                                                  diagnostics={**diagnostics, "no_speech_detected": True})
                    else:
                        asr = AccuracyPipeline(self.repo, progress, qwen_runtime=self.runtime).transcribe(
                            source, settings, checkpoint_dir=self.path(payload.get("checkpoint_dir")))
                    result = result_to_dict(asr)
                    result["diagnostics"].update({"practice_language": language,
                                                   "alignment_requested": alignment,
                                                   "alignment_available": capabilities["alignment_available"],
                                                   "worker_pid": os.getpid(),
                                                   "worker_protocol_version": VERSION,
                                                   "stt_revision": capabilities["stt_revision"],
                                                   "elapsed_seconds": time.monotonic() - started})
                    if language != "en" and not alignment:
                        result["diagnostics"]["alignment_warning"] = "Optional Hindi alignment unavailable or disabled; timings remain estimated."
                    if not result["segments"]:
                        result["diagnostics"]["no_speech_detected"] = True
            else:
                raise WorkerFailure("unsupported_operation")
            check()
            self.send(request_id, "result", result)
        except WorkerCancelled:
            self.release()
            self.send(request_id, "cancelled", {})
        except Exception as exc:
            self.release()
            code = exc.code if isinstance(exc, WorkerFailure) else (
                "gpu_out_of_memory" if "out of memory" in str(exc).lower() else
                "model_not_cached" if "model_not_cached" in str(exc) else "speech_operation_failed")
            self.send(request_id, "error", {"code": code})
        finally:
            with self.lock:
                self.active_id = None
                self.events.pop(request_id, None)


def protocol_input():
    stream = os.fdopen(os.dup(sys.stdin.fileno()), "rb", buffering=0)
    # Native imports may inspect standard input while protocol reads are blocked.
    # Keep the protocol pipe private and give dependencies an inert standard input.
    with open(os.devnull, "rb", buffering=0) as source:
        os.dup2(source.fileno(), sys.stdin.fileno())
    return stream


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    # Preserve a private protocol descriptor, then redact Python and native output.
    protocol_output = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    error_output = os.fdopen(os.dup(sys.stderr.fileno()), "wb", buffering=0)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, sys.stdout.fileno())
    os.dup2(write_fd, sys.stderr.fileno())
    os.close(write_fd)

    def redact_output() -> None:
        reported = 0
        with os.fdopen(read_fd, "rb", buffering=0) as stream:
            while stream.read(4096):
                if reported < 4:
                    error_output.write(b"[speech dependency output redacted]\n")
                    reported += 1

    threading.Thread(target=redact_output, daemon=True).start()
    install_offline_guard()
    send_lock = threading.Lock()

    def send(request_id: str, op: str, payload: dict) -> None:
        frame = encode_frame({"version": VERSION, "request_id": request_id, "op": op, "payload": payload})
        with send_lock:
            protocol_output.write(frame)

    worker = SpeechWorker(args.repo, args.root, send)
    work: queue.Queue = queue.Queue(maxsize=2)

    def execute() -> None:
        try:
            while True:
                message = work.get()
                if message is None:
                    return
                worker.run(message)
                if message["op"] == "shutdown":
                    return
        finally:
            worker.release()

    input_stream = protocol_input()
    executor = threading.Thread(target=execute, daemon=True)
    executor.start()
    send("startup", "ready", {"protocol_version": VERSION, "pid": os.getpid()})
    try:
        while True:
            message = read_frame(input_stream)
            if message is None:
                break
            if message["op"] == "cancel":
                target = message["payload"].get("request_id")
                with worker.lock:
                    event = worker.events.get(target)
                    if event:
                        event.set()
                continue
            with worker.lock:
                if message["request_id"] in worker.events:
                    raise ProtocolError("Duplicate in-flight request ID")
                worker.events[message["request_id"]] = threading.Event()
            try:
                work.put_nowait(message)
            except queue.Full:
                with worker.lock:
                    worker.events.pop(message["request_id"], None)
                send(message["request_id"], "error", {"code": "worker_busy"})
                continue
            if message["op"] == "shutdown":
                executor.join()
                break
    except (ProtocolError, OSError):
        pass
    finally:
        with worker.lock:
            for event in worker.events.values():
                event.set()
        if executor.is_alive():
            work.put(None)
            executor.join()
        input_stream.close()
        protocol_output.close()


if __name__ == "__main__":
    main()
