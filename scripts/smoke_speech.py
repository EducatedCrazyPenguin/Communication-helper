"""Explicit real-Qwen smoke using only make_smoke_audio.ps1's synthetic English WAVs.

Requires an exclusive GPU lease when using CUDA/auto. Never downloads models or
reads normal Gym settings. This checks the speech adapter, not browser capture.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import time

from communication_gym.domain.models import GymError
from communication_gym.providers.stt import SpeechProvider, build_transcript


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


async def run(args) -> int:
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)  # Fresh checkpoints, never resumed results.
    provider = SpeechProvider({"repo_path": str(args.stt_repo.resolve()),
                               "python_path": str(args.stt_python.resolve()),
                               "device": args.device, "alignment": "none"}, root)
    records = []
    report = {"schema_version": 1, "synthetic_audio_only": True,
              "browser_capture_verified": False, "records": records, "passed": False}
    launcher = None
    try:
        capabilities = await provider.capabilities()
        save(root / "capabilities.json", capabilities)
        if not capabilities["available"]:
            raise GymError(capabilities["unavailable_reason"], "Speech preflight failed.", 503)
        pid = provider.pid
        launcher = provider._owned
        report["worker_pid"] = pid
        report["launcher_pid"] = launcher.pid
        # retry-en is the exact fixture used by the recorded pre-fix baseline.
        for index, name in enumerate(("retry-en", "initial-en", "answer-en")):
            if index == 2:
                await provider.release()
                await provider.release()  # Acknowledged, idempotent release.
                report["release_acknowledged"] = True
            source = args.audio_dir.resolve() / (name + ".wav")
            owned_source = root / (name + ".wav")
            shutil.copyfile(source, owned_source)
            prepared = root / (name + "-prepared.wav")
            preparation = await provider.prepare_audio(owned_source, prepared)
            started = time.monotonic()
            original = await provider.transcribe(prepared, "en", root / (name + "-checkpoint"))
            elapsed = time.monotonic() - started
            save(root / (name + "-original.json"), original)
            transcript = build_transcript(original, name)
            save(root / (name + "-transcript.json"), transcript)
            diagnostics = original["diagnostics"]
            words = [word for segment in original["segments"] for word in segment.get("words", [])]
            checks = {"same_worker": provider.pid == diagnostics.get("worker_pid") == pid,
                      "language_en": original["language"] == "en" and diagnostics.get("language_forced") == "English",
                      "confidence_unavailable": original["language_probability"] == 0.0
                          and diagnostics.get("language_probability_kind") == "unavailable"
                          and all(word.get("confidence_kind") == "unavailable" and word.get("confidence") is None for word in words),
                      "estimated_timings": bool(words) and all(word.get("timing_origin") == "estimated" for word in words),
                      "english_content": "battery" in transcript["text"].lower()
                          and len(transcript["text"].split()) >= 10
                          and not any("\u0900" <= character <= "\u097f" for character in transcript["text"]),
                      "expected_load_count": diagnostics.get("qwen_runtime_load_count") == (1 if index < 2 else 2),
                      "cached_weights": "loaded from local Hugging Face cache" in original.get("provenance", [])}
            record = {"fixture": name, "sha256": hashlib.sha256(owned_source.read_bytes()).hexdigest(),
                      "preparation": preparation, "transcription_seconds": elapsed,
                      "worker_pid": provider.pid, "device": original["device"],
                      "model_load_count": diagnostics.get("qwen_runtime_load_count"),
                      "language": original["language"], "text": transcript["text"], "checks": checks}
            records.append(record)
            save(root / "summary.json", report)
            print(json.dumps(record, ensure_ascii=True), flush=True)
        report["distinct_recording_results"] = len({record["text"] for record in records}) == 3
        report["passed"] = report["distinct_recording_results"] and all(all(record["checks"].values()) for record in records)
    except GymError as error:
        report["safe_error_code"] = error.code
    finally:
        await provider.close()
        report["worker_pid_after_close"] = provider.pid
        report["launcher_returncode"] = launcher.returncode if launcher else None
        report["passed"] = bool(report["passed"] and provider.pid is None
                                and launcher is not None and launcher.returncode == 0)
        save(root / "summary.json", report)
    print(json.dumps({"passed": report["passed"], "artifact_directory": str(root),
                      "safe_error_code": report.get("safe_error_code")}))
    return 0 if report["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stt-repo", required=True, type=Path)
    parser.add_argument("--stt-python", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
