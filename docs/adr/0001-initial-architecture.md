# ADR 0001 — separate application, original speech package, owned worker

Status: accepted.

Communication Gym is a Python 3.12 FastAPI browser application with Pydantic domain contracts, SQLAlchemy/Alembic/SQLite persistence and local HTML/JavaScript. Audio is managed files. The original sibling `voice to text` checkout supplies recognition through its existing ML interpreter and an owned persistent framed-pipe worker. Gym has a separate dependency environment. Coach and optional TTS are independent adapters.

## Decision

Choose B: a new repository importing `voice_to_text` through process isolation. Preserve decoding, Qwen accuracy pipeline, alignment and result exports. Add explicit Qwen ownership, correct language propagation and pre-stitch/timing evidence to that package with backwards-compatible optional parameters. One expensive Gym workload owns resources at a time; release model residency when changing workloads, never kill unrelated processes.

## Evaluated choices

A, extend the existing Flask app: simple first installation but joins unrelated session/coaching/UI concerns to the daily transcriber. Regression and upgrade risk are high; its current cache does not retain Qwen.

B, selected: isolates pinned ML dependencies, keeps local structured evidence and allows a warm worker without a public service. Windows process containment and a versioned pipe contract add a small explicit boundary. Independent transcriber instances do not share model memory; avoid concurrent heavy workloads.

C, existing HTTP service: could share models between apps, but current UI-coupled job/upload routes have no stable external contract and do not keep Qwen loaded. Stabilizing authentication, model service lifecycle and another launchable server is not justified for one user. Do not call its one-shot CLI per recording.

## Privacy and licensing

Local processing is the default. No OMP credentials, implicit cloud fallback, analytics or automatic model downloads. Original audio/ASR are immutable, corrections create revisions, model answers are separate. VoiceStudio is optional HTTP, never vendored or modified; voice permission is explicit. Gym is AGPL-3.0-only, consistent with the imported STT application; model/voice terms are separate.

## Audited baseline

STT commit 47c2b5993211f9fd0795fbe6a06d687f1ba6f7bf; VoiceStudio public commit 9080d675e0bbebea5623e84bad7c679537527b28. Machine: Python 3.12.10, RTX 3090 24 GiB, CUDA-enabled torch 2.8.0+cu129. These are audit evidence, not inference-quality claims.
