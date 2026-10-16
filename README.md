# Communication Gym

Personal local-first communication practice in English, Hindi and Hinglish. Train structure and listener understanding: topic → five prep bullets → speak → verbatim transcript → at most three evidence-backed problems → independent listener view → spoken questions → shorter retry → comparison and progress. Not a pronunciation grader or automatic speech rewriter.

Development snapshot: implementation and verification are ongoing. This is source for contributors, not a verified desktop release. The workflow above describes the intended training experience.

## Setup

Python 3.12 and uv: `uv sync --extra dev`. Gym installs no speech models. It uses the existing sibling voice to text/.venv through an isolated worker. Never install Gym dependencies into that environment.

Voice to Text is a separate prerequisite; its source, environment and model weights are not bundled. The current Windows defaults look for it in `~/Documents/ChatGPT/voice to text`. Configure the local STT paths for a different location. Model inference and calibration scripts require deliberate opt-in.

## Run

`./.venv/Scripts/python.exe -m communication_gym serve` (loopback port 8787). The launch URL authenticates the local browser; do not share it. `doctor --no-load` reports prerequisites without inference. Use GYM_DATA_DIR for an isolated complete storage root.

## Verify

`./.venv/Scripts/python.exe -m pytest tests -m "not real_services and not browser"`. Real services are opt-in and require cached models plus synthetic/consented audio. See docs/ for architecture/privacy/scoring. VoiceStudio is optional.

## License and public contents

Licensed under **AGPL-3.0-only**, matching Voice to Text; see [LICENSE](LICENSE). Dependencies and separately obtained models retain their own licenses.

The public snapshot includes application source, synthetic tests, migrations, setup scripts, the dependency lockfile and technical documentation. It excludes local agent sessions/configuration, personal task notes, credentials, settings, databases, recordings, transcripts, exports, logs, environments, model weights and generated artifacts. Never add private speech or provider credentials to an issue or commit.

**Evidence: 2/3 - Publication source inspected and 41 synthetic domain/listener tests passed on Python 3.12; full browser, real-provider and installed-app flows remain unverified for this snapshot.**
