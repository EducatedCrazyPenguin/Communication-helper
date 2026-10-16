# Integrations

STT: original sibling voice to text checkout and its .venv, Qwen accuracy pipeline with recovery/readable/diarization disabled for one speaker. Offline guard and cache preflight prohibit downloads. Estimated versus aligned timings and alignment-score versus ASR confidence remain explicit. No Gemini fallback.

Coach: existing local Ollama qwen3.5:9b-q8_0, owned lazy server and release; explicitly configured OpenAI-compatible HTTPS is optional and needs cloud consent. No OMP credential reuse.

VoiceStudio: optional HTTP health/OpenAPI/profiles/speech contract; no source copying, automatic installation, service launch or cloning. User selects an authorized profile. Abandoning a client request does not prove external GPU work stopped. Installed official app may use an ephemeral port; never attach to Hindi Pilot or assume port 3900. Actual synthesis requires its external runtime/model/voice setup. MCP is available upstream but not required by Gym.
