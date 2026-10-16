# Architecture

A FastAPI transport calls session/job application services. Domain contracts validate evidence and lifecycle. SQLite stores durable metadata/structured payloads; audio/raw JSON are managed files. The original sibling STT package runs only in its installed interpreter through one owned framed worker. A single expensive-work lease coordinates ASR, local Ollama and optional external VoiceStudio. Provider results cannot directly advance session state. See adr/0001-initial-architecture.md.
