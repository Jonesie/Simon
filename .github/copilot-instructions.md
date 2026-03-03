# Copilot Instructions for Simon

## Project purpose and architecture
- Simon is a **single-file local voice assistant** centered in `simon.py`.
- Runtime pipeline is fixed and sequential: microphone capture (`sounddevice`) → transcription (`whisper-cli`) → response generation (Ollama API, CLI fallback) → TTS (`piper`) + playback (`aplay`).
- Most behavior is configured via environment variables read at module import time; there are no argparse/CLI flags.

## Key files and boundaries
- `simon.py`: all orchestration, env config, service lifecycle, and user interaction loop.
- `README.md`: operational setup (models, required external tools, env vars).
- `requirements.txt`: Python libs only (`numpy`, `soundfile`, `sounddevice`).
- `models/`: expected local Whisper/Piper assets (ignored in git via `.gitignore`).

## External integrations (critical)
- Shell dependencies must exist on `PATH`: `whisper-cli`, `ollama`, `piper`, `aplay`.
- Ollama interaction pattern in code:
  - Prefer HTTP (`SIMON_OLLAMA_API_URL`) via `_ask_ollama_api`.
  - Fallback to `ollama run` in `_ask_ollama_cli` if API fails.
  - Optional autostart/shutdown logic tracks whether Simon started `ollama serve`.
- Web research is optional (`SIMON_WEB_RESEARCH=1`) and currently uses DuckDuckGo JSON API in `research_web_duckduckgo`.

## Coding patterns to preserve
- Keep new logic in small helper functions and call from `main()` loop.
- Preserve the startup summary + status-print style (plain prints and colorized helper `color_text`).
- Keep timing instrumentation behind `SIMON_DEBUG` using `_log_timing(...)`.
- Maintain resilient behavior: network/CLI failures should degrade gracefully, not crash interaction loop.
- Prefer extending existing env-var config style (`SIMON_*`) over hardcoded values.

## Developer workflows for agents
- Setup:
  - `pip install -r requirements.txt`
- Run:
  - `python simon.py`
- Typical verification after edits:
  - Start app, do one push-to-talk cycle, confirm transcription/response/TTS path still works.
  - If touching Ollama/web code, test both reachable and unreachable API behavior.

## Repo-specific guardrails
- Do not add heavy framework structure unless explicitly requested; this project is intentionally lightweight.
- Do not commit model binaries or generated audio artifacts; models are local runtime dependencies.
- Keep changes focused; avoid unrelated refactors in `simon.py` unless required for correctness.
