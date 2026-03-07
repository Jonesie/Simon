```text        
         ███████╗██╗███╗   ███╗ ██████╗ ███╗   ██╗
         ██╔════╝██║████╗ ████║██╔═══██╗████╗  ██║
         ███████╗██║██╔████╔██║██║   ██║██╔██╗ ██║
         ╚════██║██║██║╚██╔╝██║██║   ██║██║╚██╗██║
         ███████║██║██║ ╚═╝ ██║╚██████╔╝██║ ╚████║
         ╚══════╝╚═╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
```

Simon is a local voice assistant that listens to your speech, transcribes it with Whisper, gets a response from Ollama, and speaks the response back with Piper.

Flow:
1. Record microphone input (push-to-talk)
2. Transcribe audio with `whisper-cli`
3. Generate a response with Ollama
4. Convert response text to speech with Piper and play it locally

## Project Structure

- `simon.py`: Main application (recording, transcription, LLM call, TTS)
- `models/piper/`: Piper voice model files (`.onnx` and `.json`)
- `models/whisper/`: Whisper model files (e.g. `ggml-small.bin`)

## Dependencies

## Python

- Python **3.9+** recommended

Python packages used by `simon.py`:

- `numpy`
- `soundfile`
- `sounddevice`

Install with:

```bash
pip install -r requirements.txt
```

## System / CLI tools

Simon shells out to these commands:

- `whisper-cli` (from whisper.cpp)
- `ollama`
- `piper`
- `aplay` (usually from `alsa-utils` on Linux)

You must have these available on your `PATH`.

## Models

By default, Simon expects:

- Whisper model at `./models/whisper/ggml-small.bin`
- Piper model at `./models/piper/en_GB-alan-medium.onnx` (with matching `.json` config)

### 1) Create model directories

```bash
mkdir -p models/whisper models/piper
```

### 2) Download Whisper model (`ggml-small.bin`)

From the project root:

```bash
curl -L -o models/whisper/ggml-small.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin
```

### 3) Download Piper voice model (`en_GB-alan-medium`)

Piper needs both files in the same folder:

- `en_GB-alan-medium.onnx`
- `en_GB-alan-medium.onnx.json`

Download both:

```bash
curl -L -o models/piper/en_GB-alan-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx

curl -L -o models/piper/en_GB-alan-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json
```

### 4) Verify files exist

```bash
ls -lh models/whisper/ggml-small.bin models/piper/en_GB-alan-medium.onnx models/piper/en_GB-alan-medium.onnx.json
```

If your files are elsewhere, update the constants in `simon.py`:

- `WHISPER_MODEL`
- `PIPER_MODEL`

Or override at runtime with env vars:

- `SIMON_WHISPER_MODEL`
- `SIMON_PIPER_MODEL`

## Running

From the project root:

```bash
python simon.py
```

Controls:

- Press `ENTER` to start recording
- Press `ENTER` again to stop recording and process
- Type `q` then `ENTER` to quit

Startup summary includes:

- Active Whisper model path
- Active Piper model path

## Ollama Startup and Shutdown Behavior

- On startup, Simon checks whether the Ollama API is already reachable.
- If Ollama is already running, Simon **does not** start a second server and logs:
  - `Ollama service already running; skipping autostart.`
- If Ollama is not reachable and `SIMON_OLLAMA_AUTOSTART=1`, Simon attempts to run `ollama serve` and waits briefly for the API to come up.
- Prewarm behavior (`SIMON_OLLAMA_PREWARM=1`):
  - `Ollama prewarm complete.` if warmup succeeds
  - `Ollama service already running; skipping prewarm warning.` if API is already up but warmup request returns no response
  - `Ollama prewarm skipped (API unavailable; will lazy-load on first request).` if API never becomes reachable
  - `Ollama prewarm skipped (warmup request failed; will lazy-load on first request).` if API is up but warmup request fails
- Shutdown behavior:
  - Simon only shuts down Ollama on exit when **both** are true:
    - `SIMON_OLLAMA_SHUTDOWN_ON_EXIT=1`
    - Simon was the process that started `ollama serve`
  - If Ollama was already running before Simon started, Simon leaves it running.

## Parameters and Environment Variables

Simon is primarily configured via environment variables.

### General

- `SIMON_FAST_MODE` (default: `1`)
  - If `1`, uses lower-latency Ollama defaults.

- `SIMON_DEBUG` (default: inherits `SIMON_WEB_DEBUG`, else `0`)
  - If `1`, prints component timing metrics (recording, Whisper, Ollama API/CLI, web research, Piper synthesis/playback).

- `SIMON_TTS_SPEED` (default: `1.0`)
  - TTS speed multiplier; clamped to `0.5..3.0`.
  - Higher is faster speech.

- `SIMON_WHISPER_MODEL` (default: `./models/whisper/ggml-small.bin`)
  - Path to the Whisper model file used by `whisper-cli`.

- `SIMON_PIPER_MODEL` (default: `./models/piper/en_GB-alan-medium.onnx`)
  - Path to the Piper voice model file used by `piper`.

### Ollama

- `SIMON_OLLAMA_MODEL` (default: `llama3.2:3b`)
  - Model name used with Ollama.

- `SIMON_OLLAMA_API_URL` (default: `http://127.0.0.1:11434/api/generate`)
  - Ollama generate endpoint.
  - For remote Ollama, set this to your remote host (example: `http://192.168.1.50:11434/api/generate`).

- `SIMON_OLLAMA_TAGS_URL` (default: derived from `SIMON_OLLAMA_API_URL`)
  - Ollama tags endpoint used for health/model checks.
  - If not set, Simon automatically uses the same host as `SIMON_OLLAMA_API_URL` and path `/api/tags`.

- `SIMON_OLLAMA_KEEP_ALIVE` (default: `-1`)
  - Passed through to Ollama request payload.

- `SIMON_OLLAMA_TIMEOUT` (default: `120`)
  - Timeout in seconds for Ollama API calls.

- `SIMON_OLLAMA_NUM_PREDICT`
  - Default depends on `SIMON_FAST_MODE`:
    - Fast mode (`1`): `120`
    - Normal mode (`0`): `256`

- `SIMON_OLLAMA_NUM_CTX`
  - Default depends on `SIMON_FAST_MODE`:
    - Fast mode (`1`): `1536`
    - Normal mode (`0`): `4096`

- `SIMON_OLLAMA_PREWARM` (default: `1`)
  - If `1`, sends a warm-up request on startup.

- `SIMON_OLLAMA_AUTOSTART` (default: `1`)
  - If `1`, Simon attempts `ollama serve` when API is unavailable.

- `SIMON_OLLAMA_SHUTDOWN_ON_EXIT` (default: `0`)
  - If `1`, Simon stops the Ollama process it started when exiting.

### Web research

- `SIMON_WEB_RESEARCH` (default: `0`)
  - If `1`, Simon augments answers with web context.

- `SIMON_WEB_MAX_RESULTS` (default: `3`)
  - Maximum web results included in context.

- `SIMON_WEB_DEBUG` (default: `0`)
  - If `1`, prints outgoing web search request details (provider and URL/query).

## Internal Runtime Constants

These are not env vars, but fixed in code unless edited:

- `FS = 16000` (microphone sample rate)

## Example `.env`-style configuration

```bash
export SIMON_FAST_MODE=1
export SIMON_DEBUG=0
export SIMON_OLLAMA_MODEL="llama3.2:3b"
export SIMON_TTS_SPEED=1.1
export SIMON_WHISPER_MODEL="./models/whisper/ggml-small.bin"
export SIMON_PIPER_MODEL="./models/piper/en_GB-alan-medium.onnx"
export SIMON_WEB_RESEARCH=0
export SIMON_WEB_DEBUG=0
```

## Notes

- Simon currently uses no command-line flags; configuration is done through env vars and constants in `simon.py`.
- On first run with a new Ollama model, Simon can prompt to pull the model automatically.