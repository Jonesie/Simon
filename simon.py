import numpy as np
import soundfile as sf
import sounddevice as sd
import tempfile
import subprocess
import os
import json
import urllib.request
import urllib.error
import urllib.parse
import time

# -------------------
# Paths / Models
# -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_WHISPER_MODEL = os.path.join(BASE_DIR, "models", "whisper", "ggml-small.bin")
DEFAULT_PIPER_MODEL = os.path.join(BASE_DIR, "models", "piper", "en_GB-alan-medium.onnx")

WHISPER_MODEL = os.path.expanduser(os.getenv("SIMON_WHISPER_MODEL", DEFAULT_WHISPER_MODEL))
PIPER_MODEL = os.path.expanduser(os.getenv("SIMON_PIPER_MODEL", DEFAULT_PIPER_MODEL))
WEB_RESEARCH_ENABLED = os.getenv("SIMON_WEB_RESEARCH", "0") == "1"
WEB_DEBUG_ENABLED = os.getenv("SIMON_WEB_DEBUG", "0") == "1"
WEB_MAX_RESULTS = int(os.getenv("SIMON_WEB_MAX_RESULTS", "3"))
ASSISTANT_NAME = (os.getenv("SIMON_ASSISTANT_NAME", "Simon") or "Simon").strip()
OLLAMA_MODEL = os.getenv("SIMON_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
SIMON_FAST_MODE = os.getenv("SIMON_FAST_MODE", "1") == "1"


def _debug_web_request(provider, url, payload=None):
    if not WEB_DEBUG_ENABLED:
        return

    print(color_text(f"[WEB DEBUG] Provider: {provider}", YELLOW))
    print(color_text(f"[WEB DEBUG] URL: {url}", YELLOW))

    if payload is not None:
        safe_payload = dict(payload)
        if "api_key" in safe_payload and safe_payload["api_key"]:
            safe_payload["api_key"] = "***redacted***"
        print(color_text(f"[WEB DEBUG] Payload: {json.dumps(safe_payload)}", YELLOW))


def _default_ollama_tags_url(api_url):
    fallback = "http://127.0.0.1:11434/api/tags"
    try:
        parsed = urllib.parse.urlsplit(api_url)
        if not parsed.scheme or not parsed.netloc:
            return fallback
        return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/api/tags", "", ""))
    except Exception:
        return fallback


OLLAMA_API_URL = os.getenv("SIMON_OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_TAGS_URL = os.getenv("SIMON_OLLAMA_TAGS_URL", _default_ollama_tags_url(OLLAMA_API_URL))
OLLAMA_KEEP_ALIVE = os.getenv("SIMON_OLLAMA_KEEP_ALIVE", "-1")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("SIMON_OLLAMA_TIMEOUT", "120"))
SIMON_OLLAMA_PREWARM = os.getenv("SIMON_OLLAMA_PREWARM", "1") == "1"
SIMON_OLLAMA_AUTOSTART = os.getenv("SIMON_OLLAMA_AUTOSTART", "1") == "1"
SIMON_OLLAMA_SHUTDOWN_ON_EXIT = os.getenv("SIMON_OLLAMA_SHUTDOWN_ON_EXIT", "0") == "1"

OLLAMA_SERVER_PROCESS = None
OLLAMA_STARTED_BY_SIMON = False


def _env_int(name, fast_default, normal_default):
    default_value = str(fast_default if SIMON_FAST_MODE else normal_default)
    try:
        return int(os.getenv(name, default_value))
    except ValueError:
        return int(default_value)


OLLAMA_NUM_PREDICT = _env_int("SIMON_OLLAMA_NUM_PREDICT", 120, 256)
OLLAMA_NUM_CTX = _env_int("SIMON_OLLAMA_NUM_CTX", 1536, 4096)


def _env_float(name, default_value):
    try:
        return float(os.getenv(name, str(default_value)))
    except ValueError:
        return float(default_value)


SIMON_TTS_SPEED = max(0.5, min(3.0, _env_float("SIMON_TTS_SPEED", 1.0)))
PIPER_LENGTH_SCALE = 1.0 / SIMON_TTS_SPEED

# Audio settings
FS = 16000  # sample rate

# Terminal colors
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"
YELLOW = "\033[33m"


def color_text(text, color):
    return f"{color}{text}{RESET}"

# -------------------
# Push-to-talk recording (ENTER)
# -------------------


def record_push_to_talk():
    frames = []

    try:
        print(color_text("Recording... press ENTER to stop.", CYAN), end="", flush=True)
        with sd.InputStream(samplerate=FS, channels=1, callback=lambda indata, f, t, s: frames.append(indata.copy())):
            input()
        print("\r" + color_text("Recording complete.                      ", GREEN))
    except Exception:
        print(color_text("Audio capture failed.", MAGENTA))
        return None

    if not frames:
        print(color_text("No audio captured.", MAGENTA))
        return None

    audio_data = np.concatenate(frames, axis=0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        filename = f.name
        sf.write(filename, audio_data, FS)
    return filename

# -------------------
# Whisper transcription
# -------------------


def transcribe(filename):
    result = subprocess.run(
        ["whisper-cli", "-m", WHISPER_MODEL, filename],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

# -------------------
# Ollama AI response
# -------------------


def ask_ollama(prompt):
    system_prompt = f"""
You are {ASSISTANT_NAME}, a calm and perceptive AI presence.
You respond conversationally and naturally.
"""
    full_prompt = f"User: {prompt}\n{ASSISTANT_NAME}:"

    response = _ask_ollama_api(full_prompt, system_prompt)
    if response:
        return response

    return _ask_ollama_cli(system_prompt + "\n" + full_prompt)


def _ask_ollama_cli(full_prompt):
    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=full_prompt,
        capture_output=True,
        text=True
    )
    return result.stdout.strip()


def _ask_ollama_api(prompt, system_prompt, num_predict_override=None, allow_empty_response=False):
    num_predict = num_predict_override if num_predict_override is not None else OLLAMA_NUM_PREDICT

    payload = {
        "model": OLLAMA_MODEL,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "num_predict": num_predict,
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }

    req = urllib.request.Request(
        OLLAMA_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
        data = json.loads(body)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None

    response_text = (data.get("response", "") or "").strip()
    if response_text:
        return response_text
    if allow_empty_response:
        return ""
    return None


def prewarm_ollama():
    was_running = is_ollama_api_available()

    if not ensure_ollama_api_ready():
        return "api_unavailable"

    system_prompt = "You are a warmup request. Respond minimally."
    prompt = "warmup"
    response = _ask_ollama_api(
        prompt,
        system_prompt,
        num_predict_override=1,
        allow_empty_response=True,
    )
    if response is None:
        if was_running:
            return "already_running"
        return "request_failed"
    return "ok"


def is_ollama_model_available():
    if not ensure_ollama_api_ready():
        return None

    req = urllib.request.Request(OLLAMA_TAGS_URL, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            body = response.read().decode("utf-8")
        data = json.loads(body)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None

    models = data.get("models", []) or []
    names = {item.get("name", "") for item in models if isinstance(item, dict)}
    return OLLAMA_MODEL in names


def confirm_and_pull_ollama_model():
    choice = input(color_text(f"Model '{OLLAMA_MODEL}' is missing. Pull it now? [Y/n]: ", CYAN)).strip().lower()
    if choice not in ("", "y", "yes"):
        print(color_text("Skipping model pull. You can pull it manually later.", MAGENTA))
        return False

    print(color_text(f"Pulling {OLLAMA_MODEL}...", CYAN))
    result = subprocess.run(["ollama", "pull", OLLAMA_MODEL], capture_output=True, text=True)
    if result.returncode == 0:
        print(color_text("Model pull complete.", GREEN))
        return True

    print(color_text("Model pull failed.", MAGENTA))
    if result.stderr.strip():
        print(color_text(result.stderr.strip(), MAGENTA))
    return False


def is_ollama_api_available():
    req = urllib.request.Request(OLLAMA_TAGS_URL, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=3):
            return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return False


def start_ollama_service():
    global OLLAMA_SERVER_PROCESS
    global OLLAMA_STARTED_BY_SIMON

    if is_ollama_api_available():
        OLLAMA_STARTED_BY_SIMON = False
        print(color_text("Ollama service already running; skipping autostart.", GREEN))
        return True

    try:
        OLLAMA_SERVER_PROCESS = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        OLLAMA_STARTED_BY_SIMON = True
        return True
    except Exception:
        OLLAMA_STARTED_BY_SIMON = False
        return False


def shutdown_ollama_service_if_started_by_simon():
    if not SIMON_OLLAMA_SHUTDOWN_ON_EXIT:
        return

    if not OLLAMA_STARTED_BY_SIMON:
        return

    if OLLAMA_SERVER_PROCESS is None:
        return

    if OLLAMA_SERVER_PROCESS.poll() is not None:
        return

    print(color_text("Stopping Ollama service started by Simon...", CYAN))
    OLLAMA_SERVER_PROCESS.terminate()
    try:
        OLLAMA_SERVER_PROCESS.wait(timeout=4)
    except subprocess.TimeoutExpired:
        OLLAMA_SERVER_PROCESS.kill()
    print(color_text("Ollama service stopped.", GREEN))


def ensure_ollama_api_ready(wait_seconds=8):
    if is_ollama_api_available():
        return True

    if not SIMON_OLLAMA_AUTOSTART:
        return False

    start_ollama_service()

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if is_ollama_api_available():
            return True
        time.sleep(0.4)

    return False


def ask_ollama_with_research(prompt, research_context):
    system_prompt = f"""
You are {ASSISTANT_NAME}, a calm and perceptive AI presence.
You respond conversationally and naturally.
If web research context is provided, use it as primary grounding.
If information is uncertain or missing, say so clearly.
"""
    full_prompt = (
        f"Web research context:\n{research_context}\n"
        + f"\nUser: {prompt}\n{ASSISTANT_NAME}:"
    )

    response = _ask_ollama_api(full_prompt, system_prompt)
    if response:
        return response

    return _ask_ollama_cli(system_prompt + "\n" + full_prompt)


def research_web_duckduckgo(query, max_results=3):
    url = (
        "https://api.duckduckgo.com/?"
        + urllib.parse.urlencode({
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        })
    )

    _debug_web_request("DuckDuckGo", url)

    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            body = response.read().decode("utf-8")
        data = json.loads(body)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return []

    cleaned = []

    abstract_text = (data.get("AbstractText") or "").strip()
    abstract_url = data.get("AbstractURL", "")
    abstract_title = data.get("Heading") or "DuckDuckGo Abstract"
    if abstract_text and abstract_url:
        cleaned.append({
            "title": abstract_title,
            "url": abstract_url,
            "content": abstract_text,
        })

    related_topics = data.get("RelatedTopics", [])
    for item in related_topics:
        if len(cleaned) >= max_results:
            break

        if "Topics" in item:
            for sub_item in item.get("Topics", []):
                if len(cleaned) >= max_results:
                    break
                text = (sub_item.get("Text") or "").strip()
                link = sub_item.get("FirstURL", "")
                if text and link:
                    cleaned.append({
                        "title": text.split(" - ")[0][:120] or "DuckDuckGo Result",
                        "url": link,
                        "content": text,
                    })
        else:
            text = (item.get("Text") or "").strip()
            link = item.get("FirstURL", "")
            if text and link:
                cleaned.append({
                    "title": text.split(" - ")[0][:120] or "DuckDuckGo Result",
                    "url": link,
                    "content": text,
                })

    return cleaned[:max_results]


def build_research_context(query):
    provider = "DuckDuckGo"
    results = research_web_duckduckgo(query, WEB_MAX_RESULTS)

    if not results:
        return "No web sources were retrieved."

    chunks = []
    chunks.append(f"Provider: {provider}")
    for idx, item in enumerate(results, start=1):
        snippet = item["content"][:600] if item["content"] else "(no snippet returned)"
        chunks.append(
            f"[{idx}] {item['title']}\nURL: {item['url']}\nSnippet: {snippet}"
        )
    return "\n\n".join(chunks)

# -------------------
# Piper speech
# -------------------


def speak(text):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_file = f.name
    subprocess.run(
        [
            "piper",
            "--model",
            PIPER_MODEL,
            "--output_file",
            wav_file,
            "--length_scale",
            f"{PIPER_LENGTH_SCALE:.3f}",
        ],
        input=text.encode()
    )
    subprocess.run(["aplay", wav_file])
    os.remove(wav_file)

# -------------------
# Main loop
# -------------------


def main():
    if WEB_RESEARCH_ENABLED:
        research_status = "ON (DuckDuckGo)"
    else:
        research_status = "OFF"

    banner = (
        "\n"
        "  ███████╗██╗███╗   ███╗ ██████╗ ███╗   ██╗\n"
        "  ██╔════╝██║████╗ ████║██╔═══██╗████╗  ██║\n"
        "  ███████╗██║██╔████╔██║██║   ██║██╔██╗ ██║\n"
        "  ╚════██║██║██║╚██╔╝██║██║   ██║██║╚██╗██║\n"
        "  ███████║██║██║ ╚═╝ ██║╚██████╔╝██║ ╚████║\n"
        "  ╚══════╝╚═╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝\n"
    )
    print(color_text(banner, BOLD + MAGENTA))
    print("Startup summary")
    print(f"- Assistant: {ASSISTANT_NAME}")
    print(f"- Ollama model: {OLLAMA_MODEL}")
    print(f"- Ollama fast mode: {'ON' if SIMON_FAST_MODE else 'OFF'}")
    print(f"- Ollama options: num_predict={OLLAMA_NUM_PREDICT}, num_ctx={OLLAMA_NUM_CTX}, keep_alive={OLLAMA_KEEP_ALIVE}")
    print(f"- Ollama prewarm: {'ON' if SIMON_OLLAMA_PREWARM else 'OFF'}")
    print(f"- Ollama autostart: {'ON' if SIMON_OLLAMA_AUTOSTART else 'OFF'}")
    print(f"- Ollama shutdown on exit: {'ON' if SIMON_OLLAMA_SHUTDOWN_ON_EXIT else 'OFF'}")
    print(f"- TTS speed: {SIMON_TTS_SPEED:.2f}x (SIMON_TTS_SPEED, higher is faster)")
    print(f"- Whisper model path: {WHISPER_MODEL}")
    print(f"- Piper model path: {PIPER_MODEL}")
    print(f"- Web research: {research_status}")
    print("- Controls: ENTER start/stop, Q then ENTER quit")

    model_available = is_ollama_model_available()
    if model_available is True:
        print(color_text("Ollama model check: ready.", GREEN))
    elif model_available is False:
        print(color_text(f"Ollama model check: '{OLLAMA_MODEL}' not installed. Run: ollama pull {OLLAMA_MODEL}", MAGENTA))
        pulled = confirm_and_pull_ollama_model()
        if pulled:
            model_available = is_ollama_model_available()
            if model_available is True:
                print(color_text("Ollama model check: ready after pull.", GREEN))
            else:
                print(color_text("Model was pulled but still not visible yet; continuing.", MAGENTA))
    else:
        print(color_text("Ollama model check: unable to query Ollama API (will try on first request).", MAGENTA))

    if SIMON_OLLAMA_PREWARM:
        print(color_text("Prewarming Ollama model...", CYAN))
        prewarm_status = prewarm_ollama()
        if prewarm_status == "ok":
            print(color_text("Ollama prewarm complete.", GREEN))
        elif prewarm_status == "already_running":
            print(color_text("Ollama service already running; skipping prewarm warning.", GREEN))
        elif prewarm_status == "api_unavailable":
            print(color_text("Ollama prewarm skipped (API unavailable; will lazy-load on first request).", MAGENTA))
        else:
            print(color_text("Ollama prewarm skipped (warmup request failed; will lazy-load on first request).", MAGENTA))

    try:
        while True:
            choice = input(color_text("\nPress ENTER to talk (or Q then ENTER to quit): ", CYAN)).strip().lower()
            if choice == "q":
                print("Exiting Simon.")
                break

            filename = record_push_to_talk()
            if not filename:
                print("No speech detected, try again.")
                continue
            user_text = transcribe(filename)
            if user_text:
                print(color_text("You:", YELLOW), user_text)
                if WEB_RESEARCH_ENABLED:
                    research_context = build_research_context(user_text)
                    response = ask_ollama_with_research(user_text, research_context)
                else:
                    response = ask_ollama(user_text)
                print(color_text(f"{ASSISTANT_NAME}:", GREEN), response)
                speak(response)
            else:
                print(color_text("No speech detected, try again.", MAGENTA))
            os.remove(filename)
    except KeyboardInterrupt:
        print("\nExiting Simon.")
    finally:
        shutdown_ollama_service_if_started_by_simon()


if __name__ == "__main__":
    main()
