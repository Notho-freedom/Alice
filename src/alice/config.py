"""Environment-based configuration for Alice."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# ── Kilo ────────────────────────────────────────────────────────────────
KILO_HOST = os.getenv("KILO_HOST", "127.0.0.1")
KILO_PORT = int(os.getenv("KILO_PORT", "4096"))
KILO_BASE_URL = os.getenv("KILO_BASE_URL", f"http://{KILO_HOST}:{KILO_PORT}")
KILO_DIRECTORY = os.getenv("KILO_DIRECTORY", os.getcwd())
KILO_AUTO_APPROVE = os.getenv("KILO_AUTO_APPROVE", "false").lower() in ("1", "true", "yes")

# ── Audio ───────────────────────────────────────────────────────────────
AUDIO_INPUT_DEVICE = int(os.getenv("AUDIO_INPUT_DEVICE", "1"))
AUDIO_OUTPUT_DEVICE = int(os.getenv("AUDIO_OUTPUT_DEVICE", "3"))
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_CHANNELS = int(os.getenv("AUDIO_CHANNELS", "1"))
AUDIO_CHUNK_SIZE = int(os.getenv("AUDIO_CHUNK_SIZE", "1024"))
AUDIO_FRAME_MS = int(os.getenv("AUDIO_FRAME_MS", "30"))

# ── VAD ─────────────────────────────────────────────────────────────────
VAD_SENSITIVITY = int(os.getenv("VAD_SENSITIVITY", "2"))  # 0-3
VAD_MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS", "80"))
VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "400"))

# ── Barge-in ───────────────────────────────────────────────────────────
BARGE_IN_ENABLED = os.getenv("BARGE_IN_ENABLED", "true").lower() in ("1", "true", "yes")
BARGE_IN_CANDIDATE_MS = int(os.getenv("BARGE_IN_CANDIDATE_MS", "300"))
BARGE_IN_MIN_SPEECH_RATIO = float(os.getenv("BARGE_IN_MIN_SPEECH_RATIO", "0.15"))
BARGE_IN_WINDOW_MS = int(os.getenv("BARGE_IN_WINDOW_MS", "1000"))

# ── Echo suppression ───────────────────────────────────────────────────
ECHO_SUPPRESSION_ENABLED = os.getenv("ECHO_SUPPRESSION_ENABLED", "true").lower() in ("1", "true", "yes")
ECHO_SUPPRESSION_FACTOR = float(os.getenv("ECHO_SUPPRESSION_FACTOR", "0.05"))
ECHO_SUPPRESSION_MAX_ATTENUATION = float(os.getenv("ECHO_SUPPRESSION_MAX_ATTENUATION", "0.9"))
ECHO_CANCELLATION_ENABLED = os.getenv("ECHO_CANCELLATION_ENABLED", "false").lower() in ("1", "true", "yes")

# ── Wake Word ───────────────────────────────────────────────────────────
WAKE_WORD = os.getenv("WAKE_WORD", "hey assistant")

# ── STT ─────────────────────────────────────────────────────────────────
STT_PROVIDER = os.getenv("STT_PROVIDER", "deepgram")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
STT_DETECT_LANGUAGE = os.getenv("STT_DETECT_LANGUAGE", "true").lower() in ("1", "true", "yes")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "fr")  # Default to French

# ── TTS ─────────────────────────────────────────────────────────────────
# Priority chain: ElevenLabs (key1/key2) → VAPI → Edge → Pi TTS → pyttsx3
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "auto")
TTS_RATE = int(os.getenv("TTS_RATE", "200"))
TTS_VOLUME = float(os.getenv("TTS_VOLUME", "1.0"))
TTS_VOICE = os.getenv("TTS_VOICE", "marie")  # French female voice default
TTS_BACKEND_URL = os.getenv("TTS_BACKEND_URL", "http://127.0.0.1:8000")
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "fr")  # Default to French
TTS_LANGUAGE_CODE = TTS_LANGUAGE  # ISO 639-1 code (e.g., "fr", "en", "es")

# ElevenLabs (cloud, high quality) — 2 keys for credit fallback
ELEVENLABS_API_KEY1 = os.getenv("ELEVENLABS_API_KEY1", "")
ELEVENLABS_API_KEY2 = os.getenv("ELEVENLABS_API_KEY2", "")
ELEVENLABS_API_KEY = ELEVENLABS_API_KEY1  # Primary key
ELEVENLABS_VOICE = os.getenv("ELEVENLABS_VOICE", "Rachel")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")

# VAPI
VAPI_API_KEY = os.getenv("VAPI_API_KEY", "")
VAPI_VOICE_ID = os.getenv("VAPI_VOICE_ID", "")
VAPI_BASE_URL = os.getenv("VAPI_BASE_URL", "https://api.vapi.ai")

# Pi TTS (coqui/edge-tts-pi) — local backend
PI_TTS_URL = os.getenv("PI_TTS_URL", "http://127.0.0.1:8001")

# ── Interrupt ───────────────────────────────────────────────────────────
INTERRUPT_THRESHOLD_MS = int(os.getenv("INTERRUPT_THRESHOLD_MS", "300"))

# ── Logging ─────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "")

# ── Startup validation ──────────────────────────────────────────────────
def validate_environment() -> list[str]:
    """Check that required components are available. Returns list of error messages."""
    errors = []

    import shutil
    if not shutil.which("kilo"):
        errors.append("kilo CLI not found in PATH")

    return errors


def ensure_directories():
    """Ensure runtime directories exist."""
    for d in [Path.home() / ".alice"]:
        d.mkdir(parents=True, exist_ok=True)
