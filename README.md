# Alice

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![STT](https://img.shields.io/badge/STT-Deepgram-111111?logo=deepgram&logoColor=white)](https://deepgram.com/)
[![TTS](https://img.shields.io/badge/TTS-Multi--provider-6C47FF)](#providers)
[![Kilo Code](https://img.shields.io/badge/Brain-Kilo%20Code-111111)](https://kilo.ai/)
[![Tests](https://img.shields.io/badge/Tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://pytest.org/)

Local voice assistant powered by Kilo Code.

Alice captures microphone input, transcribes speech, sends it to Kilo, and speaks the response back. It supports push-to-talk, continuous conversation with wake-word detection, selective barge-in, and echo suppression.

## Requirements

- Python 3.11+
- Kilo CLI installed and available in `PATH`
- Audio input/output devices
- Optional: `pvporcupine` + `PICOVOICE_ACCESS_KEY` for real wake-word detection

## Setup

```bash
python3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e . --no-deps
```

Configure environment variables in `.env`:

```bash
cp .env.example .env
```

Minimum required keys:
- `DEEPGRAM_API_KEY` for speech-to-text
- At least one TTS provider key if cloud TTS is desired

## CLI

```bash
alice voice --ptt
alice voice --continuous
alice terminal
alice doctor
```

### Modes

- **Push-to-talk**: hold the configured key to speak
- **Continuous**: conversation mode with VAD and optional wake word
- **Terminal**: text interface without audio

## Audio

Alice auto-detects audio devices on startup. You can override with:

```bash
AUDIO_INPUT_DEVICE=1
AUDIO_OUTPUT_DEVICE=3
```

Run `alice doctor` to verify device selection.

## Wake Word

By default Alice uses a lightweight VAD-based wake-word fallback. For real keyword spotting:

```bash
pip install pvporcupine
```

Set `PICOVOICE_ACCESS_KEY` in `.env`. The wake-word phrase is configured via `WAKE_WORD`.

## Barge-in / Echo

Alice can detect when you speak while it is talking and interrupt itself:

- Ring buffer keeps the last seconds of audio for pre-roll capture
- Barge-in detector uses 3 levels: `NONE → CANDIDATE → CONFIRMED`
- Echo suppression attenuates microphone input during TTS playback
- Optional LMS-based echo cancellation: set `ECHO_CANCELLATION_ENABLED=true`

## Providers

### STT

- Deepgram (default, streaming)
- OpenAI Whisper

### TTS

Priority chain: ElevenLabs → VAPI → Edge → Pi TTS → pyttsx3

Voice selection is centralized and supports dynamic discovery from provider APIs when keys are configured.

## Config

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `STT_PROVIDER` | Speech provider | `deepgram` |
| `TTS_PROVIDER` | TTS provider chain | `auto` |
| `TTS_LANGUAGE` | Default TTS language | `fr` |
| `WAKE_WORD` | Wake-word phrase | `hey assistant` |
| `BARGE_IN_ENABLED` | Enable barge-in | `true` |
| `ECHO_SUPPRESSION_ENABLED` | Enable echo suppression | `true` |
| `ECHO_CANCELLATION_ENABLED` | Enable LMS AEC | `false` |
| `AUDIO_FRAME_MS` | Audio frame size | `30` |

## Development

Run tests:

```bash
python3.11 -m pytest tests/ -v
```

Architecture overview:
- `src/alice/voice/assistant.py` — main voice orchestrator
- `src/alice/voice/input.py` — audio capture + ring buffer
- `src/alice/voice/vad.py` — voice activity detection
- `src/alice/voice/barge_in.py` — selective barge-in detector
- `src/alice/voice/stt/deepgram.py` — Deepgram streaming STT
- `src/alice/voice/tts/` — TTS provider cascade
- `src/alice/kilo/bridge.py` — Kilo session/streaming contract
