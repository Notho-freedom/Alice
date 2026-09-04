"""STT provider using Deepgram (cloud speech recognition).

Uses Deepgram's REST API for file-based transcription.
No SDK required — just HTTP requests with the API key.
Uses DEEPGRAM_API_KEY from environment.
"""

from __future__ import annotations

import io
import json
import logging
import struct
import urllib.request
import wave
from typing import AsyncIterator

from ... import config

log = logging.getLogger("alice.voice.stt.deepgram")


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Wrap raw 16-bit PCM data in a WAV header."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(pcm_data)
    return buf.getvalue()


class DeepgramSTT:
    """Speech-to-text using Deepgram's REST API.

    No SDK required. Uses urllib HTTP requests with the API key.
    Audio data is expected as raw 16-bit PCM (16kHz mono).
    The provider wraps it in a WAV header before sending to Deepgram.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or config.DEEPGRAM_API_KEY
        self._listening = False
        self.detected_language: str | None = None

        if self.api_key:
            log.info("deepgram_client_ready")
        else:
            log.warning("deepgram_no_api_key")

    async def start(self) -> None:
        self._listening = True
        log.info("stt_deepgram_started")

    async def stop(self) -> None:
        self._listening = False
        log.info("stt_deepgram_stopped")

    async def transcribe(self, audio: bytes) -> str:
        """Transcribe audio bytes via Deepgram REST API.

        Accepts raw 16-bit PCM audio (16kHz, mono) and wraps it
        in a WAV header before sending to the API.
        """
        if not self.api_key:
            return ""

        if len(audio) == 0:
            return ""

        try:
            # Convert audio to raw bytes
            if hasattr(audio, "tobytes"):
                raw_audio = audio.tobytes()
            elif isinstance(audio, (bytes, bytearray)):
                raw_audio = bytes(audio)
            else:
                raw_audio = bytes(audio)

            # Wrap in WAV header for Deepgram API
            wav_data = _pcm_to_wav(
                raw_audio,
                sample_rate=config.AUDIO_SAMPLE_RATE,
                channels=config.AUDIO_CHANNELS,
            )

            # Deepgram API endpoint
            url = "https://api.deepgram.com/v1/listen"
            params = "?model=nova-2&smart_format=true&punctuate=true"

            if config.STT_DETECT_LANGUAGE:
                params += "&detect_language=true"
            elif config.STT_LANGUAGE:
                params += f"&language={config.STT_LANGUAGE}"

            request = urllib.request.Request(
                url + params,
                data=wav_data,
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "audio/wav",
                    "Accept": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))

            # Extract transcript
            results = result.get("results", {})
            channels = results.get("channels", [])
            
            # Store detected language for downstream TTS voice selection
            detected = results.get("detected_language")
            if detected:
                self.detected_language = detected
                log.info("language_detected", extra={"language": detected})
                if not config.TTS_LANGUAGE:
                    config._tts_language = detected
            
            if channels:
                alternatives = channels[0].get("alternatives", [])
                if alternatives:
                    transcript = alternatives[0].get("transcript", "")
                    return transcript.strip() if transcript else ""

            return ""

        except Exception as e:
            log.error("deepgram_transcribe_error", extra={"error": str(e)})
            return ""

    async def stream(self, audio_chunk: bytes) -> AsyncIterator[str]:
        """Stream transcription — not implemented for REST API.

        Falls back to calling transcribe() on accumulated chunks.
        """
        result = await self.transcribe(audio_chunk)
        if result:
            yield result