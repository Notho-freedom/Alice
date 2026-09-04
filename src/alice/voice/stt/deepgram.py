"""STT provider using Deepgram (cloud speech recognition).

Uses Deepgram's REST API for file-based transcription.
No SDK required — just HTTP requests with the API key.
Uses DEEPGRAM_API_KEY from environment.
"""

from __future__ import annotations

import io
import json
import logging
import urllib.request
from typing import AsyncIterator

from ... import config

log = logging.getLogger("alice.voice.stt.deepgram")


class DeepgramSTT:
    """Speech-to-text using Deepgram's REST API.

    No SDK required. Uses pure HTTP requests with the API key.
    Supports WAV audio transcription via the /v1/listen endpoint.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or config.DEEPGRAM_API_KEY
        self._listening = False

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
        """Transcribe WAV audio bytes via Deepgram REST API.

        Converts WAV (16kHz mono) to the format expected by Deepgram.
        Uses the /v1/listen endpoint with nova-2 model.
        """
        if not self.api_key:
            return ""

        if len(audio) == 0:
            return ""

        try:
            # Convert numpy array or bytes to raw bytes
            if hasattr(audio, 'tobytes'):
                raw_audio = audio.tobytes()
            elif isinstance(audio, (bytes, bytearray)):
                raw_audio = bytes(audio)
            else:
                raw_audio = bytes(audio)

            # Deepgram API endpoint
            url = "https://api.deepgram.com/v1/listen"
            params = "?model=nova-2&smart_format=true&punctuate=true&language=en"

            request = urllib.request.Request(
                url + params,
                data=raw_audio,
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