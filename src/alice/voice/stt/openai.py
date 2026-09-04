"""STT provider using OpenAI's Whisper API."""

from __future__ import annotations

import io
import logging
import wave
from typing import AsyncIterator

from openai import AsyncOpenAI

from .base import SpeechRecognizer

log = logging.getLogger("alice.voice.stt.openai")


class OpenAISTT(SpeechRecognizer):
    """Transcribe speech using OpenAI's Whisper API."""

    def __init__(self, api_key: str, model: str = "whisper-1"):
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._listening = False

    async def start(self) -> None:
        self._listening = True
        log.info("stt_openai_started")

    async def stop(self) -> None:
        self._listening = False
        log.info("stt_openai_stopped")

    async def transcribe(self, audio: bytes) -> str:
        """Transcribe WAV audio bytes."""
        # Wrap bytes in a file-like object for the API
        file = io.BytesIO(audio)
        file.name = "audio.wav"

        response = await self._client.audio.transcriptions.create(
            file=file,
            model=self._model,
        )
        return response.text.strip()

    async def stream(self, audio_chunk: bytes) -> AsyncIterator[str]:
        """Stream transcription - buffers and yields on pause."""
        raise NotImplementedError("Streaming STT not yet implemented for OpenAI")
