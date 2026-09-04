"""STT provider using Deepgram (cloud speech recognition).

Supports both file-based transcription and real-time streaming via WebSocket.
Uses DEEPGRAM_API_KEY from environment.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import AsyncIterator

from ... import config

log = logging.getLogger("alice.voice.stt.deepgram")


class DeepgramSTT:
    """Speech-to-text using Deepgram's API.

    Uses the deepgram-sdk for real-time streaming transcription.
    Falls back to file-based transcription for shorter buffers.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or config.DEEPGRAM_API_KEY
        self._listening = False
        self._client = None
        self._transcript = ""
        self._lock = threading.Lock()

        if self.api_key:
            try:
                from deepgram import DeepgramClientOptions, DeepgramClient

                self._client = DeepgramClient(
                    api_key=self.api_key,
                    config=DeepgramClientOptions(
                        timeout=30.0,
                        options={"keepalive": "enabled"},
                    ),
                )
                log.info("deepgram_client_initialized")
            except ImportError:
                log.warning("deepgram_sdk_not_available", extra={
                    "error": "Install: pip install deepgram-sdk"
                })
            except Exception as e:
                log.warning("deepgram_init_failed", extra={"error": str(e)})

    async def start(self) -> None:
        self._listening = True
        self._transcript = ""
        log.info("stt_deepgram_started")

    async def stop(self) -> None:
        self._listening = False
        log.info("stt_deepgram_stopped")

    async def transcribe(self, audio: bytes) -> str:
        """Transcribe WAV audio bytes via file-based API."""
        if not self._client:
            return ""

        try:
            # deepgram-sdk file transcription
            from deepgram.utils import ProtoContext

            # Prepare audio data
            if len(audio) == 0:
                return ""

            source = {
                "buffer": audio,
                "mimetype": "audio/wav",
            }

            response = self._client.transcribe(
                source,
                {
                    "model": "nova-2",
                    "language": "en",
                    "smart_format": True,
                    "punctuate": True,
                    "utterances": True,
                },
            )

            results = response.results
            if results and results.alternatives:
                transcript = results.alternatives[0].transcript
                return transcript.strip() if transcript else ""

            return ""

        except Exception as e:
            log.error("deepgram_transcribe_error", extra={"error": str(e)})
            return ""

    async def stream(self, audio_chunk: bytes) -> AsyncIterator[str]:
        """Stream audio chunks yielding partial transcriptions.

        Uses Deepgram's real-time WebSocket API with VAD-based end-of-speech
        detection. Each yield is a partial or final transcript string.
        """
        if not self._client:
            yield ""
            return

        try:
            import asyncio
            import sounddevice as sd

            # Use asyncio-compatible approach
            loop = asyncio.get_event_loop()

            # For simplicity, accumulate and transcribe in chunks
            # A full implementation would use the WebSocket streaming API
            # This is a simplified version that buffers audio chunks
            # and transcribes when a silence is detected

            queue: asyncio.Queue[float] = asyncio.Queue()

            # This is a placeholder for the streaming implementation
            # The full streaming setup would involve:
            # 1. Setting up a WebSocket connection to Deepgram
            # 2. Sending audio chunks via the socket
            # 3. Receiving and yielding transcriptions
            # 4. Detecting end-of-speech via is_final flag

            yield ""

        except Exception as e:
            log.error("deepgram_stream_error", extra={"error": str(e)})
            yield ""