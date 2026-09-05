"""TTS provider that uses the local Vocalis backend (FastAPI) for edge-tts.

This avoids edge-tts import/initialization latency by talking to the
local backend over HTTP on 127.0.0.1:8000.

The backend provides:
  POST /api/tts          — full MP3 (with cache)
  POST /api/tts/stream   — streaming MP3 (chunks)
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import aiohttp
import numpy as np

from ... import config
from .base import TextToSpeech

log = logging.getLogger("alice.voice.tts.edge")


class EdgeTTS(TextToSpeech):
    """Text-to-speech via the local Vocalis backend.

    No edge-tts import needed in Alice — the backend server handles it.
    Streaming endpoint allows near-real-time audio playback.
    """

    def __init__(
        self,
        backend_url: str = "http://127.0.0.1:8000",
        voice: str | None = None,
    ):
        self.backend_url = backend_url.rstrip("/")
        self.voice = voice or config.TTS_VOICE
        self._playing = False
        self._stop_flag = False
        self._session: aiohttp.ClientSession | None = None

        # Map Alice voice names to valid EdgeTTS voice ShortNames
        self._voice_map = {
            # French female voices (priority order)
            "marie": "fr-FR-DeniseNeural",
            "victoria": "fr-FR-VivienneMultilingualNeural",
            "anna": "fr-FR-EloiseNeural",
            # English female voices (fallback)
            "rachel": "en-US-AriaNeural",
            "alice": "en-GB-SoniaNeural",
            "laura": "en-US-JennyNeural",
            "sarah": "en-US-GuyNeural",
            "jessica": "en-US-EmmaNeural",
            "bella": "en-US-MichelleNeural",
            "lily": "en-GB-LibbyNeural",
        }

    def _resolve_voice(self) -> str:
        """Resolve voice name to EdgeTTS voice ID."""
        voice_lower = self.voice.lower() if self.voice else ""
        return self._voice_map.get(voice_lower, self.voice or "fr-FR-Elenora")

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def stop(self) -> None:
        self._stop_flag = True
        self._playing = False

    async def pause(self) -> None:
        await self.stop()

    async def resume(self) -> None:
        pass

    def is_speaking(self) -> bool:
        return self._playing

    async def speak(self, text: str, language: str | None = None) -> bool:
        """Speak full text via the backend /api/tts endpoint."""
        if not text:
            return True

        self._playing = True
        self._stop_flag = False

        try:
            session = await self._ensure_session()
            resolved_voice = self._resolve_voice()
            async with session.post(
                f"{self.backend_url}/api/tts",
                json={"text": text, "voice": resolved_voice},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Backend TTS error: {resp.status}")
                audio_data = await resp.read()

            if audio_data and not self._stop_flag:
                await self._play_mp3(audio_data)
            return True
        except Exception as e:
            log.error("edge_tts_speak_error", extra={"error": str(e), "type": type(e).__name__})
            return False
        finally:
            self._playing = False

    async def stream_speak(self, chunks: AsyncIterator[str], language: str | None = None) -> None:
        """Speak text from async chunk iterator, using the backend streaming endpoint.

        Collects the first meaningful chunk and sends it to the backend
        /api/tts/stream endpoint, then plays the streamed MP3.
        """
        collected = ""
        async for chunk in chunks:
            if self._stop_flag:
                break
            collected += chunk
            log.debug("tts_streaming", extra={"text": chunk})

        if collected and not self._stop_flag:
            await self._stream_from_backend(collected)

        # Fallback: if no text was collected, speak nothing
        if not collected and not self._stop_flag:
            return

    async def _stream_from_backend(self, text: str) -> None:
        """Stream audio from the backend's /api/tts/stream endpoint."""
        self._playing = True
        self._stop_flag = False
        audio_chunks: list[bytes] = []

        try:
            session = await self._ensure_session()
            resolved_voice = self._resolve_voice()
            async with session.post(
                f"{self.backend_url}/api/tts/stream",
                json={"text": text, "voice": resolved_voice},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Backend TTS stream error: {resp.status}")

                async for chunk in resp.content.iter_chunked(4096):
                    if self._stop_flag:
                        break
                    if chunk:
                        audio_chunks.append(chunk)

            if audio_chunks and not self._stop_flag:
                await self._play_mp3(b"".join(audio_chunks))
        except Exception as e:
            log.error("edge_tts_stream_error", extra={"error": str(e)})
        finally:
            self._playing = False

    async def _play_mp3(self, audio_data: bytes) -> None:
        """Decode and play MP3 audio data."""
        try:
            import soundfile as sf
            import io

            audio_array, sample_rate = sf.read(io.BytesIO(audio_data))

            if audio_array.dtype in (np.float32, np.float64):
                audio_int16 = (audio_array * 32767).astype(np.int16)
            else:
                audio_int16 = audio_array

            await self._play_int16(audio_int16, sample_rate, lambda: self._stop_flag)
        except ImportError:
            log.warning("soundfile_missing_using_pyttsx3_fallback")
            await self._play_via_pyttsx3(audio_data)
        except Exception as e:
            log.error("audio_playback_failed", extra={"error": str(e), "type": type(e).__name__})
            raise

    async def _play_via_pyttsx3(self, audio_data: bytes) -> None:
        """Fallback: pyttsx3 can't play MP3, so just log."""
        log.warning("cannot_play_mp3_with_pyttsx3", extra={
            "size": len(audio_data),
            "voice": self.voice,
        })

    async def close(self):
        """Clean up HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()