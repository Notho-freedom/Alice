"""TTS provider using local Pi TTS (coqui) via HTTP backend.

Falls back to pyttsx3 if the backend is unavailable.
Uses local backend on localhost.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import aiohttp
import numpy as np

from ... import config
from .base import TextToSpeech

log = logging.getLogger("alice.voice.tts.pi_tts")


class PiTTSTTS(TextToSpeech):
    """Text-to-speech via local Pi TTS (coqui) backend.

    Provides locally-run neural voices. Good fallback when cloud
    services are unavailable.
    """

    def __init__(self, backend_url: str | None = None):
        self.backend_url = (backend_url or config.PI_TTS_URL).rstrip("/")
        self._playing = False
        self._stop_flag = False
        self._session: aiohttp.ClientSession | None = None

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

    def _select_voice(self, language: str | None = None) -> str | None:
        """Select voice name appropriate for the language."""
        if not language:
            language = config.TTS_LANGUAGE_CODE

        lang_code = language.split("-")[0].lower() if language else "en"

        voice_map = {
            "en": "en_US/amy/medium",
            "fr": "fr/francesco/medium",
            "es": "es/es/davefx/medium",
            "de": "de/de/thorsten/medium",
            "zh": "zh/zh/zh_english/medium",
            "ja": None,  # Pi TTS doesn't have good Japanese
            "ko": None,
            "it": "it/it/mirko/medium",
            "pt": "pt/pt/pedro/medium",
            "ru": "ru/ru/irina/medium",
            "ar": None,
            "hi": None,
        }
        return voice_map.get(lang_code)

    async def speak(self, text: str, language: str | None = None) -> bool:
        """Speak text via Pi TTS backend. Returns True on success, False on failure."""
        if not text:
            return True

        self._playing = True
        self._stop_flag = False

        try:
            session = await self._ensure_session()

            # Check backend health
            async with session.get(
                f"{self.backend_url}/api/tts",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as health_resp:
                if health_resp.status == 404:
                    pass  # endpoint exists, just need POST

            voice = self._select_voice(language)

            payload = {"text": text}
            if voice:
                payload["voice"] = voice
            elif language:
                payload["language"] = language.split("-")[0].lower()

            async with session.post(
                f"{self.backend_url}/api/tts",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    log.error("pi_tts_api_error", extra={
                        "status": resp.status,
                        "body": error[:200],
                    })
                    return False

                content_type = resp.headers.get("content-type", "")
                audio_data = await resp.read()

            if audio_data and not self._stop_flag:
                await self._play_audio(audio_data, content_type)

            return True

        except Exception as e:
            log.error("pi_tts_error", extra={"error": str(e), "type": type(e).__name__})
            return False
        finally:
            self._playing = False

    async def stream_speak(self, chunks: AsyncIterator[str], language: str | None = None) -> None:
        """Speak text from async chunk iterator."""
        collected = ""
        async for chunk in chunks:
            if self._stop_flag:
                break
            collected += chunk

        if collected and not self._stop_flag:
            await self.speak(collected, language)

    async def _play_audio(self, audio_data: bytes, content_type: str = "") -> None:
        """Decode and play audio."""
        try:
            import soundfile as sf
            import io

            content_type_lower = content_type.lower()

            if "mp3" in content_type_lower or audio_data[:3] == b"\xff\xfb":
                audio_array, sample_rate = sf.read(io.BytesIO(audio_data))
            elif "wav" in content_type_lower or audio_data[:2] == b"RI":
                audio_array, sample_rate = sf.read(io.BytesIO(audio_data))
            else:
                audio_array, sample_rate = sf.read(io.BytesIO(audio_data))

            if audio_array.dtype in (np.float32, np.float64):
                audio_int16 = (audio_array * 32767).astype(np.int16)
            else:
                audio_int16 = audio_array

            await self._play_int16(audio_int16, sample_rate, lambda: self._stop_flag)
        except ImportError:
            log.warning("soundfile_not_available_for_pi_tts")
        except Exception as e:
            log.error("audio_playback_failed", extra={"error": str(e)})

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()