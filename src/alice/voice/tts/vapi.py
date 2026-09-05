"""TTS provider using VAPI API.

High-quality voices via VAPI's text-to-speech endpoint.
Uses VAPI_API_KEY from environment.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import aiohttp
import numpy as np

from ... import config
from .base import TextToSpeech

log = logging.getLogger("alice.voice.tts.vapi")


class VAPITTS(TextToSpeech):
    """Text-to-speech via VAPI cloud API.

    Provides neural voices with multilingual support.
    """

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
    ):
        self.api_key = api_key or config.VAPI_API_KEY
        self.voice_id = voice_id or config.VAPI_VOICE_ID
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

    def _select_voice(self, language: str | None = None) -> str:
        """Select an appropriate voice for the given language."""
        if not language:
            language = config.TTS_LANGUAGE_CODE

        lang_code = language.split("-")[0].lower() if language else "en"

        # VAPI voice selection by language
        voice_map = {
            "en": self.voice_id or "9b7d887d-3394-4b5e-9adf-021287086938",
            "fr": self.voice_id or "c2b0de7d-3394-4b5e-9adf-021287086938",
            "es": "3b01b5e0-3394-4b5e-9adf-021287086938",
            "de": "08d47bdc-3394-4b5e-9adf-021287086938",
            "zh": "746116a3-3394-4b5e-9adf-021287086938",
            "ja": "09b931a0-3394-4b5e-9adf-021287086938",
            "ko": "47653a10-3394-4b5e-9adf-021287086938",
            "it": "66ce2150-3394-4b5e-9adf-021287086938",
            "pt": "5f9c01c0-3394-4b5e-9adf-021287086938",
            "ru": "3f9b11a0-3394-4b5e-9adf-021287086938",
            "ar": "a3b511a0-3394-4b5e-9adf-021287086938",
            "hi": "b3b511a0-3394-4b5e-9adf-021287086938",
        }
        return voice_map.get(lang_code, self.voice_id)

    async def speak(self, text: str, language: str | None = None) -> bool:
        """Speak text via VAPI. Returns True on success, False on failure."""
        if not text:
            return True

        self._playing = True
        self._stop_flag = False

        try:
            session = await self._ensure_session()
            voice_id = self._select_voice(language)

            payload = {
                "input": text,
                "voice": voice_id or "en",
                "model": "gpt-4o-mini-tts",
            }

            # VAPI uses OpenAI-compatible TTS endpoint
            url = f"{config.VAPI_BASE_URL}/v1/audio/speech"
            if not self.voice_id:
                url = f"{config.VAPI_BASE_URL}/v1/audio/speech"

            async with session.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    log.error("vapi_tts_error", extra={
                        "status": resp.status,
                        "body": error_body[:200],
                    })
                    return False

                content_type = resp.headers.get("content-type", "")
                audio_data = await resp.read()

            if audio_data and not self._stop_flag:
                if "json" in content_type or audio_data[:1] == b"{":
                    # Response is JSON with base64 audio
                    result = bytes(audio_data).decode("utf-8")
                    import json as json_mod
                    parsed = json_mod.loads(result)
                    audio_b64 = parsed.get("audio")
                    if audio_b64:
                        import base64
                        audio_data = base64.b64decode(audio_b64)

                await self._play_mp3(audio_data)

            return True

        except Exception as e:
            log.error("vapi_tts_error", extra={"error": str(e)})
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

    async def _play_mp3(self, audio_data: bytes) -> None:
        """Decode and play MP3/WAV audio."""
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
            log.warning("soundfile_not_available_for_vapi")
        except Exception as e:
            log.error("audio_playback_failed", extra={"error": str(e)})

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()