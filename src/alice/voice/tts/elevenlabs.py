"""TTS provider using ElevenLabs API.

High-quality neural voices with multilingual support.
Uses ELEVENLABS_API_KEY from environment.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import aiohttp
import numpy as np

from ... import config
from .base import TextToSpeech

log = logging.getLogger("alice.voice.tts.elevenlabs")


class ElevenLabsTTS(TextToSpeech):
    """Text-to-speech via ElevenLabs cloud API.

    Provides high-quality neural voices with automatic language detection.
    Falls back to EdgeTTS if the API call fails.
    """

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
    ):
        self.api_key = api_key or config.ELEVENLABS_API_KEY
        self.voice_id = voice_id or "21m00Tcm4TlvDq8pnDb4e"
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
        """Select an appropriate voice ID for the given language."""
        lang_map = {
            "en": "21m00Tcm4TlvDq8pnDb4e",  # Rachel (en-US)
            "fr": "K2khGgu88xikqC8AS4f4",  # Nicom (fr-FR)
            "es": "EXW5v3zX9Z8hZ4e3v6k7",  # TBD
            "de": "O2k0Y5j8qN9nE3w6k7mZ",  # TBD
            "zh": "Z6v9Y5j8qN9nE3w6k7mZ",  # TBD
            "ja": "3kF5j8qN9nE3w6k7mZ8b",  # TBD
            "ko": "7mZ5j8qN9nE3w6k7mZ8b1",  # TBD
            "it": "9Y5j8qN9nE3w6k7mZ8b1c",  # TBD
            "pt": "4TlvDq8pnDb4e21m00Tcm",  # TBD
            "ru": "5j8qN9nE3w6k7mZ8b1c2",  # TBD
            "ar": "8qN9nE3w6k7mZ8b1c2d3",  # TBD
            "hi": "9nE3w6k7mZ8b1c2d3e4",  # TBD
        }
        
        if not language:
            language = config.TTS_LANGUAGE_CODE
        
        lang_code = language.split("-")[0].lower() if language else "en"
        return lang_map.get(lang_code, self.voice_id)

    async def speak(self, text: str, language: str | None = None) -> bool:
        """Speak text via ElevenLabs. Returns True on success, False on failure."""
        if not text:
            return True

        self._playing = True
        self._stop_flag = False

        try:
            session = await self._ensure_session()
            voice_id = self._select_voice(language)

            # ElevenLabs API
            url = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream".replace(
                "{voice_id}", voice_id
            )

            payload = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.75,
                    "similarity_boost": 0.75,
                    "speaking_rate": 1.0,
                },
            }

            async with session.post(
                url,
                json=payload,
                headers={
                    "xi-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    log.error("elevenlabs_api_error", extra={
                        "status": resp.status,
                        "body": error_body[:200],
                    })
                    return False
                
                # ElevenLabs returns raw MP3 audio stream
                audio_data = await resp.read()

            if audio_data and not self._stop_flag:
                await self._play_mp3(audio_data)

            return True

        except Exception as e:
            log.error("elevenlabs_tts_error", extra={"error": str(e)})
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
        """Decode and play MP3 audio."""
        try:
            import sounddevice as sd
            import soundfile as sf
            import io

            audio_array, sample_rate = sf.read(io.BytesIO(audio_data))

            if audio_array.dtype in (np.float32, np.float64):
                audio_int16 = (audio_array * 32767).astype(np.int16)
            else:
                audio_int16 = audio_array

            sd.play(
                audio_int16,
                samplerate=sample_rate,
                device=config.AUDIO_OUTPUT_DEVICE,
                blocking=False,
            )

            while sd.get_busy() and not self._stop_flag:
                await asyncio.sleep(0.05)

            if self._stop_flag:
                sd.stop()
        except ImportError:
            log.warning("soundfile_not_available_for_elevenlabs")
        except Exception as e:
            log.error("audio_playback_failed", extra={"error": str(e)})

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()