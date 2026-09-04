"""TTS provider using ElevenLabs API.

High-quality neural voices with multilingual support.
Supports two API keys for credit fallback.
Uses ELEVENLABS_API_KEY1 and ELEVENLABS_API_KEY2 from environment.
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


# French female voices (verified from ElevenLabs API)
FRENCH_FEMALE_VOICES = {
    "victoria": "O31r762Gb3WFygrEOGh0",  # Parisian accent, Content Creator
    "marie": "tMyQcCxfGDdIt7wJ2RQw",    # Soft, Calm and Captivating
    "anna": "PSVUmed8NvS8aUA3d5oO",     # Standard accent, audiobook
}


class ElevenLabsTTS(TextToSpeech):
    """Text-to-speech via ElevenLabs cloud API.

    Provides high-quality neural voices with automatic language detection.
    Supports two API keys for credit fallback.
    """

    def __init__(
        self,
        api_keys: list[str] | None = None,
        voice_name: str | None = None,
        language: str | None = None,
    ):
        # Support two API keys for fallback
        if api_keys:
            self.api_keys = api_keys
        else:
            self.api_keys = [config.ELEVENLABS_API_KEY1, config.ELEVENLABS_API_KEY2]
            self.api_keys = [k for k in self.api_keys if k]  # Remove empty keys
        
        if not self.api_keys:
            raise ValueError("No ElevenLabs API keys provided")
        
        self.voice_name = voice_name or config.ELEVENLABS_VOICE or "marie"
        self.language = language or config.TTS_LANGUAGE_CODE or "fr"
        self._playing = False
        self._stop_flag = False
        self._session: aiohttp.ClientSession | None = None
        self._current_key_index = 0

    def _get_current_key(self) -> str:
        """Get the current API key."""
        return self.api_keys[self._current_key_index % len(self.api_keys)]

    def _rotate_key(self) -> bool:
        """Try the next API key. Returns True if there's another key to try."""
        self._current_key_index += 1
        return self._current_key_index < len(self.api_keys)

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

    def _resolve_voice_id(self, language: str | None = None) -> str:
        """Resolve the voice ID based on language."""
        lang = language or self.language
        lang_code = lang.split("-")[0].lower() if lang else "fr"
        
        # If voice_name is a known French female voice, use it
        if self.voice_name.lower() in FRENCH_FEMALE_VOICES:
            return FRENCH_FEMALE_VOICES[self.voice_name.lower()]
        
        # Default French female voice
        if lang_code == "fr":
            return FRENCH_FEMALE_VOICES.get("marie", "tMyQcCxfGDdIt7wJ2RQw")
        
        # English fallback
        if lang_code == "en":
            return "21m00Tcm4TlvDq8pnDb4e"  # Rachel
        
        # Default to Marie (multilingual)
        return "tMyQcCxfGDdIt7wJ2RQw"

    async def _call_elevenlabs(self, text: str, voice_id: str, api_key: str) -> tuple[bytes, int]:
        """Call ElevenLabs API with a specific key. Returns (audio_data, status_code)."""
        session = await self._ensure_session()
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
        
        payload = {
            "text": text,
            "model_id": config.ELEVENLABS_MODEL,
            "voice_settings": {
                "stability": 0.75,
                "similarity_boost": 0.75,
            },
        }
        
        async with session.post(
            url,
            json=payload,
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status == 200:
                return await resp.read(), resp.status
            
            error_body = await resp.text()
            log.error("elevenlabs_api_error", extra={
                "status": resp.status,
                "body": error_body[:300],
                "key_index": self._current_key_index,
            })
            return b"", resp.status

    async def speak(self, text: str, language: str | None = None) -> bool:
        """Speak text via ElevenLabs. Returns True on success, False on failure."""
        if not text:
            return True

        self._playing = True
        self._stop_flag = False

        try:
            voice_id = self._resolve_voice_id(language)
            audio_data = b""
            status = 0
            
            # Try all API keys
            max_attempts = len(self.api_keys)
            for _ in range(max_attempts):
                api_key = self._get_current_key()
                audio_data, status = await self._call_elevenlabs(text, voice_id, api_key)
                
                if status == 200:
                    break
                elif status in (401, 403):
                    # Invalid key, try next
                    log.warning("elevenlabs_key_invalid", extra={"key_index": self._current_key_index})
                    if not self._rotate_key():
                        break
                elif status == 429:
                    # Rate limit, try next key
                    log.warning("elevenlabs_rate_limited", extra={"key_index": self._current_key_index})
                    if not self._rotate_key():
                        break
                else:
                    # Other error, stop trying
                    break

            if status == 200 and audio_data and not self._stop_flag:
                await self._play_mp3(audio_data)
                return True
            
            return False

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

            while not self._stop_flag:
                await asyncio.sleep(0.05)
                stream = sd.get_stream()
                if stream is not None and not stream.active:
                    break

            if self._stop_flag:
                sd.stop()
        except ImportError:
            log.warning("soundfile_not_available_for_elevenlabs")
        except Exception as e:
            log.error("audio_playback_failed", extra={"error": str(e)})

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()