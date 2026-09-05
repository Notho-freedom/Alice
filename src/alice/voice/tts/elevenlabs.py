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
from ..voice_picker import resolve_voice_name

log = logging.getLogger("alice.voice.tts.elevenlabs")


# French female voices (verified from ElevenLabs API)
# Note: Only "Anna", "Victoria", "Marie Alice" are French but require paid plan
# Free plan voices are primarily English - we use English female voices as fallback
FRENCH_FEMALE_VOICES = {
    "marie": "tMyQcCxfGDdIt7wJ2RQw",  # Requires paid plan
    "victoria": "O31r762Gb3WFygrEOGh0",  # Requires paid plan
    "anna": "PSVUmed8NvS8aUA3d5oO",  # Requires paid plan
}

# All working voices on free plan (tested)
FREE_VOICES = {
    # Female voices (priority order)
    "sarah": "EXAVITQu4vr4xnSDxMaL",  # American female, mature
    "laura": "FGY2WhTYpPnrIDTdsKH5",  # American female, enthusiastic
    "alice": "Xb7hH8MSUJpSbSDYk0k2",  # British female, educator
    "matilda": "XrExE9yKIg1WjnnlVkGX",  # American female, professional
    "jessica": "cgSgspJ2msm6clMCkdW9",  # American female, playful
    "bella": "hpp4J3VqNfWAUOO0d1Us",  # American female, professional
    "lily": "pFZP5JQG7iQjIQuC4Bku",  # British female, actress
    # Male voices (fallback)
    "roger": "CwhRBWXzGAHq8TQ4Fs17",  # American male, casual
    "charlie": "IKne3meq5aSn9XLyUdCD",  # Australian male
    "george": "JBFqnCBsd6RMkjVDRZzb",  # British male, storyteller
    "callum": "N2lVS1w4EtoT3dr4eOWO",  # American male, trickster
    "river": "SAz9YHcvj6GT2YYXdXww",  # American, neutral
    "harry": "SOYHLrjzK2X1ezoPC6cr",  # American male, warrior
    "liam": "TX3LPaxmHKxFdv7VOQHJ",  # American male, energetic
    "will": "bIHbv24MWmeRgasZH58o",  # American male, optimist
    "eric": "cjVigY5qzO86Huf0OWal",  # American male, smooth
    "chris": "iP95p4xoKVk53GoZ742B",  # American male, charming
    "brian": "nPczCjzI2devNBz1zQrb",  # American male, comforting
    "daniel": "onwK4e9ZLuTAKqWW03F9",  # British male, broadcaster
    "adam": "pNInz6obpgDQGcFmaJgB",  # American male, dominant
    "bill": "pqHfZKP75CvOlQylNhV4",  # American male, wise
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
        """Resolve the voice ID based on language, with fallback chain."""
        return resolve_voice_name("elevenlabs", self.voice_name, language or self.language)

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
        """Speak text via ElevenLabs. Returns True on success, False on failure.
        
        Tests multiple voices in priority order if the first one fails.
        """
        if not text:
            return True

        self._playing = True
        self._stop_flag = False

        try:
            # Get prioritized list of voices to try
            voices_to_try = self._get_voice_priority_list(language)
            
            # Try each API key with each voice
            for voice_id in voices_to_try:
                # Try all API keys for this voice
                max_attempts = len(self.api_keys)
                for _ in range(max_attempts):
                    api_key = self._get_current_key()
                    audio_data, status = await self._call_elevenlabs(text, voice_id, api_key)
                    
                    if status == 200:
                        if audio_data and not self._stop_flag:
                            await self._play_mp3(audio_data)
                            return True
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
                        # Other error (e.g., 402 paid plan), try next voice
                        break
            
            return False

        except Exception as e:
            log.error("elevenlabs_tts_error", extra={"error": str(e)})
            return False
        finally:
            self._playing = False

    def _get_voice_priority_list(self, language: str | None = None) -> list[str]:
        """Get prioritized list of voice IDs to try based on language."""
        lang = language or self.language
        lang_code = lang.split("-")[0].lower() if lang else "fr"
        
        voices = []
        
        if lang_code == "fr":
            # French: try French voices first, then English female, then English male
            for name in ["marie", "victoria", "anna"]:
                if name in FRENCH_FEMALE_VOICES:
                    voices.append(FRENCH_FEMALE_VOICES[name])
            for name in ["sarah", "laura", "alice", "matilda", "jessica", "bella", "lily"]:
                if name in FREE_VOICES:
                    voices.append(FREE_VOICES[name])
            for name in ["roger", "charlie", "george", "callum", "river"]:
                if name in FREE_VOICES:
                    voices.append(FREE_VOICES[name])
        elif lang_code == "en":
            # English: female first, then male
            for name in ["sarah", "laura", "alice", "matilda", "jessica", "bella", "lily"]:
                if name in FREE_VOICES:
                    voices.append(FREE_VOICES[name])
            for name in ["roger", "charlie", "george", "callum", "river"]:
                if name in FREE_VOICES:
                    voices.append(FREE_VOICES[name])
        else:
            # Other languages: try all free voices in priority order
            for name in ["sarah", "laura", "alice", "matilda", "jessica", "bella", "lily"]:
                if name in FREE_VOICES:
                    voices.append(FREE_VOICES[name])
            for name in ["roger", "charlie", "george", "callum", "river"]:
                if name in FREE_VOICES:
                    voices.append(FREE_VOICES[name])
        
        # Add custom voice if specified
        if self.voice_name and self.voice_name.lower() in FREE_VOICES:
            voices.insert(0, FREE_VOICES[self.voice_name.lower()])
        
        return voices if voices else ["EXAVITQu4vr4xnSDxMaL"]  # Default to Sarah

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
            import soundfile as sf
            import io

            audio_array, sample_rate = sf.read(io.BytesIO(audio_data))

            if audio_array.dtype in (np.float32, np.float64):
                audio_int16 = (audio_array * 32767).astype(np.int16)
            else:
                audio_int16 = audio_array

            await self._play_int16(audio_int16, sample_rate, lambda: self._stop_flag)
        except ImportError:
            log.warning("soundfile_not_available_for_elevenlabs")
        except Exception as e:
            log.error("audio_playback_failed", extra={"error": str(e)})

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()