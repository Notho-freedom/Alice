"""TTS provider using pyttsx3 (local, offline)."""

from __future__ import annotations

import asyncio
import logging
import threading

from pyttsx3 import init as pyttsx3_init

from .base import TextToSpeech

log = logging.getLogger("alice.voice.tts.pyttsx3")


class PyTTSX3TTS(TextToSpeech):
    """Local text-to-speech using pyttsx3."""

    def __init__(self, rate: int = 200, volume: float = 1.0):
        self._engine = pyttsx3_init()
        self._engine.setProperty("rate", rate)
        self._engine.setProperty("volume", volume)
        self._speaking = False
        self._lock = threading.Lock()

    async def speak(self, text: str, language: str | None = None) -> bool:
        if not text:
            return True
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._executor_speak, text)
            return True
        except Exception as e:
            log.error("pyttsx3_speak_error", extra={"error": str(e)})
            return False

    def _executor_speak(self, text: str) -> None:
        self._speaking = True
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        finally:
            self._speaking = False

    async def stream_speak(self, chunks, language: str | None = None) -> None:
        collected = ""
        async for chunk in chunks:
            collected += chunk
            log.debug("tts_streaming", extra={"text": chunk})
        if collected:
            await self.speak(collected)

    async def stop(self) -> None:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._executor_stop, None)
        except Exception as e:
            log.warning("pyttsx3_stop_failed", extra={"error": str(e)})

    def _executor_stop(self, _=None) -> None:
        self._engine.stop()
        self._speaking = False

    async def pause(self) -> None:
        self._engine.pause()  # type: ignore[attr-defined]

    async def resume(self) -> None:
        self._engine.resume()  # type: ignore[attr-defined]

    def is_speaking(self) -> bool:
        return self._speaking