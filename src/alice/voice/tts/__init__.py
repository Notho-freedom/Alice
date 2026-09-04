"""Factory for selecting TTS providers based on configuration.

Priority chain (per spec):
1. ElevenLabs — cloud API, highest quality, multilingual
2. VAPI — cloud API, good quality, multilingual
3. Edge (local backend) — local server, good quality via edge-tts
4. Pi TTS (coqui) — local backend, offline neural TTS
5. pyttsx3 — last-resort fallback, low quality but always works
"""

from __future__ import annotations

import logging

from ... import config
from .base import TextToSpeech
from .pyttsx3 import PyTTSX3TTS
from .edge import EdgeTTS
from .elevenlabs import ElevenLabsTTS
from .vapi import VAPITTS
from .pi_tts import PiTTSTTS

log = logging.getLogger("alice.voice.tts.factory")


class TTSCascade:
    """Tries multiple TTS providers in priority order.

    Each provider is attempted in sequence; the first successful
    speak() call wins. Falls through on API errors or failures.
    """

    def __init__(self, providers: list[TextToSpeech]):
        self.providers = providers
        self._active: TextToSpeech | None = None

    @property
    def active(self) -> TextToSpeech | None:
        return self._active

    async def speak(self, text: str, language: str | None = None) -> bool:
        """Try each provider in order until one succeeds."""
        for provider in self.providers:
            self._active = provider
            try:
                result = await provider.speak(text, language)
                if result is True:
                    return True
            except Exception as e:
                log.warning(f"tts_provider_failed", extra={
                    "provider": provider.__class__.__name__,
                    "error": str(e),
                })
        return False

    async def stream_speak(self, chunks, language: str | None = None) -> None:
        """Use the primary provider for streaming."""
        if self.providers:
            self._active = self.providers[0]
            await self.providers[0].stream_speak(chunks, language)

    async def stop(self) -> None:
        """Stop whichever provider is currently speaking."""
        if self._active:
            try:
                await self._active.stop()
            except Exception as e:
                log.warning("tts_stop_failed", extra={"error": str(e)})

    async def pause(self) -> None:
        if self._active:
            await self._active.pause()

    async def resume(self) -> None:
        if self._active:
            await self._active.resume()

    def is_speaking(self) -> bool:
        return self._active is not None and self._active.is_speaking()

    async def close(self):
        """Clean up all providers."""
        for provider in self.providers:
            try:
                if hasattr(provider, "close"):
                    await provider.close()
            except Exception:
                pass


def create_tts() -> TextToSpeech | None:
    """Create a TTS provider or fallback cascade.

    Priority:
    1. ElevenLabs (if ELEVENLABS_API_KEY set)
    2. VAPI (if VAPI_API_KEY set)
    3. Edge (local backend, default)
    4. Pi TTS (local backend)
    5. pyttsx3 (offline fallback)
    """
    provider = config.TTS_PROVIDER.lower()
    providers: list[TextToSpeech] = []

    # Auto mode: build cascade from available providers
    if provider == "auto" or provider == "elevenlabs":
        if config.ELEVENLABS_API_KEY:
            try:
                providers.append(ElevenLabsTTS())
                log.info("elevenlabs_tts_added")
            except Exception as e:
                log.warning("elevenlabs_init_failed", extra={"error": str(e)})

    if provider == "auto" or provider == "vapi":
        if config.VAPI_API_KEY:
            try:
                providers.append(VAPITTS())
                log.info("vapi_tts_added")
            except Exception as e:
                log.warning("vapi_init_failed", extra={"error": str(e)})

    if provider == "auto" or provider == "edge":
        try:
            providers.append(EdgeTTS(
                backend_url=config.TTS_BACKEND_URL,
                voice=config.TTS_VOICE,
            ))
            log.info("edge_tts_added")
        except Exception as e:
            log.warning("edge_tts_init_failed", extra={"error": str(e)})

    if provider == "auto" or provider == "pi_tts":
        try:
            providers.append(PiTTSTTS())
            log.info("pi_tts_added")
        except Exception as e:
            log.warning("pi_tts_init_failed", extra={"error": str(e)})

    # pyttsx3 is always available as last resort
    if provider in ("auto", "pyttsx3", "edge", "pi_tts") or not providers:
        try:
            providers.append(PyTTSX3TTS(rate=config.TTS_RATE, volume=config.TTS_VOLUME))
        except Exception as e:
            log.warning("pyttsx3_init_failed", extra={"error": str(e)})

    if not providers:
        log.error("no_tts_providers_available")
        return None

    if len(providers) == 1:
        return providers[0]

    return TTSCascade(providers)