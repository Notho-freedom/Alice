"""Factory for selecting TTS providers based on configuration."""

from __future__ import annotations

import logging

from ... import config
from .base import TextToSpeech
from .pyttsx3 import PyTTSX3TTS
from .edge import EdgeTTS

log = logging.getLogger("alice.voice.tts.factory")


def create_tts() -> TextToSpeech | None:
    """Create the configured TTS provider."""
    provider = config.TTS_PROVIDER.lower()

    if provider == "edge":
        try:
            return EdgeTTS(
                backend_url=config.TTS_BACKEND_URL,
                voice=config.TTS_VOICE,
            )
        except Exception as e:
            log.warning("edge_tts_init_failed", extra={"error": str(e)})
            # Fall back to pyttsx3

    if provider == "pyttsx3" or provider == "edge":
        try:
            return PyTTSX3TTS(
                rate=config.TTS_RATE,
                volume=config.TTS_VOLUME,
            )
        except Exception as e:
            log.warning("pyttsx3_init_failed", extra={"error": str(e)})
            return None

    log.warning("unknown_tts_provider", extra={"provider": provider})
    return None