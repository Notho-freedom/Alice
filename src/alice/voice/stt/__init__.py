"""Factory for selecting STT and TTS providers based on configuration."""

from __future__ import annotations

import logging

from ... import config
from .base import SpeechRecognizer
from .openai import OpenAISTT
from .deepgram import DeepgramSTT

log = logging.getLogger("alice.voice.stt.factory")


def create_stt() -> SpeechRecognizer | None:
    """Create the configured STT provider."""
    provider = config.STT_PROVIDER.lower()

    if provider == "deepgram":
        if not config.DEEPGRAM_API_KEY:
            log.warning("no_deepgram_key")
            return None
        return DeepgramSTT(api_key=config.DEEPGRAM_API_KEY)

    elif provider == "openai":
        if not config.OPENAI_API_KEY:
            log.warning("no_openai_key")
            return None
        return OpenAISTT(api_key=config.OPENAI_API_KEY)

    else:
        log.warning("unknown_stt_provider", extra={"provider": provider})
        return None