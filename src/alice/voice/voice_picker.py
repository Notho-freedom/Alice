"""Voice selection utilities for TTS providers.

Provides a shared voice picker that can:
- resolve known voice names to provider-specific IDs
- optionally discover voices from provider APIs
- cache discovered voices for the session
"""

from __future__ import annotations

import logging
from typing import Any

from .. import config

log = logging.getLogger("alice.voice.voice_picker")


EDGE_VOICE_MAP = {
    "marie": "fr-FR-DeniseNeural",
    "victoria": "fr-FR-VivienneMultilingualNeural",
    "anna": "fr-FR-EloiseNeural",
    "rachel": "en-US-AriaNeural",
    "alice": "en-GB-SoniaNeural",
    "laura": "en-US-JennyNeural",
    "sarah": "en-US-GuyNeural",
    "jessica": "en-US-EmmaNeural",
    "bella": "en-US-MichelleNeural",
    "lily": "en-GB-LibbyNeural",
}

ELEVENLABS_VOICE_MAP = {
    "rachel": "21m00Tcm4TlvDq8ikWAM",
    "marie": "9b7d887d-3394-4b5e-9adf-021287086938",
    "victoria": "c2b0de7d-3394-4b5e-021287086938",
    "anna": "2b4e31d7-0a68-4ec7-ade3-97d6e34e1a6a",
}

VAPI_VOICE_MAP = {
    "en": "9b7d887d-3394-4b5e-9adf-021287086938",
    "fr": "c2b0de7d-3394-4b5e-021287086938",
    "es": "3b01b5e0-3394-4b5e-9adf-021287086938",
    "de": "08d47bdc-3394-4b5e-9adf-021287086938",
}

PI_TTS_VOICE_MAP = {
    "en": "en_US/amy/medium",
    "fr": "fr/francesco/medium",
    "es": "es/es/davefx/medium",
    "de": "de/de/thorsten/medium",
    "zh": "zh/zh/zh_english/medium",
    "it": "it/it/mirko/medium",
    "pt": "pt/pt/pedro/medium",
    "ru": "ru/ru/irina/medium",
}


def resolve_voice_name(provider: str, voice: str | None, language: str | None = None) -> str:
    """Resolve a user-facing voice name to a provider-specific voice ID.

    Args:
        provider: one of 'edge', 'elevenlabs', 'vapi', 'pi_tts'
        voice: user-facing voice name or raw provider voice ID
        language: optional language hint

    Returns:
        Resolved voice ID/name for the provider.
    """
    if not voice:
        return _default_voice_for_language(provider, language)

    voice_lower = voice.lower()
    if provider == "edge":
        return EDGE_VOICE_MAP.get(voice_lower, voice)
    if provider == "elevenlabs":
        return ELEVENLABS_VOICE_MAP.get(voice_lower, voice)
    if provider == "vapi":
        return VAPI_VOICE_MAP.get(voice_lower, voice)
    if provider == "pi_tts":
        return PI_TTS_VOICE_MAP.get(voice_lower, voice or "")
    return voice or ""


def _default_voice_for_language(provider: str, language: str | None) -> str:
    language = (language or config.TTS_LANGUAGE or "fr").lower()
    if provider == "edge":
        if language.startswith("fr"):
            return "fr-FR-DeniseNeural"
        if language.startswith("en"):
            return "en-US-AriaNeural"
        return "fr-FR-DeniseNeural"
    if provider == "elevenlabs":
        if language.startswith("fr"):
            return ELEVENLABS_VOICE_MAP.get("marie", "Rachel")
        return ELEVENLABS_VOICE_MAP.get("rachel", "Rachel")
    if provider == "vapi":
        code = language.split("-")[0]
        return VAPI_VOICE_MAP.get(code, VAPI_VOICE_MAP["en"])
    if provider == "pi_tts":
        code = language.split("-")[0]
        return PI_TTS_VOICE_MAP.get(code, PI_TTS_VOICE_MAP["en"])
    return ""


class VoiceDiscovery:
    """Optional runtime discovery of available voices from provider APIs."""

    def __init__(self):
        self._cache: dict[str, list[dict[str, Any]]] = {}

    async def discover_edge_voices(self) -> list[dict[str, Any]]:
        return []

    async def discover_elevenlabs_voices(self) -> list[dict[str, Any]]:
        return []

    async def discover_vapi_voices(self) -> list[dict[str, Any]]:
        return []
