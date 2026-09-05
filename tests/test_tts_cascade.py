"""Tests for TTSCascade and voice picker."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alice.voice.tts import TTSCascade
from alice.voice.voice_picker import resolve_voice_name, VoiceDiscovery


class TestTTSCascade:
    """Test TTSCascade behavior."""

    @pytest.mark.asyncio
    async def test_speak_skips_none_providers(self):
        """Test cascade skips providers that return None."""
        provider1 = MagicMock()
        provider1.speak = AsyncMock(return_value=None)
        provider2 = MagicMock()
        provider2.speak = AsyncMock(return_value=True)

        cascade = TTSCascade([provider1, provider2])
        result = await cascade.speak("hello")

        assert result is True
        provider1.speak.assert_called_once()
        provider2.speak.assert_called_once()

    @pytest.mark.asyncio
    async def test_speak_stops_on_success(self):
        """Test cascade stops on first successful provider."""
        provider1 = MagicMock()
        provider1.speak = AsyncMock(return_value=True)
        provider2 = MagicMock()
        provider2.speak = AsyncMock(return_value=True)

        cascade = TTSCascade([provider1, provider2])
        result = await cascade.speak("hello")

        assert result is True
        provider1.speak.assert_called_once()
        provider2.speak.assert_not_called()

    @pytest.mark.asyncio
    async def test_speak_continues_on_failure(self):
        """Test cascade continues to next provider on failure."""
        provider1 = MagicMock()
        provider1.speak = AsyncMock(side_effect=RuntimeError("failed"))
        provider2 = MagicMock()
        provider2.speak = AsyncMock(return_value=True)

        cascade = TTSCascade([provider1, provider2])
        result = await cascade.speak("hello")

        assert result is True
        provider1.speak.assert_called_once()
        provider2.speak.assert_called_once()

    @pytest.mark.asyncio
    async def test_speak_returns_false_if_all_fail(self):
        """Test cascade returns False if all providers fail."""
        provider1 = MagicMock()
        provider1.speak = AsyncMock(side_effect=RuntimeError("failed"))
        provider2 = MagicMock()
        provider2.speak = AsyncMock(side_effect=RuntimeError("failed"))

        cascade = TTSCascade([provider1, provider2])
        result = await cascade.speak("hello")

        assert result is False

    def test_active_provider_tracking(self):
        """Test active provider is tracked."""
        provider1 = MagicMock()
        provider2 = MagicMock()

        cascade = TTSCascade([provider1, provider2])
        assert cascade.active is None

        cascade._active = provider1
        assert cascade.active is provider1


class TestVoicePicker:
    """Test voice picker utilities."""

    def test_resolve_voice_name_edge_known(self):
        assert resolve_voice_name("edge", "marie") == "fr-FR-DeniseNeural"

    def test_resolve_voice_name_edge_unknown(self):
        assert resolve_voice_name("edge", "unknown-voice") == "unknown-voice"

    def test_resolve_voice_name_elevenlabs_known(self):
        assert resolve_voice_name("elevenlabs", "rachel") == "21m00Tcm4TlvDq8ikWAM"

    def test_resolve_voice_name_vapi_known(self):
        assert resolve_voice_name("vapi", "en") == "9b7d887d-3394-4b5e-9adf-021287086938"

    def test_resolve_voice_name_pi_tts_known(self):
        assert resolve_voice_name("pi_tts", "fr") == "fr/francesco/medium"

    def test_resolve_voice_name_fallback_for_language(self):
        assert resolve_voice_name("edge", None, "fr") == "fr-FR-DeniseNeural"
        assert resolve_voice_name("edge", None, "en") == "en-US-AriaNeural"


class TestVoiceDiscovery:
    """Test voice discovery."""

    @pytest.mark.asyncio
    async def test_discover_edge_voices_empty_on_failure(self):
        discovery = VoiceDiscovery()
        with patch("alice.voice.voice_picker.aiohttp.ClientSession") as MockSession:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session.get = MagicMock(side_effect=RuntimeError("network error"))
            MockSession.return_value = mock_session

            voices = await discovery.discover_edge_voices()
            assert voices == []

    @pytest.mark.asyncio
    async def test_discover_elevenlabs_voices_empty_without_key(self):
        discovery = VoiceDiscovery()
        with patch("alice.voice.voice_picker.config.ELEVENLABS_API_KEY", ""):
            voices = await discovery.discover_elevenlabs_voices()
            assert voices == []
