"""Tests for wake-word integration."""

from __future__ import annotations

import asyncio
import pytest

from alice.voice.assistant import VoiceAssistant, VoiceConfig
from alice.voice.wake import KeywordWakeWord, PorcupineWakeWord, PassthroughWakeWord


class TestWakeWordHealthCheck:
    def test_health_check_returns_status(self):
        assistant = VoiceAssistant()
        status = assistant.health_check()
        assert "wake_word" in status
        assert "wake_word_available" in status

    def test_wake_word_fallback_when_porcupine_unavailable(self):
        assistant = VoiceAssistant()
        asyncio.run(assistant.initialize())
        assert assistant._wake is not None
        assert isinstance(assistant._wake, (KeywordWakeWord, PorcupineWakeWord, PassthroughWakeWord))


class TestPorcupineWakeWord:
    def test_porcupine_unavailable_fallback(self):
        vad = None
        wake = PorcupineWakeWord(vad, keyword="hey assistant")
        assert wake._available is False

    def test_passthrough_always_false(self):
        wake = PassthroughWakeWord()
        assert wake.process(b"") is False
        assert wake.process(b"some audio") is False
