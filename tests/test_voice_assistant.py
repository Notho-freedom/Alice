"""Tests for VoiceAssistant with mocked KiloBridge."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alice.voice.assistant import VoiceAssistant, VoiceConfig
from alice.core.state_machine import State, Event


class TestVoiceAssistantIntegration:
    """Integration tests for VoiceAssistant with mocked KiloBridge."""

    @pytest.mark.asyncio
    async def test_initialize_sets_bridge(self):
        """Test initialize creates KiloBridge and sets session ID."""
        with patch("alice.voice.assistant.KiloBridge") as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.start = AsyncMock()
            mock_bridge.session_id = "test-session-123"
            MockBridge.return_value = mock_bridge

            with patch("alice.voice.assistant.create_stt", return_value=None):
                with patch("alice.voice.assistant.create_tts", return_value=None):
                    assistant = VoiceAssistant()
                    await assistant.initialize()

            assert assistant._bridge is mock_bridge
            assert assistant.lifecycle.context.session_id == "test-session-123"
            mock_bridge.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_and_speak_streams_chunks(self):
        """Test _send_and_speak processes streamed chunks."""
        with patch("alice.voice.assistant.KiloBridge") as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.start = AsyncMock()
            mock_bridge.session_id = "test-session-123"

            async def mock_stream(text):
                from alice.kilo.stream import ResponseChunk, ChunkType
                yield ResponseChunk(ChunkType.TEXT, "Hello")
                yield ResponseChunk(ChunkType.TEXT, " world")
                yield ResponseChunk(ChunkType.COMPLETION, "")

            mock_bridge.send_and_stream = mock_stream
            MockBridge.return_value = mock_bridge

            with patch("alice.voice.assistant.create_stt", return_value=None):
                with patch("alice.voice.assistant.create_tts") as MockTTS:
                    mock_tts = MagicMock()
                    mock_tts.speak = AsyncMock()
                    mock_tts.stop = AsyncMock()
                    mock_tts.is_speaking = MagicMock(return_value=False)
                    MockTTS.return_value = mock_tts

                    assistant = VoiceAssistant()
                    await assistant.initialize()
                    assistant._bridge = mock_bridge
                    assistant._tts = mock_tts
                    assistant._interrupter = None

                    await assistant._send_and_speak("test prompt")

            mock_tts.speak.assert_called()

    @pytest.mark.asyncio
    async def test_handle_interruption_stops_tts(self):
        """Test _handle_interruption stops TTS and captures pre-roll."""
        with patch("alice.voice.assistant.KiloBridge") as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.start = AsyncMock()
            mock_bridge.interrupt = AsyncMock()
            MockBridge.return_value = mock_bridge

            with patch("alice.voice.assistant.create_stt", return_value=None):
                with patch("alice.voice.assistant.create_tts") as MockTTS:
                    mock_tts = MagicMock()
                    mock_tts.speak = AsyncMock()
                    mock_tts.stop = AsyncMock()
                    mock_tts._playing = False
                    mock_tts._engine = MagicMock()
                    MockTTS.return_value = mock_tts

                    assistant = VoiceAssistant()
                    await assistant.initialize()
                    assistant._tts = mock_tts
                    assistant._bridge = mock_bridge

                    pre_roll = b"pre-roll-audio-data"
                    assistant._handle_interruption(pre_roll=pre_roll)

            assert bytes(assistant._transcription_buffer) == pre_roll
            mock_tts._engine.stop.assert_called_once()
            mock_bridge.interrupt.assert_called_once()

    @pytest.mark.asyncio
    async def test_audio_chunk_accumulates_during_listening(self):
        """Test audio chunks are accumulated during LISTENING state."""
        with patch("alice.voice.assistant.KiloBridge") as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.start = AsyncMock()
            MockBridge.return_value = mock_bridge

            with patch("alice.voice.assistant.create_stt", return_value=None):
                with patch("alice.voice.assistant.create_tts", return_value=None):
                    assistant = VoiceAssistant()
                    await assistant.initialize()

            assistant.state_machine.fire(Event.START_LISTENING)
            assert assistant.state_machine.state == State.LISTENING

            assistant._on_audio_chunk(b"audio chunk 1")
            assistant._on_audio_chunk(b"audio chunk 2")

            assert bytes(assistant._transcription_buffer) == b"audio chunk 1audio chunk 2"
