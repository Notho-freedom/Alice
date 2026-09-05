"""Tests for TerminalAssistant with mocked KiloBridge."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alice.terminal.interface import TerminalAssistant
from alice.core.state_machine import State, Event


class TestTerminalAssistantIntegration:
    """Integration tests for TerminalAssistant with mocked KiloBridge."""

    @pytest.mark.asyncio
    async def test_initialize_sets_session_id(self):
        """Test initialize creates KiloBridge and sets session ID."""
        with patch("alice.terminal.interface.KiloBridge") as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.start = AsyncMock()
            mock_bridge.session_id = "terminal-session-123"
            MockBridge.return_value = mock_bridge

            assistant = TerminalAssistant()
            await assistant.initialize()

            assert assistant._bridge is mock_bridge
            assert assistant._session_id == "terminal-session-123"
            mock_bridge.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_streams_response(self):
        """Test send_message processes streamed response."""
        with patch("alice.terminal.interface.KiloBridge") as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.start = AsyncMock()
            mock_bridge.session_id = "terminal-session-123"

            async def mock_stream(text):
                from alice.kilo.stream import ResponseChunk, ChunkType
                yield ResponseChunk(ChunkType.TEXT, "Hello")
                yield ResponseChunk(ChunkType.TEXT, " from")
                yield ResponseChunk(ChunkType.TEXT, " terminal")
                yield ResponseChunk(ChunkType.COMPLETION, "")

            mock_bridge.send_and_stream = mock_stream
            MockBridge.return_value = mock_bridge

            assistant = TerminalAssistant()
            await assistant.initialize()
            assistant._bridge = mock_bridge

            await assistant.send_message("test prompt")

            assert assistant.state_machine.state == State.IDLE

    @pytest.mark.asyncio
    async def test_cleanup_closes_bridge(self):
        """Test cleanup closes KiloBridge."""
        with patch("alice.terminal.interface.KiloBridge") as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.start = AsyncMock()
            mock_bridge.close = AsyncMock()
            MockBridge.return_value = mock_bridge

            assistant = TerminalAssistant()
            await assistant.initialize()
            await assistant.cleanup()

            mock_bridge.close.assert_called_once()
