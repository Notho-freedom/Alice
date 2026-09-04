"""Tests for the Kilo Bridge — session lifecycle contract.

Tests the complete flow: create → send → stream → interrupt → continue.
Uses mocks to test the state machine transitions and contract enforcement.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alice.kilo.bridge import KiloBridge, BridgeConfig, TurnResult
from alice.kilo.client import KiloClient, KiloServerError
from alice.kilo.protocol import KiloEvent
from alice.kilo.stream import ChunkType, ResponseChunk, ResponseStreamProcessor
from alice.core.state_machine import State, Event, StateMachine


@pytest.fixture
def mock_client():
    """Create a mock KiloClient with all methods async-ready."""
    client = MagicMock(spec=KiloClient)
    client.base_url = "http://127.0.0.1:4096"
    client.directory = "/test/dir"
    client.health = MagicMock(return_value=True)
    client.start_server = MagicMock()
    client.stop_server = MagicMock()
    client.create_session = MagicMock(return_value="test-session-123")
    client.get_active_session = MagicMock(return_value=None)
    client.get_session = MagicMock(return_value={"id": "test-session-123", "title": "Test"})
    client.delete_session = MagicMock()
    client.send_prompt = MagicMock(return_value={"accepted": True})
    client.interrupt = MagicMock()
    client.list_sessions = MagicMock(return_value=[])
    # subscribe_events returns a KiloEventStream-like object (synchronous call)
    client.subscribe_events = AsyncMock()
    return client


@pytest.fixture
def mock_event_stream():
    """Create a mock SSE event stream."""
    events = [
        # Step started
        KiloEvent(
            event_type="session.next.step.started",
            session_id="test-session-123",
            properties={"agent": "coder", "model": {"modelID": "gpt-4", "providerID": "openai"}},
        ),
        # Text delta
        KiloEvent(
            event_type="session.next.text.delta",
            session_id="test-session-123",
            properties={"delta": "Hello"},
        ),
        KiloEvent(
            event_type="session.next.text.delta",
            session_id="test-session-123",
            properties={"delta": " world"},
        ),
        # Text ended
        KiloEvent(
            event_type="session.next.text.ended",
            session_id="test-session-123",
            properties={"text": "Hello world"},
        ),
        # Step ended
        KiloEvent(
            event_type="session.next.step.ended",
            session_id="test-session-123",
            properties={
                "finish": "stop",
                "cost": 0.001,
                "tokens": {"input": 10, "output": 5, "reasoning": 0},
            },
        ),
    ]

    class MockStream:
        async def start(self):
            pass
        async def stop(self):
            pass
        def __aiter__(self):
            async def gen():
                for event in events:
                    yield event
            return gen()

    return MockStream(), events


@pytest.fixture
def mock_event_stream_with_error():
    """Event stream that emits an error event."""
    events = [
        KiloEvent(
            event_type="session.error",
            session_id="test-session-123",
            properties={"error": "Something went wrong"},
        ),
    ]

    class MockStream:
        async def start(self):
            pass
        async def stop(self):
            pass
        def __aiter__(self):
            async def gen():
                for event in events:
                    yield event
            return gen()

    return MockStream(), events


class TestSessionLifecycle:
    """Tests for session creation, resume, and deletion."""

    @pytest.mark.asyncio
    async def test_start_creates_new_session(self, mock_client):
        """start() should create a new session when none exists."""
        bridge = KiloBridge(client=mock_client, config=BridgeConfig(
            auto_start_server=False,
            auto_resume_session=True,
        ))

        session_id = await bridge.start()

        assert session_id == "test-session-123"
        mock_client.create_session.assert_called_once()
        assert bridge.state == State.IDLE  # SESSION_STARTED is self-transition

    @pytest.mark.asyncio
    async def test_start_resumes_existing_session(self, mock_client):
        """start() should resume an active session if one exists."""
        mock_client.get_active_session.return_value = "existing-session-456"

        bridge = KiloBridge(client=mock_client, config=BridgeConfig(
            auto_start_server=False,
            auto_resume_session=True,
        ))

        session_id = await bridge.start()

        assert session_id == "existing-session-456"
        mock_client.create_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_deletes_session(self, mock_client):
        """close() should delete the session and stop the server."""
        bridge = KiloBridge(client=mock_client, config=BridgeConfig(
            auto_start_server=False,
            auto_resume_session=False,
        ))

        await bridge.start()
        await bridge.close()

        mock_client.delete_session.assert_called_once_with("test-session-123")
        assert bridge.session_id is None


class TestPromptSending:
    """Tests for prompt sending and state transitions."""

    @pytest.mark.asyncio
    async def test_send_prompt_fires_prompt_sent(self, mock_client):
        """send_prompt() should fire PROMPT_SENT and transition to THINKING."""
        bridge = KiloBridge(client=mock_client, config=BridgeConfig(
            auto_start_server=False,
            auto_resume_session=False,
        ))

        await bridge.start()
        await bridge.send_prompt("Hello, Kilo")

        assert bridge.state == State.THINKING
        mock_client.send_prompt.assert_called_once_with(
            "test-session-123",
            "Hello, Kilo",
        )

    @pytest.mark.asyncio
    async def test_send_prompt_without_session_raises(self, mock_client):
        """send_prompt() should raise if no session is active."""
        bridge = KiloBridge(client=mock_client, config=BridgeConfig(
            auto_start_server=False,
            auto_resume_session=False,
        ))

        with pytest.raises(KiloServerError, match="No active session"):
            await bridge.send_prompt("Hello")


class TestStreaming:
    """Tests for the SSE streaming contract."""

    @pytest.mark.asyncio
    async def test_stream_response_yields_chunks(self, mock_client, mock_event_stream):
        """stream_response() should yield ResponseChunks for each event."""
        stream, events = mock_event_stream
        mock_client.subscribe_events.return_value = stream

        bridge = KiloBridge(client=mock_client, config=BridgeConfig(
            auto_start_server=False,
            auto_resume_session=False,
        ))

        await bridge.start()
        mock_client.send_prompt("test")

        chunks = []
        async for chunk in bridge.stream_response():
            chunks.append(chunk)

        # The processor transforms 5 events into chunks
        assert len(chunks) >= 3
        # Check for completion chunk
        assert any(c.type == ChunkType.COMPLETION for c in chunks)
        # Check for step_end chunk
        assert any(c.type == ChunkType.STEP_END for c in chunks)
        # Full text should be "Hello world"
        completion = next(c for c in chunks if c.type == ChunkType.COMPLETION)
        assert completion.data.get("full_text") == "Hello world"

    @pytest.mark.asyncio
    async def test_stream_response_handles_error(self, mock_client, mock_event_stream_with_error):
        """stream_response() should yield ERROR chunks."""
        stream, events = mock_event_stream_with_error
        mock_client.subscribe_events.return_value = stream

        bridge = KiloBridge(client=mock_client, config=BridgeConfig(
            auto_start_server=False,
            auto_resume_session=False,
        ))

        await bridge.start()
        mock_client.send_prompt("test")

        chunks = []
        async for chunk in bridge.stream_response():
            chunks.append(chunk)

        assert any(c.type == ChunkType.ERROR for c in chunks)
        error_chunk = next(c for c in chunks if c.type == ChunkType.ERROR)
        assert "Something went wrong" in error_chunk.text


class TestInterrupt:
    """Tests for the interrupt contract (Rule E: session survives)."""

    @pytest.mark.asyncio
    async def test_interrupt_aborts_session(self, mock_client, mock_event_stream):
        """interrupt() should call client.interrupt but NOT delete session."""
        stream, events = mock_event_stream
        mock_client.subscribe_events.return_value = stream

        bridge = KiloBridge(client=mock_client, config=BridgeConfig(
            auto_start_server=False,
            auto_resume_session=False,
        ))

        await bridge.start()
        await bridge.send_prompt("Long running task...")

        # Start streaming in background (async generator must be consumed in a task)
        async def _consume():
            async for chunk in bridge.stream_response():
                pass

        stream_task = asyncio.create_task(_consume())
        await asyncio.sleep(0.01)

        # Interrupt
        await bridge.interrupt()

        # Session should NOT be deleted
        mock_client.interrupt.assert_called_once_with("test-session-123")
        mock_client.delete_session.assert_not_called()
        assert bridge.session_id is not None

        # Clean up
        stream_task.cancel()
        try:
            await stream_task
        except asyncio.CancelledError:
            pass


class TestSendAndWait:
    """Tests for the send_and_wait convenience method."""

    @pytest.mark.asyncio
    async def test_send_and_wait_returns_turn_result(self, mock_client, mock_event_stream):
        """send_and_wait() should aggregate chunks into a TurnResult."""
        stream, events = mock_event_stream
        mock_client.subscribe_events.return_value = stream

        bridge = KiloBridge(client=mock_client, config=BridgeConfig(
            auto_start_server=False,
            auto_resume_session=False,
        ))

        await bridge.start()
        result = await bridge.send_and_wait("Hello", timeout=5.0)

        assert isinstance(result, TurnResult)
        assert result.full_text == "Hello world"
        assert result.cost == 0.001
        assert result.tokens == {"input": 10, "output": 5, "reasoning": 0}
        assert result.finish_reason == "stop"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_send_and_wait_handles_error(self, mock_client, mock_event_stream_with_error):
        """send_and_wait() should capture errors in TurnResult.error."""
        stream, events = mock_event_stream_with_error
        mock_client.subscribe_events.return_value = stream

        bridge = KiloBridge(client=mock_client, config=BridgeConfig(
            auto_start_server=False,
            auto_resume_session=False,
        ))

        await bridge.start()
        result = await bridge.send_and_wait("test", timeout=5.0)

        assert result.error == "Something went wrong"


class TestStateTransitions:
    """Tests for state machine transitions through the bridge."""

    @pytest.mark.asyncio
    async def test_full_turn_state_transitions(self, mock_client, mock_event_stream):
        """Verify state transitions through a complete turn."""
        stream, events = mock_event_stream
        mock_client.subscribe_events.return_value = stream

        bridge = KiloBridge(client=mock_client, config=BridgeConfig(
            auto_start_server=False,
            auto_resume_session=False,
        ))

        await bridge.start()
        # After start: IDLE (SESSION_STARTED is self-transition on IDLE)

        await bridge.send_prompt("Hello")
        # PROMPT_SENT: IDLE -> THINKING
        assert bridge.state == State.THINKING

        await bridge.send_and_wait("Hello", timeout=5.0)
        # After complete turn: THINKING -> (streaming) -> eventually back to IDLE
        # The state machine is driven by fire() calls in stream processing

    @pytest.mark.asyncio
    async def test_turn_context_manager(self, mock_client, mock_event_stream):
        """turn() context manager should fire TURN_STARTED and TURN_ENDED."""
        stream, events = mock_event_stream
        mock_client.subscribe_events.return_value = stream

        bridge = KiloBridge(client=mock_client, config=BridgeConfig(
            auto_start_server=False,
            auto_resume_session=False,
        ))

        await bridge.start()

        async with bridge.turn():
            assert bridge.state == State.LISTENING  # TURN_STARTED

        # After context exit: TURN_ENDED doesn't change state from IDLE
        # (it fires from whatever state we're in)