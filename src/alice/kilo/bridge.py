"""Kilo Bridge — complete session lifecycle management.

Implements the Alice → Kilo integration contract:
  create_session → send_prompt → stream_response → interrupt → continue

This is the thin, tested layer between Alice's voice/terminal interfaces
and the Kilo Code brain. Per architecture spec, Alice delegates all
reasoning, context, tools, MCP, tasks, permissions, and session state
to Kilo — it does not duplicate any of those capabilities.

Lifecycle (spec section 3):
  1. create_session  — POST /api/session (or v1 POST /session)
  2. send_prompt     — POST /api/session/{id}/prompt
  3. stream_response — GET /global/event (SSE) filtered by session
  4. interrupt       — POST /api/session/{id}/interrupt (abort, keep session)
  5. continue        — send_prompt again (reuses same session ID)
  6. close           — DELETE /api/session/{id}

Rule E: interrupting speech ≠ closing session. The Kilo session persists
across voice turns for context continuity.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from ..core.state_machine import State, Event, StateMachine
from .client import KiloClient, KiloServerError
from .protocol import KiloEvent
from .stream import ResponseChunk, ResponseStreamProcessor

log = logging.getLogger("alice.kilo.bridge")


@dataclass
class BridgeConfig:
    """Configuration for the Kilo bridge lifecycle."""
    auto_start_server: bool = True
    auto_resume_session: bool = True
    max_turns: int = 100
    directory: str = ""


@dataclass
class TurnResult:
    """Complete result of a single conversation turn."""
    chunks: list[ResponseChunk] = field(default_factory=list)
    full_text: str = ""
    thinking: str = ""
    tokens: dict[str, int] = field(default_factory=dict)
    cost: float = 0.0
    finish_reason: str = ""
    interrupted: bool = False
    error: str | None = None


class KiloBridge:
    """Orchestrates the full Kilo session lifecycle.

    Usage:
        bridge = KiloBridge(client, config)
        await bridge.start()

        # Normal turn
        for chunk in await bridge.send_and_stream("Hello"):
            print(chunk.text)

        # Interrupted turn
        stream_task = asyncio.create_task(bridge.stream_until_end("Long prompt..."))
        await asyncio.sleep(2)
        await bridge.interrupt()  # abort, session stays alive
        result = await stream_task

        await bridge.close()
    """

    def __init__(
        self,
        client: KiloClient | None = None,
        config: BridgeConfig | None = None,
    ):
        self.client = client or KiloClient()
        self.config = config or BridgeConfig()
        self.state_machine = StateMachine()
        self._processor = ResponseStreamProcessor()
        self._session_id: str | None = None
        self._stream_subscribed = False
        self._interrupt_requested = False

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def state(self) -> State:
        return self.state_machine.state

    # ── Session lifecycle ─────────────────────────────────────────────────

    async def start(self) -> str:
        """Create or resume a Kilo session.

        Per Rule E and spec section 3.1: if auto_resume_session is True,
        we attempt to reuse the most recent session before creating a new one.
        """
        self.config.directory = self.config.directory or self.client.directory

        if self.config.auto_start_server:
            await asyncio.to_thread(self.client.start_server)

        if self.config.auto_resume_session:
            self._session_id = await asyncio.to_thread(
                self.client.get_active_session
            )

        if not self._session_id:
            self._session_id = await asyncio.to_thread(
                self.client.create_session,
                title="Alice Voice Session",
            )
            log.info("kilo_session_created", extra={"session_id": self._session_id})
        else:
            log.info("kilo_session_resumed", extra={"session_id": self._session_id})

        self.state_machine.fire(Event.SESSION_STARTED)
        return self._session_id

    async def close(self):
        """Delete the Kilo session and stop the server if we started it."""
        if self._session_id:
            try:
                self.client.delete_session(self._session_id)
            except Exception as e:
                log.warning("kilo_session_delete_failed", extra={"error": str(e)})
            self._session_id = None

        if self.config.auto_start_server:
            self.client.stop_server()

        self.state_machine.fire(Event.SESSION_ENDED)

    # ── Prompt + streaming ────────────────────────────────────────────────

    async def send_prompt(self, text: str, **kwargs) -> dict[str, Any]:
        """Send a prompt to the active session. Returns admission info."""
        if not self._session_id:
            raise KiloServerError("No active session. Call start() first.")

        self._interrupt_requested = False
        self.state_machine.fire(Event.PROMPT_SENT)

        result = await asyncio.to_thread(
            self.client.send_prompt,
            self._session_id,
            text,
            **kwargs,
        )
        log.info(
            "kilo_prompt_sent",
            extra={"session_id": self._session_id, "prompt_length": len(text)},
        )
        return result

    async def send_and_stream(
        self,
        text: str,
        session_id: str | None = None,
    ) -> AsyncIterator[ResponseChunk]:
        """Send a prompt and immediately stream the response.

        Convenience method that combines send_prompt + stream_response.
        The prompt is sent first (non-blocking), then we subscribe to
        SSE events and yield chunks as they arrive.
        """
        await self.send_prompt(text)
        async for chunk in self.stream_response(session_id=session_id):
            yield chunk

    async def stream_response(
        self,
        session_id: str | None = None,
    ) -> AsyncIterator[ResponseChunk]:
        """Subscribe to SSE events and yield normalized response chunks.

        Filters for the given session_id. Automatically skips sync events.
        Stops when a completion event is seen (step.ended or turn.ended).
        """
        sid = session_id or self._session_id
        if not sid:
            raise KiloServerError("No session ID for streaming.")

        stream = await self.client.subscribe_events(session_id=sid)
        await stream.start()

        try:
            async for event in stream:
                if event is None:
                    break

                # Check for interrupt signal
                if self._interrupt_requested:
                    log.info("kilo_stream_interrupted", extra={"session_id": sid})
                    break

                chunks = self._processor.process(event)
                for chunk in chunks:
                    yield chunk
        finally:
            await stream.stop()

    # ── Complete turn: send + stream as one unit ──────────────────────────

    async def send_and_stream(
        self,
        text: str,
        timeout: float = 60.0,
    ) -> list[ResponseChunk]:
        """Send a prompt and collect all response chunks.

        Convenience method that returns the full chunk list (for simple callers).
        For streaming/incremental processing, use send_prompt + stream_response.
        """
        await self.send_prompt(text)
        chunks: list[ResponseChunk] = []

        async def _collect():
            async for chunk in self.stream_response():
                chunks.append(chunk)

        try:
            await asyncio.wait_for(_collect(), timeout=timeout)
        except asyncio.TimeoutError:
            log.warning("kilo_stream_timeout", extra={"session_id": self._session_id})

        return chunks

    async def send_and_wait(
        self,
        text: str,
        timeout: float = 60.0,
    ) -> TurnResult:
        """Send a prompt and wait for the complete turn result.

        Aggregates all chunks into a TurnResult with:
        - full_text (extracted from COMPLETION chunk)
        - thinking (concatenated from THINKING chunks)
        - tokens (from STEP_END chunk)
        - cost (from STEP_END chunk)
        - finish_reason (from STEP_END chunk)
        """
        chunks = await self.send_and_stream(text, timeout=timeout)

        result = TurnResult(chunks=chunks)
        for chunk in chunks:
            if chunk.type.value == "completion":
                result.full_text = chunk.data.get("full_text", chunk.text)
            elif chunk.type.value == "thinking":
                if chunk.text:
                    result.thinking += chunk.text
            elif chunk.type.value == "step_end":
                result.tokens = chunk.data.get("tokens", {})
                result.cost = chunk.data.get("cost", 0)
                result.finish_reason = chunk.data.get("reason", "")
            elif chunk.type.value == "error":
                result.error = chunk.text

        return result

    # ── Interrupt ─────────────────────────────────────────────────────────

    async def interrupt(self) -> bool:
        """Abort a running turn.

        Returns True if the interrupt was accepted by Kilo.
        Per Rule E: the session is NOT destroyed — only the current
        in-flight response is aborted.
        """
        if not self._session_id:
            return False

        self._interrupt_requested = True
        self.state_machine.fire(Event.INTERRUPTION_DETECTED)

        await asyncio.to_thread(self.client.interrupt, self._session_id)

        self.state_machine.fire(Event.INTERRUPTION_COMPLETED)
        log.info("kilo_turn_interrupted", extra={"session_id": self._session_id})
        return True

    # ── Context managers ──────────────────────────────────────────────────

    @asynccontextmanager
    async def turn(self) -> AsyncIterator["KiloBridge"]:
        """Context manager for a turn lifecycle.

        Ensures proper state transitions and cleanup.
        """
        self.state_machine.fire(Event.TURN_STARTED)
        try:
            yield self
        except Exception as e:
            self.state_machine.fire(Event.ERROR_OCCURRED)
            log.error("kilo_turn_error", extra={"error": str(e)})
            raise
        finally:
            self.state_machine.fire(Event.TURN_ENDED)

    # ── Status ────────────────────────────────────────────────────────────

    async def is_healthy(self) -> bool:
        """Check if Kilo server is reachable."""
        return await asyncio.to_thread(self.client.health)

    def has_session(self) -> bool:
        """Check if we have an active session."""
        return self._session_id is not None