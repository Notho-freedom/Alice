"""Response stream processor for Kilo SSE events.

Consumes KiloEvent objects from the SSE stream and emits
normalized ResponseChunk events for the voice/terminal layer.

Pipeline (per spec section 8):
  Kilo SSE events → Response Stream Processor → Speech Queue → TTS
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .protocol import (
    KiloEvent,
    COMPLETION_EVENTS,
    TextDelta,
    TextEnded,
    TextStarted,
    ToolStarted,
    ToolEnded,
    StepStarted,
    StepEnded,
    PromptAdmitted,
    SessionError,
)

log = logging.getLogger("alice.kilo.stream")


class ChunkType(str, Enum):
    TEXT = "text"
    TOOL_START = "tool_start"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    STEP_START = "step_start"
    STEP_END = "step_end"
    ERROR = "error"
    COMPLETION = "completion"
    STOP = "stop"


@dataclass
class ResponseChunk:
    """A normalized response chunk emitted to consumers."""
    type: ChunkType
    text: str = ""
    data: dict[str, Any] = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}


class ResponseStreamProcessor:
    """Processes KiloEvent objects into ResponseChunk objects.

    Distinguishes text, tool activity, thinking, completion, and error
    per M3 spec requirement.
    """

    def __init__(self):
        self._event_handlers = {
            "session.next.text.started": self._on_text_started,
            "session.next.text.delta": self._on_text_delta,
            "session.next.text.ended": self._on_text_ended,
            "session.next.tool.started": self._on_tool_started,
            "session.next.tool.ended": self._on_tool_ended,
            "session.next.reasoning.started": self._on_reasoning_started,
            "session.next.reasoning.delta": self._on_reasoning_delta,
            "session.next.reasoning.ended": self._on_reasoning_ended,
            "session.next.step.started": self._on_step_started,
            "session.next.step.ended": self._on_step_ended,
            "session.next.prompt.admitted": self._on_prompt_admitted,
            "session.error": self._on_error,
        }

    def process(self, event: KiloEvent) -> list[ResponseChunk]:
        """Convert a KiloEvent into one or more ResponseChunks."""
        handler = self._event_handlers.get(event.event_type)
        if handler:
            result = handler(event)
            if result is None:
                return []
            if isinstance(result, list):
                return result
            return [result]
        return []


    # ── Handlers ──────────────────────────────────────────────────────

    def _on_text_started(self, event: KiloEvent) -> ResponseChunk:
        log.debug("text_started", extra={"session_id": event.session_id})
        return ResponseChunk(type=ChunkType.TEXT, text="", data={"started": True})

    def _on_text_delta(self, event: KiloEvent) -> ResponseChunk:
        delta = event.properties.get("delta", "")
        return ResponseChunk(type=ChunkType.TEXT, text=delta)

    def _on_text_ended(self, event: KiloEvent) -> ResponseChunk:
        text = event.properties.get("text", "")
        return ResponseChunk(
            type=ChunkType.COMPLETION,
            text=text,
            data={"full_text": text},
        )

    def _on_tool_started(self, event: KiloEvent) -> ResponseChunk:
        props = event.properties
        return ResponseChunk(
            type=ChunkType.TOOL_START,
            data={
                "tool": props.get("tool", ""),
                "call_id": props.get("callID", ""),
            },
        )

    def _on_tool_ended(self, event: KiloEvent) -> ResponseChunk:
        props = event.properties
        status = props.get("status", props.get("state", {}).get("status", ""))
        return ResponseChunk(
            type=ChunkType.TOOL_RESULT,
            data={
                "tool": props.get("tool", ""),
                "call_id": props.get("callID", ""),
                "status": status,
                "output": props.get("output", ""),
            },
        )

    def _on_reasoning_started(self, event: KiloEvent) -> ResponseChunk:
        log.debug("reasoning_started", extra={"session_id": event.session_id})
        return ResponseChunk(type=ChunkType.THINKING, data={"started": True})

    def _on_reasoning_delta(self, event: KiloEvent) -> ResponseChunk:
        delta = event.properties.get("delta", "")
        return ResponseChunk(type=ChunkType.THINKING, text=delta)

    def _on_reasoning_ended(self, event: KiloEvent) -> ResponseChunk:
        return ResponseChunk(type=ChunkType.THINKING, data={"ended": True})

    def _on_step_started(self, event: KiloEvent) -> ResponseChunk:
        model = event.properties.get("model", {})
        return ResponseChunk(
            type=ChunkType.STEP_START,
            data={
                "agent": event.properties.get("agent", ""),
                "model_id": model.get("modelID", "") if isinstance(model, dict) else "",
                "provider_id": model.get("providerID", "") if isinstance(model, dict) else "",
            },
        )

    def _on_step_ended(self, event: KiloEvent) -> ResponseChunk:
        props = event.properties
        return ResponseChunk(
            type=ChunkType.STEP_END,
            data={
                "reason": props.get("finish", ""),
                "cost": props.get("cost", 0),
                "tokens": props.get("tokens", {}),
            },
        )

    def _on_prompt_admitted(self, event: KiloEvent) -> ResponseChunk:
        prompt = event.properties.get("prompt", {})
        return ResponseChunk(
            type=ChunkType.STEP_START,
            data={
                "prompt_text": prompt.get("text", "") if isinstance(prompt, dict) else "",
            },
        )

    def _on_error(self, event: KiloEvent) -> ResponseChunk:
        error = event.properties.get("error", "")
        log.error("kilo_session_error", extra={"error": str(error)})
        return ResponseChunk(
            type=ChunkType.ERROR,
            text=str(error),
            data={"error": error},
        )
