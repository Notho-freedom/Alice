"""Protocol-level event types from Kilo's SSE event stream.

Events arrive on GET /global/event as:
  {"directory": "...", "payload": {"id": "...", "type": "...", "properties": {...}}}

We skip 'sync' events (duplicate normalized copies) and focus on native event types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import json


@dataclass
class KiloEvent:
    """A parsed Kilo SSE event."""
    event_type: str
    session_id: str
    properties: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> KiloEvent | None:
        """Parse a 'payload' object from the SSE stream."""
        event_type = payload.get("type", "unknown")
        if event_type == "sync":
            return None

        props = payload.get("properties", {})
        session_id = props.get("sessionID", "")

        return cls(
            event_type=event_type,
            session_id=session_id,
            properties=props,
            raw=payload,
        )


@dataclass
class TextDelta(KiloEvent):
    delta: str = ""


@dataclass
class TextStarted(KiloEvent):
    pass


@dataclass
class TextEnded(KiloEvent):
    text: str = ""


@dataclass
class StepStarted(KiloEvent):
    agent: str = ""
    model_id: str = ""
    provider_id: str = ""


@dataclass
class StepEnded(KiloEvent):
    reason: str = ""
    cost: float = 0
    tokens: dict = field(default_factory=dict)


@dataclass
class ToolStarted(KiloEvent):
    tool: str = ""
    call_id: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class ToolEnded(KiloEvent):
    tool: str = ""
    call_id: str = ""
    output: str = ""
    status: str = ""


@dataclass
class PromptAdmitted(KiloEvent):
    prompt_text: str = ""
    delivery: str = ""


@dataclass
class SessionError(KiloEvent):
    error: Any = None


# ── Event classification helpers ────────────────────────────────────────
TEXT_EVENTS = {
    "session.next.text.started",
    "session.next.text.delta",
    "session.next.text.ended",
}

TOOL_EVENTS = {
    "session.next.tool.started",
    "session.next.tool.ended",
    "session.next.tool.requested",
}

STEP_EVENTS = {
    "session.next.step.started",
    "session.next.step.ended",
}

COMPLETION_EVENTS = {
    "session.next.step.ended",
    "session.next.turn.ended",
}


def parse_event_line(line: str) -> dict[str, Any] | None:
    """Parse a line from the SSE stream. Returns the data dict or None."""
    if not line or not line.startswith("data: "):
        return None
    try:
        raw = line[6:]
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
