"""Tests for the Kilo SSE protocol parsing (spec M3)."""

import json
import pytest

from alice.kilo.protocol import (
    KiloEvent,
    parse_event_line,
)
from alice.kilo.stream import ResponseStreamProcessor, ChunkType


class TestProtocolParsing:

    def test_parse_text_delta(self):
        line = 'data: {"type":"session.next.text.delta","properties":{"delta":"Hello","sessionID":"ses123"}}'
        result = parse_event_line(line)
        assert result is not None
        event = KiloEvent.from_payload(result)
        assert event is not None
        assert event.event_type == "session.next.text.delta"
        assert event.session_id == "ses123"
        assert event.properties["delta"] == "Hello"

    def test_parse_sync_skipped(self):
        """Sync events (duplicate normalized copies) should be skipped."""
        line = 'data: {"type":"sync","syncEvent":{"type":"session.next.text.delta.1"}}'
        result = parse_event_line(line)
        event = KiloEvent.from_payload(result)
        assert event is None

    def test_parse_empty_line(self):
        assert parse_event_line("") is None
        assert parse_event_line(": heartbeat") is None

    def test_parse_malformed_json(self):
        line = 'data: {not valid json'
        assert parse_event_line(line) is None

    def test_parse_step_started(self):
        line = 'data: {"type":"session.next.step.started","properties":{"agent":"build","model":{"modelID":"gpt-4","providerID":"openai"},"sessionID":"ses123"}}'
        result = parse_event_line(line)
        event = KiloEvent.from_payload(result)
        assert event.event_type == "session.next.step.started"


class TestStreamProcessor:

    def setup_method(self):
        self.processor = ResponseStreamProcessor()

    def _make_event(self, event_type, properties=None):
        payload = {"type": event_type, "properties": properties or {}}
        return KiloEvent.from_payload(payload)

    def test_text_delta_produces_text_chunk(self):
        event = self._make_event("session.next.text.delta", {"delta": "Hello ", "sessionID": "ses1"})
        chunks = self.processor.process(event)
        assert len(chunks) == 1
        assert chunks[0].type == ChunkType.TEXT
        assert chunks[0].text == "Hello "

    def test_text_started_produces_chunk(self):
        event = self._make_event("session.next.text.started", {"sessionID": "ses1"})
        chunks = self.processor.process(event)
        assert len(chunks) == 1
        assert chunks[0].type == ChunkType.TEXT
        assert chunks[0].data.get("started") is True

    def test_text_ended_produces_completion(self):
        event = self._make_event("session.next.text.ended", {"text": "Hello world", "sessionID": "ses1"})
        chunks = self.processor.process(event)
        assert len(chunks) == 1
        assert chunks[0].type == ChunkType.COMPLETION
        assert chunks[0].data["full_text"] == "Hello world"

    def test_tool_started_produces_tool_chunk(self):
        event = self._make_event("session.next.tool.started", {
            "tool": "read", "callID": "call1", "input": {"path": "/test"},
            "sessionID": "ses1"
        })
        chunks = self.processor.process(event)
        assert len(chunks) == 1
        assert chunks[0].type == ChunkType.TOOL_START
        assert chunks[0].data["tool"] == "read"

    def test_step_ended_produces_step_end(self):
        event = self._make_event("session.next.step.ended", {
            "finish": "stop", "cost": 0.01, "tokens": {"total": 100},
            "sessionID": "ses1"
        })
        chunks = self.processor.process(event)
        assert len(chunks) == 1
        assert chunks[0].type == ChunkType.STEP_END
        assert chunks[0].data["reason"] == "stop"

    def test_error_event_produces_error_chunk(self):
        event = self._make_event("session.error", {"error": {"message": "timeout"}, "sessionID": "ses1"})
        chunks = self.processor.process(event)
        assert len(chunks) == 1
        assert chunks[0].type == ChunkType.ERROR

    def test_unknown_event_produces_no_chunks(self):
        event = self._make_event("some.unknown.event", {"sessionID": "ses1"})
        chunks = self.processor.process(event)
        assert len(chunks) == 0

    def test_sync_filtered(self):
        """Sync events should produce no chunks."""
        payload = {"type": "sync", "syncEvent": {"type": "session.next.text.delta.1"}}
        event = KiloEvent.from_payload(payload)
        assert event is None

    def test_multiple_delta_chunks_concatenate(self):
        """Multiple text deltas should concatenate into full response."""
        deltas = ["Hello", " ", "world"]
        full_text = ""
        for delta in deltas:
            event = self._make_event("session.next.text.delta", {"delta": delta, "sessionID": "ses1"})
            chunks = self.processor.process(event)
            full_text += chunks[0].text
        assert full_text == "Hello world"
