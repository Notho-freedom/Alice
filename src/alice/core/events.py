"""Event definitions shared across voice and terminal interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class VoiceEvent(Enum):
    """High-level events the conversation engine reacts to."""
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    TURN_STARTED = "turn_started"
    TURN_ENDED = "turn_ended"
    WAKE_WORD_DETECTED = "wake_word_detected"
    PTT_PRESSED = "ptt_pressed"
    PTT_RELEASED = "ptt_released"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    TRANSCRIPTION_READY = "transcription_ready"
    ASSISTANT_STARTED = "assistant_started"
    TEXT_CHUNK = "text_chunk"
    ASSISTANT_FINISHED = "assistant_finished"
    INTERRUPTION = "interruption"
    ERROR = "error"


@dataclass
class AppEvent:
    """A high-level event flowing through the conversation engine."""
    type: VoiceEvent
    text: str = ""
    data: dict[str, Any] | None = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}
