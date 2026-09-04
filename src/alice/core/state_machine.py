"""Conversation state machine for Alice.

States per spec section 11:
  IDLE -> LISTENING -> TRANSCRIBING -> THINKING -> SPEAKING -> LISTENING
  SPEAKING -> INTERRUPTING -> LISTENING
  ANY STATE -> ERROR -> IDLE

The state machine is testable independently of audio hardware
(spec section 11: "La machine d'état doit être testable indépendamment du matériel audio").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable

log = logging.getLogger("alice.core.state_machine")


class State(Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTING = "INTERRUPTING"
    ERROR = "ERROR"


class Event(Enum):
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    TURN_STARTED = "turn_started"
    TURN_ENDED = "turn_ended"
    WAKE_WORD_DETECTED = "wake_word_detected"
    START_LISTENING = "start_listening"
    SPEECH_DETECTED = "speech_detected"
    SPEECH_ENDED = "speech_ended"
    PROMPT_SENT = "prompt_sent"
    THINKING_STARTED = "thinking_started"
    TEXT_RECEIVED = "text_received"
    THINKING_COMPLETED = "thinking_completed"
    TTS_STARTED = "tts_started"
    TTS_COMPLETED = "tts_completed"
    INTERRUPTION_DETECTED = "interruption_detected"
    INTERRUPTION_COMPLETED = "interruption_completed"
    ERROR_OCCURRED = "error_occurred"
    RECOVER = "recover"


_TRANSITIONS: dict[State, dict[Event, State]] = {
    State.IDLE: {
        Event.SESSION_STARTED: State.IDLE,
        Event.SESSION_ENDED: State.IDLE,
        Event.START_LISTENING: State.LISTENING,
        Event.PROMPT_SENT: State.THINKING,
        Event.TURN_STARTED: State.LISTENING,
        Event.ERROR_OCCURRED: State.ERROR,
    },
    State.LISTENING: {
        Event.SPEECH_DETECTED: State.TRANSCRIBING,
        Event.SPEECH_ENDED: State.THINKING,
        Event.PROMPT_SENT: State.THINKING,
        Event.SESSION_ENDED: State.IDLE,
        Event.ERROR_OCCURRED: State.ERROR,
        Event.RECOVER: State.IDLE,
    },
    State.TRANSCRIBING: {
        Event.SPEECH_ENDED: State.THINKING,
        Event.PROMPT_SENT: State.THINKING,
        Event.TEXT_RECEIVED: State.TRANSCRIBING,
        Event.SESSION_ENDED: State.IDLE,
        Event.ERROR_OCCURRED: State.ERROR,
    },
    State.THINKING: {
        Event.TEXT_RECEIVED: State.THINKING,
        Event.THINKING_STARTED: State.THINKING,
        Event.THINKING_COMPLETED: State.SPEAKING,
        Event.INTERRUPTION_DETECTED: State.INTERRUPTING,
        Event.SESSION_ENDED: State.IDLE,
        Event.ERROR_OCCURRED: State.ERROR,
        Event.RECOVER: State.IDLE,
    },
    State.SPEAKING: {
        Event.INTERRUPTION_DETECTED: State.INTERRUPTING,
        Event.TTS_COMPLETED: State.IDLE,
        Event.TTS_STARTED: State.SPEAKING,
        Event.TEXT_RECEIVED: State.SPEAKING,
        Event.SESSION_ENDED: State.IDLE,
        Event.ERROR_OCCURRED: State.ERROR,
    },
    State.INTERRUPTING: {
        Event.INTERRUPTION_COMPLETED: State.LISTENING,
        Event.SPEECH_DETECTED: State.TRANSCRIBING,
        Event.SESSION_ENDED: State.IDLE,
        Event.RECOVER: State.IDLE,
        Event.ERROR_OCCURRED: State.ERROR,
    },
    State.ERROR: {
        Event.RECOVER: State.IDLE,
        Event.START_LISTENING: State.LISTENING,
        Event.SESSION_STARTED: State.IDLE,
        Event.SESSION_ENDED: State.IDLE,
    },
}


@dataclass
class StateTransition:
    event: Event
    from_state: State
    to_state: State


class StateMachine:
    """Finite state machine for conversation flow.

    Usage:
        sm = StateMachine()
        sm.on_transition(handler)
        sm.fire(Event.START_LISTENING)  # IDLE -> LISTENING
    """

    def __init__(self):
        self._state: State = State.IDLE
        self._history: list[StateTransition] = []
        self._handlers: list[Callable[[StateTransition], None]] = []

    @property
    def state(self) -> State:
        return self._state

    def fire(self, event: Event) -> State:
        """Attempt a state transition. Returns the new state."""
        current = self._state
        next_state = _TRANSITIONS.get(current, {}).get(event)

        if next_state is None:
            log.warning(
                "state_transition_blocked",
                extra={"current": current.value, "event": event.value},
            )
            return current

        transition = StateTransition(
            event=event,
            from_state=current,
            to_state=next_state,
        )
        self._state = next_state
        self._history.append(transition)

        for handler in self._handlers:
            handler(transition)

        log.info(
            "state_change",
            extra={
                "from": current.value,
                "to": next_state.value,
                "event": event.value,
            },
        )
        return next_state

    def can_fire(self, event: Event) -> bool:
        """Check if an event is valid in the current state."""
        return _TRANSITIONS.get(self._state, {}).get(event) is not None

    def on_transition(self, handler: Callable[[StateTransition], None]):
        """Register a handler called on every state transition."""
        self._handlers.append(handler)

    @property
    def history(self) -> list[StateTransition]:
        return list(self._history)

    def reset(self):
        """Reset to IDLE state (e.g. after error recovery)."""
        self._state = State.IDLE
        self._history.clear()
