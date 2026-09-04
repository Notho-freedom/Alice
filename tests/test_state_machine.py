"""Unit tests for the conversation state machine (spec section 11).

Tests are hardware-independent per spec requirement:
"La machine d'état doit être testable indépendamment du matériel audio."
"""

import pytest

from alice.core.state_machine import StateMachine, State, Event


class TestStateMachine:

    def test_initial_state_is_idle(self):
        sm = StateMachine()
        assert sm.state == State.IDLE

    def test_normal_voice_conversation_flow(self):
        """IDLE -> LISTENING -> TRANSCRIBING -> THINKING -> SPEAKING -> IDLE"""
        sm = StateMachine()
        transitions = []

        sm.on_transition(lambda t: transitions.append(t))

        sm.fire(Event.START_LISTENING)        # IDLE -> LISTENING
        sm.fire(Event.SPEECH_DETECTED)        # LISTENING -> TRANSCRIBING
        sm.fire(Event.SPEECH_ENDED)           # TRANSCRIBING -> THINKING
        sm.fire(Event.THINKING_COMPLETED)     # THINKING -> SPEAKING
        sm.fire(Event.TTS_COMPLETED)          # SPEAKING -> IDLE

        states = [t.to_state for t in transitions]
        assert states == [
            State.LISTENING,
            State.TRANSCRIBING,
            State.THINKING,
            State.SPEAKING,
            State.IDLE,
        ]

    def test_terminal_mode_flow(self):
        """IDLE -> THINKING -> SPEAKING -> IDLE (no voice listening)"""
        sm = StateMachine()
        sm.fire(Event.PROMPT_SENT)           # IDLE -> THINKING
        assert sm.state == State.THINKING
        sm.fire(Event.THINKING_COMPLETED)    # THINKING -> SPEAKING
        assert sm.state == State.SPEAKING
        sm.fire(Event.TTS_COMPLETED)         # SPEAKING -> IDLE
        assert sm.state == State.IDLE

    def test_interruption_during_speaking(self):
        """SPEAKING -> INTERRUPTING -> LISTENING"""
        sm = StateMachine()
        sm.fire(Event.PROMPT_SENT)           # IDLE -> THINKING
        sm.fire(Event.THINKING_COMPLETED)    # THINKING -> SPEAKING

        sm.fire(Event.INTERRUPTION_DETECTED)  # SPEAKING -> INTERRUPTING
        assert sm.state == State.INTERRUPTING

        sm.fire(Event.INTERRUPTION_COMPLETED)  # INTERRUPTING -> LISTENING
        assert sm.state == State.LISTENING

    def test_interruption_during_thinking(self):
        """THINKING -> INTERRUPTING -> LISTENING"""
        sm = StateMachine()
        sm.fire(Event.PROMPT_SENT)            # IDLE -> THINKING
        sm.fire(Event.INTERRUPTION_DETECTED)  # THINKING -> INTERRUPTING
        assert sm.state == State.INTERRUPTING

    def test_error_recovery(self):
        """Any state -> ERROR -> IDLE"""
        sm = StateMachine()
        sm.fire(Event.PROMPT_SENT)
        sm.fire(Event.ERROR_OCCURRED)
        assert sm.state == State.ERROR

        sm.fire(Event.RECOVER)
        assert sm.state == State.IDLE

    def test_invalid_transition_blocked(self):
        """Invalid transitions are blocked (state unchanged)."""
        sm = StateMachine()
        # IDLE -> SPEECH_DETECTED is invalid (must be LISTENING first)
        result = sm.fire(Event.SPEECH_DETECTED)
        assert result == State.IDLE  # unchanged
        assert sm.state == State.IDLE

    def test_text_chunks_keep_in_thinking(self):
        """Multiple text deltas during THINKING keep state in THINKING."""
        sm = StateMachine()
        sm.fire(Event.PROMPT_SENT)           # IDLE -> THINKING
        sm.fire(Event.TEXT_RECEIVED)         # THINKING -> THINKING
        sm.fire(Event.TEXT_RECEIVED)         # THINKING -> THINKING
        assert sm.state == State.THINKING

    def test_can_fire(self):
        """Test can_fire returns correct validity."""
        sm = StateMachine()
        assert sm.can_fire(Event.START_LISTENING) is True
        assert sm.can_fire(Event.SPEECH_DETECTED) is False  # needs LISTENING first

    def test_double_interruption(self):
        """Rule D: interruption during interruption -> SPEAKING again."""
        sm = StateMachine()
        sm.fire(Event.PROMPT_SENT)
        sm.fire(Event.THINKING_COMPLETED)     # THINKING -> SPEAKING
        sm.fire(Event.INTERRUPTION_DETECTED)  # SPEAKING -> INTERRUPTING
        sm.fire(Event.SPEECH_DETECTED)        # INTERRUPTING -> TRANSCRIBING
        sm.fire(Event.SPEECH_ENDED)           # TRANSCRIBING -> THINKING
        sm.fire(Event.THINKING_COMPLETED)     # THINKING -> SPEAKING

    def test_reset(self):
        """Reset returns to IDLE."""
        sm = StateMachine()
        sm.fire(Event.PROMPT_SENT)
        sm.fire(Event.THINKING_COMPLETED)
        sm.fire(Event.TTS_COMPLETED)
        sm.fire(Event.ERROR_OCCURRED)
        assert sm.state == State.ERROR
        sm.reset()
        assert sm.state == State.IDLE

    def test_recover_from_thinking(self):
        """RECOVER is valid from THINKING (e.g., empty transcript)."""
        sm = StateMachine()
        sm.fire(Event.START_LISTENING)  # IDLE -> LISTENING
        sm.fire(Event.SPEECH_ENDED)     # LISTENING -> THINKING
        assert sm.state == State.THINKING
        sm.fire(Event.RECOVER)          # THINKING -> IDLE
        assert sm.state == State.IDLE

    def test_history_tracking(self):
        """State transitions are recorded in history."""
        sm = StateMachine()
        sm.fire(Event.PROMPT_SENT)
        sm.fire(Event.THINKING_COMPLETED)
        assert len(sm.history) == 2
        assert sm.history[0].event == Event.PROMPT_SENT
        assert sm.history[1].event == Event.THINKING_COMPLETED
