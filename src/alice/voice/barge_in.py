"""Barge-in detection for voice assistant.

Implements selective barge-in with multiple confidence levels:
- NONE: no speech detected
- CANDIDATE: possible speech, needs confirmation
- CONFIRMED: definite user speech, interrupt TTS
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Optional

from .. import config
from .vad import VoiceActivityDetector

log = logging.getLogger("alice.voice.barge_in")


class BargeInLevel(Enum):
    """Barge-in confidence levels."""
    NONE = "none"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"


class BargeInDetector:
    """Detects when user is trying to interrupt Alice while she's speaking.

    Uses multiple signals to reduce false positives:
    - VAD speech detection with sliding window
    - Audio energy levels
    - Duration of speech activity
    """

    def __init__(
        self,
        sample_rate: int = config.AUDIO_SAMPLE_RATE,
        vad_sensitivity: int = config.VAD_SENSITIVITY,
        candidate_ms: int = 300,  # Time to confirm candidate -> confirmed
        min_speech_ratio: float = 0.6,  # Ratio of speech frames needed in window
        window_ms: int = 1000,  # Sliding window for speech ratio
    ):
        self._vad = VoiceActivityDetector(
            sample_rate=sample_rate,
            sensitivity=vad_sensitivity,
        )
        self._sample_rate = sample_rate
        self._candidate_ms = candidate_ms
        self._min_speech_ratio = min_speech_ratio
        self._window_ms = window_ms

        self._level = BargeInLevel.NONE
        self._candidate_start: float = 0.0
        self._frame_ms = 30  # VAD frame size
        self._frame_size = int(sample_rate * self._frame_ms / 1000)  # samples
        self._frame_bytes = self._frame_size * 2  # 16-bit
        self._window_frames = int(window_ms / self._frame_ms)
        self._speech_history: list[bool] = []

    def reset(self) -> None:
        """Reset detector state."""
        self._level = BargeInLevel.NONE
        self._candidate_start = 0.0
        self._speech_history = []
        self._vad.reset()

    def process(self, audio: bytes) -> BargeInLevel:
        """Process audio chunk and return current barge-in level.

        Args:
            audio: Raw PCM audio bytes (16-bit, mono)

        Returns:
            Current barge-in confidence level
        """
        if len(audio) < self._frame_bytes:
            return self._level

        # Process in VAD frame-sized chunks
        num_frames = len(audio) // self._frame_bytes
        for i in range(num_frames):
            start = i * self._frame_bytes
            end = start + self._frame_bytes
            frame = audio[start:end]

            vad_result = self._vad.process(frame)
            is_speech = vad_result == "speech"
            self._speech_history.append(is_speech)

        # Keep only recent history
        if len(self._speech_history) > self._window_frames:
            self._speech_history = self._speech_history[-self._window_frames:]

        # Calculate speech ratio in recent window
        if self._speech_history:
            speech_ratio = sum(self._speech_history) / len(self._speech_history)
        else:
            speech_ratio = 0.0

        # State machine for barge-in detection
        if self._level == BargeInLevel.NONE:
            if speech_ratio >= self._min_speech_ratio and len(self._speech_history) >= int(self._candidate_ms / self._frame_ms):
                self._level = BargeInLevel.CANDIDATE
                self._candidate_start = time.time()
                log.debug("barge_in_candidate", extra={"ratio": speech_ratio})

        elif self._level == BargeInLevel.CANDIDATE:
            # Check if we should confirm or reject
            if speech_ratio < self._min_speech_ratio * 0.5:
                # Not enough speech, go back to none
                self._level = BargeInLevel.NONE
                self._speech_history = []
                log.debug("barge_in_rejected")
            elif time.time() - self._candidate_start >= self._candidate_ms / 1000.0:
                # Enough time has passed with speech, confirm
                self._level = BargeInLevel.CONFIRMED
                log.debug("barge_in_confirmed")

        # CONFIRMED stays confirmed until reset

        return self._level

    @property
    def level(self) -> BargeInLevel:
        return self._level

    @property
    def is_confirmed(self) -> bool:
        return self._level == BargeInLevel.CONFIRMED

    @property
    def is_candidate(self) -> bool:
        return self._level == BargeInLevel.CANDIDATE
