"""Wake word detection base + simple implementations."""

from __future__ import annotations

import logging
import queue
import threading
from abc import ABC, abstractmethod

import numpy as np

from .. import config

log = logging.getLogger("alice.voice.wake")


class WakeWordDetector(ABC):
    """Abstract wake word detector. Must be interchangeable (spec section 6.2)."""

    @abstractmethod
    def process(self, audio: bytes) -> bool:
        """Process an audio chunk. Returns True if the wake word was detected."""


class KeywordWakeWord(WakeWordDetector):
    """Simple keyword matching via VAD envelope detection.

    A lightweight fallback that detects a sustained pause followed by speech,
    treating any new speech episode as a wake-up trigger.
    """

    def __init__(self, vad):
        self._vad = vad
        self._keyword = config.WAKE_WORD
        self._reset()

    def _reset(self):
        self._waiting_for_speech = True
        self._detected = False

    def process(self, audio: bytes) -> bool:
        if self._detected:
            return True

        vad_result = self._vad.process(audio)
        if vad_result == "speech" and self._waiting_for_speech:
            self._waiting_for_speech = False
            self._detected = True
            log.info("wake_word_detected")
            return True
        return False

    def reset(self):
        self._reset()


class PassthroughWakeWord(WakeWordDetector):
    """No wake word — always returns False until reset."""

    def process(self, audio: bytes) -> bool:
        return False

    def reset(self):
        pass
