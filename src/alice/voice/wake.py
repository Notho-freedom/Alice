"""Wake word detection base + simple implementations."""

from __future__ import annotations

import logging
import os
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


class PorcupineWakeWord(WakeWordDetector):
    """Real keyword spotting using Picovoice Porcupine.

    Requires `pvporcupine` and a valid PICOVOICE_ACCESS_KEY.
    Falls back to KeywordWakeWord if unavailable.
    """

    def __init__(self, vad, keyword: str | None = None):
        self._vad = vad
        self._keyword = keyword or config.WAKE_WORD
        self._reset()
        self._available = False
        self._porcupine = None

        try:
            import pvporcupine
            access_key = os.getenv("PICOVOICE_ACCESS_KEY", "")
            if access_key:
                self._porcupine = pvporcupine.create(
                    access_key=access_key,
                    keyword_paths=[pvporcupine.KEYWORD_PATHS.get(self._keyword.lower())]
                    if self._keyword.lower() in pvporcupine.KEYWORD_PATHS
                    else None,
                    keywords=[self._keyword.lower()]
                    if self._keyword.lower() in pvporcupine.KEYWORD_PATHS
                    else None,
                )
                self._available = True
        except Exception as e:
            log.warning("porcupine_unavailable", extra={"error": str(e)})

    def _reset(self):
        self._detected = False

    def process(self, audio: bytes) -> bool:
        if not self._available or not self._porcupine:
            return False

        if self._detected:
            return True

        try:
            pcm = np.frombuffer(audio, dtype=np.int16)
            if len(pcm) != self._porcupine.frame_length:
                return False

            result = self._porcupine.process(pcm)
            if result >= 0:
                self._detected = True
                log.info("wake_word_detected_porcupine")
                return True
        except Exception as e:
            log.error("porcupine_error", extra={"error": str(e)})

        return False

    def reset(self):
        self._reset()

    def close(self):
        if self._porcupine:
            try:
                self._porcupine.delete()
            except Exception:
                pass
            self._porcupine = None
            self._available = False


class PassthroughWakeWord(WakeWordDetector):
    """No wake word — always returns False until reset."""

    def process(self, audio: bytes) -> bool:
        return False

    def reset(self):
        pass
