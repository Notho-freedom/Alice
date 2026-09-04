"""Voice activity detection using webrtcvad."""

from __future__ import annotations

import logging

import numpy as np
import webrtcvad

from .. import config

log = logging.getLogger("alice.voice.vad")


class VoiceActivityDetector:
    """WebRTC-based VAD for voice activity detection (spec section 5).

    Distinguishes SILENCE / NOISE / SPEECH and triggers interruption.
    """

    def __init__(
        self,
        sample_rate: int = config.AUDIO_SAMPLE_RATE,
        sensitivity: int = config.VAD_SENSITIVITY,
    ):
        self._sample_rate = sample_rate
        self._vad = webrtcvad.Vad(sensitivity)
        self._frame_ms = 30
        self._frame_size = int(sample_rate * self._frame_ms / 1000)
        self._silence_count = 0
        self._speech_active = False
        self.speech_detected = False
        self.silence_detected = False

    def process(self, audio: bytes) -> str:
        """Process a chunk of 16-bit PCM audio. Returns 'speech', 'silence', or 'noise'."""
        if len(audio) < self._frame_size * 2:
            return "noise"

        is_speech = self._vad.is_speech(audio[: self._frame_size * 2], self._sample_rate)

        if is_speech:
            self._silence_count = 0
            if not self._speech_active:
                self._speech_active = True
                self.speech_detected = True
                log.debug("vad_speech_detected")
                return "speech"
            self.silence_detected = False
            return "speech"
        else:
            self._silence_count += 1
            if self._speech_active:
                self.silence_detected = True
                self._speech_active = False
                self.speech_detected = False
                log.debug("vad_speech_ended")
                return "silence"
            return "noise"

    def reset(self) -> None:
        """Reset internal state."""
        self._silence_count = 0
        self._speech_active = False
        self.speech_detected = False
        self.silence_detected = False

    @property
    def is_speech_active(self) -> bool:
        return self._speech_active
