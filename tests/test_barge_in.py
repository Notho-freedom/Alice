"""Tests for barge-in detection."""

from __future__ import annotations

import struct

import pytest

from alice.voice.barge_in import BargeInDetector, BargeInLevel
from alice import config


def _make_silence(sample_rate: int = 16000, ms: int = 100) -> bytes:
    samples = int(sample_rate * ms / 1000)
    return b"\x00\x00" * samples


def _make_tone(sample_rate: int = 16000, freq: float = 440.0, ms: int = 100) -> bytes:
    samples = int(sample_rate * ms / 1000)
    import math
    data = []
    for i in range(samples):
        val = int(32767 * math.sin(2 * math.pi * freq * i / sample_rate))
        data.append(struct.pack("<h", val))
    return b"".join(data)


class TestBargeInDetector:
    """Test barge-in detection."""

    def test_initial_level_none(self):
        """Test initial level is NONE."""
        detector = BargeInDetector()
        assert detector.level == BargeInLevel.NONE

    def test_silence_stays_none(self):
        """Test silence keeps level at NONE."""
        detector = BargeInDetector()
        audio = _make_silence()
        for _ in range(10):
            detector.process(audio)
        assert detector.level == BargeInLevel.NONE

    def test_short_speech_becomes_candidate(self):
        """Test short speech becomes CANDIDATE."""
        detector = BargeInDetector(candidate_ms=100, min_speech_ratio=0.15)
        tone = _make_tone(ms=500)

        detector.reset()
        detector.process(tone)
        assert detector.level == BargeInLevel.CANDIDATE

    def test_candidate_with_enough_time_becomes_confirmed(self):
        """Test CANDIDATE becomes CONFIRMED after enough time."""
        detector = BargeInDetector(candidate_ms=50, min_speech_ratio=0.15)
        tone = _make_tone(ms=500)

        detector.reset()
        for i in range(5):
            detector.process(tone)
            if detector.level == BargeInLevel.CANDIDATE:
                # Simulate time passing
                detector._candidate_start = 0.0

        assert detector.level == BargeInLevel.CONFIRMED
        assert detector.is_confirmed is True

    def test_candidate_reverts_on_silence(self):
        """Test CANDIDATE reverts to NONE on silence."""
        detector = BargeInDetector(candidate_ms=50, min_speech_ratio=0.15)
        silence = _make_silence()
        tone = _make_tone(ms=500)

        detector.reset()
        detector.process(tone)
        assert detector.level == BargeInLevel.CANDIDATE

        # Feed silence to revert
        for _ in range(20):
            detector.process(silence)
        assert detector.level == BargeInLevel.NONE

    def test_reset_clears_state(self):
        """Test reset clears all state."""
        detector = BargeInDetector(candidate_ms=100)
        silence = _make_silence()
        tone = _make_tone(ms=150)

        detector.process(tone)
        assert detector.level == BargeInLevel.CANDIDATE

        detector.reset()
        assert detector.level == BargeInLevel.NONE

    def test_is_confirmed_property(self):
        """Test is_confirmed property."""
        detector = BargeInDetector()
        assert detector.is_confirmed is False
        assert detector.is_candidate is False
