"""Tests for audio input, AEC and barge-in."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from alice.voice.input import RingBuffer, AudioInput, SimpleAEC
from alice import config


def _make_tone(sample_rate: int = 16000, freq: float = 440.0, ms: int = 100) -> bytes:
    samples = int(sample_rate * ms / 1000)
    data = []
    for i in range(samples):
        val = int(32767 * np.sin(2 * np.pi * freq * i / sample_rate))
        data.append(val)
    return np.int16(data).tobytes()


class TestRingBuffer:
    def test_write_and_read(self):
        buf = RingBuffer(100)
        buf.write(b"hello")
        assert buf.read_last(5) == b"hello"

    def test_wrap_around(self):
        buf = RingBuffer(10)
        buf.write(b"1234567890abcdef")
        assert len(buf.read_last(10)) == 10
        assert buf.read_last(10) == b"7890abcdef"

    def test_read_more_than_written(self):
        buf = RingBuffer(100)
        buf.write(b"hi")
        assert buf.read_last(10) == b"hi"

    def test_empty_read(self):
        buf = RingBuffer(100)
        assert buf.read_last(10) == b""

    def test_thread_safety(self):
        buf = RingBuffer(1000)
        data = b"x" * 100

        def writer():
            for _ in range(100):
                buf.write(data)
                time.sleep(0.001)

        def reader():
            for _ in range(100):
                buf.read_last(100)
                time.sleep(0.001)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert buf.total_written > 0


class TestAudioInput:
    def test_pre_roll_default(self):
        audio = AudioInput(pre_roll_ms=500)
        assert audio._pre_roll_ms == 500

    def test_ring_buffer_created(self):
        audio = AudioInput(pre_roll_ms=500)
        assert audio.ring_buffer is not None

    def test_get_pre_roll_empty(self):
        audio = AudioInput(pre_roll_ms=500)
        assert audio.get_pre_roll() == b""

    def test_echo_suppression_attenuates(self):
        audio = AudioInput(pre_roll_ms=500)

        ref = np.int16([10000] * 100).tobytes()
        audio.update_echo_reference(ref)

        test_audio = np.int16([10000] * 100).tobytes()
        suppressed = audio._suppress_echo(test_audio)

        suppressed_arr = np.frombuffer(suppressed, dtype=np.int16)
        original_arr = np.frombuffer(test_audio, dtype=np.int16)
        assert np.mean(np.abs(suppressed_arr)) < np.mean(np.abs(original_arr))

    def test_clear_echo_reference(self):
        audio = AudioInput(pre_roll_ms=500)
        ref = np.int16([10000] * 100).tobytes()
        audio.update_echo_reference(ref)
        assert audio._echo_energy > 0

        audio.clear_echo_reference()
        assert audio._echo_energy == 0.0
        assert len(audio._echo_history) == 0

    def test_aec_created_when_enabled(self):
        original = config.ECHO_CANCELLATION_ENABLED
        try:
            config.ECHO_CANCELLATION_ENABLED = True
            audio = AudioInput(pre_roll_ms=500)
            assert audio._aec is not None
        finally:
            config.ECHO_CANCELLATION_ENABLED = original

    def test_aec_process_reduces_echo(self):
        aec = SimpleAEC()
        mic = _make_tone(ms=50)
        ref = _make_tone(ms=50)

        mic_arr = np.frombuffer(mic, dtype=np.int16).astype(np.float32)
        ref_arr = np.frombuffer(ref, dtype=np.int16).astype(np.float32)

        processed = aec.process(mic_arr, ref_arr)
        assert processed.size > 0

    def test_aec_reset_clears_state(self):
        aec = SimpleAEC()
        mic = _make_tone(ms=50)
        ref = _make_tone(ms=50)
        aec.process(np.frombuffer(mic, dtype=np.int16).astype(np.float32), np.frombuffer(ref, dtype=np.int16).astype(np.float32))
        aec.reset()
        assert np.all(aec._filter == 0)
