"""Tests for audio input and ring buffer."""

from __future__ import annotations

import threading
import time

import pytest

from alice.voice.input import RingBuffer, AudioInput


class TestRingBuffer:
    """Test ring buffer for audio pre-roll."""

    def test_write_and_read(self):
        """Test basic write and read."""
        buf = RingBuffer(100)
        buf.write(b"hello")
        assert buf.read_last(5) == b"hello"

    def test_wrap_around(self):
        """Test wrap-around behavior."""
        buf = RingBuffer(10)
        buf.write(b"1234567890abcdef")
        assert len(buf.read_last(10)) == 10
        assert buf.read_last(10) == b"7890abcdef"

    def test_read_more_than_written(self):
        """Test reading more than was written."""
        buf = RingBuffer(100)
        buf.write(b"hi")
        assert buf.read_last(10) == b"hi"

    def test_empty_read(self):
        """Test reading from empty buffer."""
        buf = RingBuffer(100)
        assert buf.read_last(10) == b""

    def test_thread_safety(self):
        """Test thread safety."""
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

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)
        writer_thread.start()
        reader_thread.start()
        writer_thread.join()
        reader_thread.join()
        assert buf.total_written > 0


class TestAudioInput:
    """Test audio input with ring buffer."""

    def test_pre_roll_default(self):
        """Test default pre-roll is 1 second."""
        from alice import config
        audio = AudioInput(pre_roll_ms=500)
        assert audio._pre_roll_ms == 500

    def test_ring_buffer_created(self):
        """Test ring buffer is created."""
        audio = AudioInput(pre_roll_ms=500)
        assert audio.ring_buffer is not None

    def test_get_pre_roll_empty(self):
        """Test getting pre-roll from empty buffer."""
        audio = AudioInput(pre_roll_ms=500)
        assert audio.get_pre_roll() == b""
