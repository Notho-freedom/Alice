"""Audio input capture using sounddevice."""

from __future__ import annotations

import collections
import logging
import threading
from typing import Callable

import numpy as np
import sounddevice as sd

from .. import config

log = logging.getLogger("alice.voice.input")


class RingBuffer:
    """Thread-safe ring buffer for audio data."""

    def __init__(self, max_bytes: int):
        self._max_bytes = max_bytes
        self._buffer = bytearray(max_bytes)
        self._view = memoryview(self._buffer)
        self._write_pos = 0
        self._total_written = 0
        self._lock = threading.Lock()

    def write(self, data: bytes) -> None:
        """Write data to the ring buffer."""
        with self._lock:
            data_len = len(data)
            if data_len >= self._max_bytes:
                self._buffer[:] = data[-self._max_bytes:]
                self._write_pos = 0
                self._total_written = self._max_bytes
                return

            if self._write_pos + data_len <= self._max_bytes:
                self._buffer[self._write_pos:self._write_pos + data_len] = data
                self._write_pos = (self._write_pos + data_len) % self._max_bytes
            else:
                # Wrap around
                first_part = self._max_bytes - self._write_pos
                self._buffer[self._write_pos:] = data[:first_part]
                self._buffer[:data_len - first_part] = data[first_part:]
                self._write_pos = (self._write_pos + data_len) % self._max_bytes

            self._total_written += data_len

    def read_last(self, num_bytes: int) -> bytes:
        """Read the last num_bytes from the buffer."""
        with self._lock:
            if self._total_written == 0:
                return b""

            if num_bytes > self._total_written:
                num_bytes = self._total_written

            if num_bytes > self._max_bytes:
                num_bytes = self._max_bytes

            if self._total_written < self._max_bytes:
                # Buffer not full, data starts at 0
                start = max(0, self._total_written - num_bytes)
                return bytes(self._buffer[start:self._total_written])

            # Buffer is full, need to wrap around
            end_pos = self._write_pos
            start_pos = (end_pos - num_bytes) % self._max_bytes

            if start_pos < end_pos:
                return bytes(self._buffer[start_pos:end_pos])
            else:
                # Wrapped around
                part1 = bytes(self._buffer[start_pos:])
                part2 = bytes(self._buffer[:end_pos])
                return part1 + part2

    @property
    def total_written(self) -> int:
        with self._lock:
            return self._total_written


class AudioInput:
    """Captures audio from the microphone with a callback interface."""

    def __init__(
        self,
        device: int = config.AUDIO_INPUT_DEVICE,
        sample_rate: int = config.AUDIO_SAMPLE_RATE,
        channels: int = config.AUDIO_CHANNELS,
        callback: Callable[[bytes], None] | None = None,
        pre_roll_ms: int = 1000,
        tts_reference_callback: Callable[[bytes], None] | None = None,
    ):
        self._device = device
        self._sample_rate = sample_rate
        self._channels = channels
        self._callback = callback
        self._stream: sd.InputStream | None = None
        self._listening = False

        # Ring buffer for continuous audio capture (pre-roll + barge-in)
        pre_roll_bytes = int(sample_rate * pre_roll_ms / 1000) * 2  # 16-bit
        ring_buffer_bytes = max(pre_roll_bytes * 4, 1024 * 1024)  # At least 1MB
        self._ring_buffer = RingBuffer(ring_buffer_bytes)
        self._pre_roll_ms = pre_roll_ms

        # Simple echo suppression: keep a short reference of TTS playback
        self._tts_reference_callback = tts_reference_callback
        self._echo_history = collections.deque(maxlen=20)
        self._echo_energy = 0.0

    @property
    def ring_buffer(self) -> RingBuffer:
        return self._ring_buffer

    def start(self) -> None:
        """Start capturing audio. Calls callback with raw PCM bytes per chunk."""
        if self._stream is not None:
            return

        self._listening = True
        self._stream = sd.InputStream(
            device=self._device,
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            blocksize=int(self._sample_rate * config.AUDIO_CHUNK_SIZE / 1024),
        )
        self._stream.start()
        log.info("audio_input_started", extra={"device": self._device})

        # Start a reader thread since sounddevice uses callbacks
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def stop(self) -> None:
        """Stop capturing audio."""
        self._listening = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        log.info("audio_input_stopped")

    def _read_loop(self):
        frames = int(self._sample_rate * 0.1)  # 100ms chunks
        while self._listening and self._stream:
            try:
                data = self._stream.read(frames)
                pcm_bytes = data[0].tobytes()

                # Always write to ring buffer for continuous recording
                self._ring_buffer.write(pcm_bytes)

                # Simple echo suppression: attenuate chunks that look like playback
                filtered = self._suppress_echo(pcm_bytes)

                # Call callback if provided
                if self._callback:
                    self._callback(filtered)
            except Exception as e:
                log.error("audio_input_error", extra={"error": str(e)})
                break

    def _suppress_echo(self, pcm_bytes: bytes) -> bytes:
        """Apply simple echo suppression based on TTS reference energy."""
        try:
            if not pcm_bytes:
                return pcm_bytes

            current = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
            energy = float(np.mean(current ** 2))

            if self._echo_energy > 0 and energy > self._echo_energy * 0.05:
                attenuation = max(0.1, 1.0 - min(energy / (self._echo_energy * 4), 0.9))
                current *= attenuation
                return current.astype(np.int16).tobytes()

            return pcm_bytes
        except Exception:
            return pcm_bytes

    def update_echo_reference(self, pcm_bytes: bytes) -> None:
        """Update echo reference from TTS playback."""
        if not pcm_bytes:
            return

        try:
            reference = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
            if reference.size:
                self._echo_energy = float(np.mean(reference ** 2))
                self._echo_history.append(self._echo_energy)
                if len(self._echo_history) > 1:
                    self._echo_energy = float(np.mean(self._echo_history))
        except Exception:
            pass

    def clear_echo_reference(self) -> None:
        """Clear echo reference after TTS stops."""
        self._echo_history.clear()
        self._echo_energy = 0.0

    def read_chunk(self) -> bytes | None:
        """Read one chunk of audio. Used for push-to-talk mode."""
        if not self._stream:
            return None
        frames = int(self._sample_rate * 0.1)
        try:
            data = self._stream.read(frames)
            return data[0].tobytes()
        except Exception:
            return None

    def get_pre_roll(self) -> bytes:
        """Get pre-roll audio from the ring buffer."""
        pre_roll_bytes = int(self._sample_rate * self._pre_roll_ms / 1000) * 2  # 16-bit
        return self._ring_buffer.read_last(pre_roll_bytes)

    @property
    def is_listening(self) -> bool:
        return self._listening
