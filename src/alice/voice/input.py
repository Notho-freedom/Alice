"""Audio input capture using sounddevice."""

from __future__ import annotations

import logging
import threading
from typing import Callable

import numpy as np
import sounddevice as sd

from .. import config

log = logging.getLogger("alice.voice.input")


class AudioInput:
    """Captures audio from the microphone with a callback interface."""

    def __init__(
        self,
        device: int = config.AUDIO_INPUT_DEVICE,
        sample_rate: int = config.AUDIO_SAMPLE_RATE,
        channels: int = config.AUDIO_CHANNELS,
        callback: Callable[[bytes], None] | None = None,
    ):
        self._device = device
        self._sample_rate = sample_rate
        self._channels = channels
        self._callback = callback
        self._stream: sd.InputStream | None = None
        self._listening = False

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
                if self._callback:
                    pcm_bytes = data[0].tobytes()
                    self._callback(pcm_bytes)
            except Exception as e:
                log.error("audio_input_error", extra={"error": str(e)})
                break

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

    @property
    def is_listening(self) -> bool:
        return self._listening
