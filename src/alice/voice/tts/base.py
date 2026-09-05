"""Abstract TTS interface and providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Callable

import numpy as np

from .. import config


class TextToSpeech(ABC):
    """Abstract text-to-speech interface (spec section 7).

    Implementations must be interchangeable without modifying the core.
    """

    def __init__(self):
        self._echo_reference_callback: Callable[[bytes], None] | None = None

    @abstractmethod
    async def speak(self, text: str, language: str | None = None) -> bool | None:
        """Speak the given text fully, blocking until done.

        Returns True on success, False on failure, None for providers
        that don't track success/failure.
        """

    @abstractmethod
    async def stream_speak(self, chunks: AsyncIterator[str], language: str | None = None) -> None:
        """Speak text from an async iterator of chunks.

        Should interleave with VAD monitoring so speech can be interrupted.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Stop any ongoing speech immediately."""

    @abstractmethod
    async def pause(self) -> None:
        """Pause ongoing speech."""

    @abstractmethod
    async def resume(self) -> None:
        """Resume paused speech."""

    @abstractmethod
    def is_speaking(self) -> bool:
        """Return True if currently speaking."""

    def set_echo_reference_callback(self, callback: Callable[[bytes], None]) -> None:
        """Set callback to receive audio playback data for echo cancellation."""
        self._echo_reference_callback = callback

    def _send_echo_reference(self, audio_bytes: bytes) -> None:
        """Send audio data to echo reference callback if set."""
        if self._echo_reference_callback:
            try:
                self._echo_reference_callback(audio_bytes)
            except Exception:
                pass

    async def _play_int16(self, audio_int16: np.ndarray, sample_rate: int, stop_flag_getter) -> None:
        """Shared playback path for int16 PCM via sounddevice.

        Providers should convert decoded audio to int16 and call this.
        """
        try:
            import asyncio
            import sounddevice as sd

            self._send_echo_reference(audio_int16.tobytes())

            sd.play(
                audio_int16,
                samplerate=sample_rate,
                device=config.AUDIO_OUTPUT_DEVICE,
                blocking=False,
            )

            while not stop_flag_getter():
                await asyncio.sleep(0.05)
                stream = sd.get_stream()
                if stream is not None and not stream.active:
                    break

            if stop_flag_getter():
                sd.stop()
        except ImportError:
            raise
        except Exception as e:
            raise RuntimeError(f"audio_playback_failed: {e}") from e
