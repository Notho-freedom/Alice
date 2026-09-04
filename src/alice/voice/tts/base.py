"""Abstract TTS interface and providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator


class TextToSpeech(ABC):
    """Abstract text-to-speech interface (spec section 7).

    Implementations must be interchangeable without modifying the core.
    """

    @abstractmethod
    async def speak(self, text: str) -> None:
        """Speak the given text fully, blocking until done."""

    @abstractmethod
    async def stream_speak(self, chunks: AsyncIterator[str]) -> None:
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
