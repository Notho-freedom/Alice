"""Abstract STT interface and providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator


class SpeechRecognizer(ABC):
    """Abstract speech-to-text interface (spec section 4.1).

    Implementations must be interchangeable without modifying the core.
    """

    @abstractmethod
    async def start(self) -> None:
        """Start listening for speech."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop listening."""

    @abstractmethod
    async def transcribe(self, audio: bytes) -> str:
        """Transcribe a complete audio buffer. Returns text."""

    @abstractmethod
    async def stream(self, audio_chunk: bytes) -> AsyncIterator[str]:
        """Stream audio chunks and yield partial transcriptions."""
