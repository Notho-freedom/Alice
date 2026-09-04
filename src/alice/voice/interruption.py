"""Real-time interruption handling (spec sections 9-10, 22).

When the user speaks during TTS playback, the VAD immediately
triggers interruption: TTS stops, the Kilo session is aborted,
and the new speech is captured for transcription.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from .. import config
from .vad import VoiceActivityDetector

log = logging.getLogger("alice.voice.interruption")


@dataclass
class InterruptResult:
    interrupt_ms: float = 0
    threshold_ms: float = 0
    interrupted: bool = False


class InterruptionHandler:
    """Monitors VAD during TTS and triggers interruption on speech."""

    def __init__(self, vad: VoiceActivityDetector):
        self._vad = vad
        self._tts_stop_cb: callable | None = None
        self._kilo_interrupt_cb: callable | None = None
        self._monitor_task: asyncio.Task | None = None
        self._monitoring = False

    def set_tts_stop_callback(self, cb: callable) -> None:
        """Callback to immediately stop TTS playback."""
        self._tts_stop_cb = cb

    def set_kilo_interrupt_callback(self, cb: callable) -> None:
        """Callback to interrupt the Kilo session."""
        self._kilo_interrupt_cb = cb

    def start_monitoring(self) -> None:
        """Begin monitoring for interruption during TTS."""
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        log.info("interruption_monitoring_started")

    def stop_monitoring(self) -> None:
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None
        log.info("interruption_monitoring_stopped")

    async def _monitor_loop(self):
        """Continuously check VAD during TTS for speech."""
        threshold_samples = int(
            config.INTERRUPT_THRESHOLD_MS * config.AUDIO_SAMPLE_RATE / 1000
        )
        frame_size = self._vad._frame_size * 2
        silence_streak = 0

        while self._monitoring:
            try:
                await asyncio.sleep(0.01)  # 10ms poll
            except asyncio.CancelledError:
                break

            # When we have a new audio chunk, check VAD
            # The actual audio chunk is processed externally and sets
            # vad.speech_detected
            if self._vad.speech_detected:
                start_time = time.time()

                # Rule A: speech during TTS → immediate interruption
                if self._tts_stop_cb:
                    self._tts_stop_cb()
                    log.info("tts_interrupted")

                # Rule C: produce a new conversational event
                if self._kilo_interrupt_cb:
                    self._kilo_interrupt_cb()
                    log.info("kilo_interrupted")

                self._monitoring = False
                elapsed_ms = (time.time() - start_time) * 1000
                self._interruption_result = InterruptResult(
                    interrupted=True,
                    interrupt_ms=elapsed_ms,
                    threshold_ms=config.INTERRUPT_THRESHOLD_MS,
                )
                log.info("interruption_complete", extra={
                    "interrupt_ms": elapsed_ms,
                    "threshold_ms": config.INTERRUPT_THRESHOLD_MS,
                })
                break

    @property
    def monitoring(self) -> bool:
        return self._monitoring
