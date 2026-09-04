"""Voice assistant orchestrator.

Combines audio input, VAD, STT, Kilo bridge, SSE streaming, TTS,
interruption, and wake word into a single conversation loop.

Spec sections:
  - 4: Voice input (Mic → VAD → STT → Text → Kilo)
  - 5: VAD (Silence / Noise / Speech)
  - 6: Wake word + conversation continuity
  - 8: Streaming (Kilo SSE → text chunks → TTS)
  - 9-10: Interruption (speech during TTS → stop → capture)
  - 11: State machine
"""

from __future__ import annotations

import asyncio
import logging
import time
import wave
from dataclasses import dataclass, field

from .. import config
from ..kilo.client import KiloClient
from ..kilo.session import SessionManager
from ..kilo.stream import ResponseStreamProcessor, ChunkType
from ..core.state_machine import StateMachine, State, Event
from ..core.lifecycle import Lifecycle, AppConfig
from ..core.events import VoiceEvent, AppEvent
from ..voice.input import AudioInput
from ..voice.vad import VoiceActivityDetector
from ..voice.wake import KeywordWakeWord
from ..voice.stt import create_stt
from ..voice.tts import create_tts
from ..voice.interruption import InterruptionHandler

log = logging.getLogger("alice.voice.assistant")


@dataclass
class VoiceConfig:
    input_device: int = 1
    output_device: int = 3
    sample_rate: int = 16000
    stt_provider: str = "deepgram"
    tts_provider: str = "auto"
    wake_word: str = "hey assistant"
    ptt: bool = True  # push-to-talk mode
    continuous: bool = False  # conversation continuity (M8)
    voice: str | None = None  # TTS voice name or ID
    language: str | None = None  # Language code (e.g., "fr", "en")


class VoiceAssistant:
    """Orchestrates the full voice → Kilo → voice conversation loop."""

    def __init__(self, voice_config: VoiceConfig | None = None):
        self.cfg = voice_config or VoiceConfig()
        self.lifecycle = Lifecycle(AppConfig(
            directory=config.KILO_DIRECTORY,
            kilo_base_url=config.KILO_BASE_URL,
            auto_approve=config.KILO_AUTO_APPROVE,
        ))
        self.state_machine = StateMachine()
        self._kilo: KiloClient | None = None
        self._sessions: SessionManager | None = None
        self._processor = ResponseStreamProcessor()
        self._stream = None

        # Voice components
        self._audio: AudioInput | None = None
        self._vad = VoiceActivityDetector()
        self._wake = KeywordWakeWord(self._vad)
        self._stt: OpenAISTT | None = None
        self._tts: PyTTSX3TTS | None = None
        self._interrupter: InterruptionHandler | None = None

        # State tracking
        self._transcription_buffer = bytearray()
        self._speaking_text: list[str] = []
        self._text_chunks_received = False

    async def initialize(self):
        """Start Kilo server, create session, initialize voice components."""
        self.lifecycle.startup()

        # Kilo
        self._kilo = KiloClient()
        if not self._kilo.health():
            log.info("kilo_starting")
            self._kilo.start_server()
        self._sessions = SessionManager(self._kilo)
        self.lifecycle.context.session_id = self._sessions.get_or_create(
            title="Alice Voice Session"
        )

        # STT
        self._stt = create_stt()
        if self._stt:
            log.info("stt_provider_available", extra={"provider": config.STT_PROVIDER})
        else:
            log.warning("stt_no_provider", extra={"provider": config.STT_PROVIDER})

        # TTS
        self._tts = create_tts(
            voice=self.cfg.voice,
            language=self.cfg.language,
        )
        if self._tts:
            log.info("tts_provider_available", extra={"provider": config.TTS_PROVIDER})
        else:
            log.warning("tts_unavailable")

        # Interruption handler (only if we have TTS)
        if self._tts:
            self._interrupter = InterruptionHandler(self._vad)
            self._interrupter.set_tts_stop_callback(
                lambda: asyncio.create_task(self._tts.stop())
            )
            self._interrupter.set_kilo_interrupt_callback(
                lambda: self._kilo.interrupt(self.lifecycle.context.session_id)
            )

        # Wire state machine transitions
        self.state_machine.on_transition(self._on_state_change)

    def _on_state_change(self, transition):
        """Called on every state transition - can be used for UI updates."""
        log.info("state_transition", extra={
            "from": transition.from_state.value,
            "to": transition.to_state.value,
            "event": transition.event.value,
        })

    # ── Push-to-talk mode ─────────────────────────────────────────────

    async def run_push_to_talk(self, key: str = "ctrl"):
        """Run in push-to-talk mode. Hold Ctrl to speak."""
        print("\n  Alice Voice Assistant (Push-to-Talk)")
        print(f"  Session: {self.lifecycle.context.session_id[:24]}...")
        print(f"  Hold {key.upper()} to speak, release to send.")
        print("  Type '/quit' in another terminal or press Ctrl+C to exit.\n")

        self._audio = AudioInput(callback=self._on_audio_chunk)

        while self.lifecycle.context.running:
            try:
                press = await asyncio.get_event_loop().run_in_executor(
                    None, self._wait_for_keypress, key
                )
                if not press:
                    break

                await self._ptt_cycle()
            except KeyboardInterrupt:
                break

        await self.cleanup()

    def _wait_for_keypress(self, key: str) -> bool:
        """Simple keypress detection. Returns False if quit requested."""
        import keyboard  # optional
        return False

    async def _ptt_cycle(self):
        """One PTT cycle: listen → STT → Kilo → TTS."""
        # 1. LISTENING state
        self.state_machine.fire(Event.START_LISTENING)
        self._vad.reset()

        # 2. Capture audio while key is held
        self._audio.start()
        print("\r  [Listening...]   ", end="", flush=True)

        # In a real implementation, we'd capture until key release
        # For now, capture for a fixed duration
        await asyncio.sleep(0.1)

        # Simulate key release and stop
        self._audio.stop()
        self.state_machine.fire(Event.SPEECH_ENDED)

        # 3. Transcribe
        audio_data = bytes(self._transcription_buffer)
        if len(audio_data) < 1000:
            print("  (nothing heard)\n")
            return

        self.state_machine.fire(Event.PROMPT_SENT)
        try:
            if self._stt:
                text = await self._stt.transcribe(audio_data)
            else:
                text = ""  # fallback
        except Exception as e:
            log.error("stt_failed", extra={"error": str(e)})
            print(f"  STT error: {e}\n")
            self.state_machine.fire(Event.ERROR_OCCURRED)
            self.state_machine.fire(Event.RECOVER)
            return

        self._transcription_buffer.clear()
        if not text.strip():
            print("  (empty transcription)\n")
            return

        print(f"  You: {text}\n")

        # 4. Send to Kilo and stream response
        await self._send_and_speak(text)

    # ── Conversation continuity mode (M8) ─────────────────────────────

    async def run_continuous(self):
        """Run in continuous conversation mode with VAD-based activation (M8).

        After wake word activation, the user can continue speaking
        without re-activating, until a silence period ends the turn.
        """
        print("\n  Alice Voice Assistant (Continuous Mode)")
        print(f"  Session: {self.lifecycle.context.session_id[:24]}...")
        print(f"  Speak naturally. Silence > {config.VAD_MIN_SILENCE_MS}ms ends a turn.")
        print("  Press Ctrl+C to exit.\n")

        self._audio = AudioInput(callback=self._on_audio_chunk)
        self._audio.start()

        # Start in LISTENING state immediately
        self.state_machine.fire(Event.START_LISTENING)
        print("  [Listening...]", end="", flush=True)

        while self.lifecycle.context.running:
            await asyncio.sleep(0.05)

            # Handle speech → STT → Kilo → TTS cycle
            if self.state_machine.state == State.THINKING:
                await self._handle_transcription()

        await self.cleanup()

    # ── Audio callback ────────────────────────────────────────────────

    def _on_audio_chunk(self, audio: bytes):
        """Called for each audio chunk from the microphone (callback thread).

        Handles VAD detection, interruption during TTS (spec section 9),
        and audio buffering for STT during listening (spec section 4).
        """
        # Feed VAD
        vad_result = self._vad.process(audio)

        # ── Rule A: speech during SPEAKING → immediate interruption ──
        if self.state_machine.state == State.SPEAKING:
            if vad_result == "speech":
                # Rule D: keep Kilo session intact (don't destroy)
                # Rule E: interrupt speech ≠ destroy session
                self._handle_interruption()
            return

        # ── During INTERRUPTING: capture new speech ──
        if self.state_machine.state == State.INTERRUPTING:
            if vad_result == "speech":
                self.state_machine.fire(Event.SPEECH_DETECTED)
            elif vad_result == "silence":
                self.state_machine.fire(Event.SPEECH_ENDED)
            return

        # ── Accumulate audio for STT during LISTENING/TRANSCRIBING ──
        if self.state_machine.state in (State.LISTENING, State.TRANSCRIBING):
            self._transcription_buffer.extend(audio)

            if vad_result == "speech":
                if self.state_machine.state == State.LISTENING:
                    self.state_machine.fire(Event.SPEECH_DETECTED)
            elif vad_result == "silence":
                if self.state_machine.state == State.TRANSCRIBING:
                    self.state_machine.fire(Event.SPEECH_ENDED)

    def _handle_interruption(self):
        """User spoke during TTS — immediate interruption (spec section 9)."""
        log.info("interruption_triggered")
        self.state_machine.fire(Event.INTERRUPTION_DETECTED)

        # Rule A: stop TTS immediately (sync for pyttsx3, async for edge)
        if self._tts:
            try:
                if hasattr(self._tts, '_engine'):
                    self._tts._engine.stop()
                else:
                    asyncio.create_task(self._tts.stop())
                self._tts._playing = False
            except Exception as e:
                log.warning("tts_stop_failed", extra={"error": str(e)})

        # Rule E: do NOT destroy Kilo session
        if self._kilo:
            self._kilo.interrupt(self.lifecycle.context.session_id)

        self.state_machine.fire(Event.INTERRUPTION_COMPLETED)
        # After interruption, we go to LISTENING for new speech

    async def _handle_transcription(self):
        """Process the transcription buffer: STT → Kilo → TTS."""
        audio_data = bytes(self._transcription_buffer)
        self._transcription_buffer.clear()

        if len(audio_data) < 1000:
            # No meaningful audio - transition back to LISTENING
            self.state_machine.fire(Event.RECOVER)
            self.state_machine.fire(Event.START_LISTENING)
            print("\r  [Listening...]      ", end="", flush=True)
            return

        # Transcribe
        try:
            if self._stt:
                text = await self._stt.transcribe(audio_data)
            else:
                text = ""
        except Exception as e:
            log.error("stt_failed", extra={"error": str(e)})
            print(f"\n  STT error: {e}\n")
            self.state_machine.fire(Event.ERROR_OCCURRED)
            self.state_machine.fire(Event.RECOVER)
            self.state_machine.fire(Event.START_LISTENING)
            print("  [Listening...]", end="", flush=True)
            return

        if not text.strip():
            print("\r  [Listening...]      ", end="", flush=True)
            self.state_machine.fire(Event.RECOVER)
            self.state_machine.fire(Event.START_LISTENING)
            return

        print(f"\r  You: {text}\n", flush=True)

        # Send to Kilo and speak
        await self._send_and_speak(text)

    # ── Kilo response → TTS pipeline ────────────────────────────────

    async def _send_and_speak(self, text: str):
        """Send text to Kilo, stream response, and speak it via TTS."""
        sid = self.lifecycle.context.session_id

        # Start SSE stream first (before sending prompt to catch all events)
        self._stream = await self._kilo.subscribe_events(sid)
        await self._stream.start()
        self._interrupt_requested = False

        # Send prompt
        self._sessions.touch(sid)
        self._kilo.send_prompt(sid, text)

        # Queue for streaming TTS
        speech_queue: asyncio.Queue[str | None] = asyncio.Queue()

        # Start TTS consumer task
        tts_task = asyncio.create_task(self._tts_consumer(speech_queue))

        # Start interruption monitoring during SPEAKING
        if self._interrupter and self._tts:
            self._interrupter.start_monitoring()

        # Collect events and feed speech queue
        self._text_chunks_received = False
        got_completion = False
        timeout = time.time() + 120
        text_buffer = ""

        while time.time() < timeout and not got_completion:
            try:
                event = await asyncio.wait_for(self._stream.next_event(), timeout=3)
            except asyncio.TimeoutError:
                if got_completion:
                    break
                continue

            if event is None:
                break

            chunks = self._processor.process(event)
            for chunk in chunks:
                if chunk.type == ChunkType.TEXT and chunk.text:
                    text_buffer += chunk.text
                    self._text_chunks_received = True
                    await speech_queue.put(chunk.text)
                elif chunk.type == ChunkType.COMPLETION:
                    got_completion = True
                elif chunk.type == ChunkType.TOOL_START:
                    print(f"\n[Using tool: {chunk.data.get('tool', '')}]", flush=True)
                elif chunk.type == ChunkType.STEP_END:
                    pass  # handled after loop
                elif chunk.type == ChunkType.ERROR:
                    log.error("kilo_response_error", extra={"error": chunk.text})
                    print(f"\n[ERROR: {chunk.text}]", flush=True)
                    got_completion = True

        # Signal end of speech queue
        await speech_queue.put(None)
        try:
            await asyncio.wait_for(tts_task, timeout=30)
        except asyncio.TimeoutError:
            log.warning("tts_consumer_timeout")
            tts_task.cancel()

        if self._interrupter:
            self._interrupter.stop_monitoring()

        await self._stream.stop()
        self._stream = None

        # Return to listening state (state transitions happened in _speak_text)
        if self.state_machine.state != State.LISTENING:
            self.state_machine.fire(Event.TTS_COMPLETED)  # SPEAKING -> IDLE if needed
            self.state_machine.fire(Event.START_LISTENING)  # IDLE -> LISTENING
        print(f"\r  [Listening...]      ", end="", flush=True)

    async def _tts_consumer(self, queue: asyncio.Queue):
        """Consume text chunks from queue and speak them."""
        pending = ""
        while True:
            chunk = await queue.get()
            if chunk is None:
                # End of stream
                if pending:
                    await self._speak_text(pending)
                    pending = ""
                break

            pending += chunk

            # If the pending text contains sentence-ending punctuation,
            # flush a sentence to allow interruption boundaries
            if any(p in pending for p in ".!?.\n") and len(pending) > 10:
                # Speak the accumulated text
                await self._speak_text(pending)
                pending = ""

        # Speak any remaining
        if pending:
            await self._speak_text(pending)

    async def _speak_text(self, text: str):
        """Speak text, checking for interruption."""
        if not self._tts or not text.strip():
            return

        # Check if we're in INTERRUPTING state (interrupted)
        if self.state_machine.state == State.INTERRUPTING:
            return

        # Get detected language from STT for appropriate voice selection
        detected_lang = None
        if self._stt and hasattr(self._stt, "detected_language"):
            detected_lang = self._stt.detected_language

        self.state_machine.fire(Event.TTS_STARTED)
        try:
            await self._tts.speak(text, detected_lang)
        except Exception as e:
            log.error("tts_error", extra={"error": str(e)})
        self.state_machine.fire(Event.TTS_COMPLETED)

    async def cleanup(self):
        """Clean up resources."""
        if self._audio:
            self._audio.stop()
        if self._interrupter:
            self._interrupter.stop_monitoring()
        if self._stream:
            await self._stream.stop()
        if self._kilo:
            self._kilo.stop_server()
        self.lifecycle.shutdown()
