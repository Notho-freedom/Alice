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
from ..kilo.bridge import KiloBridge
from ..core.state_machine import StateMachine, State, Event
from ..core.lifecycle import Lifecycle, AppConfig
from ..voice.input import AudioInput
from ..voice.vad import VoiceActivityDetector
from ..voice.wake import KeywordWakeWord, PorcupineWakeWord, PassthroughWakeWord
from ..voice.stt import create_stt
from ..voice.tts import create_tts
from ..voice.interruption import InterruptionHandler
from ..voice.barge_in import BargeInDetector, BargeInLevel

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
        self._bridge: KiloBridge | None = None

        # Voice components
        self._audio: AudioInput | None = None
        self._vad = VoiceActivityDetector()
        self._wake: KeywordWakeWord | PorcupineWakeWord | PassthroughWakeWord | None = None
        self._stt = None
        self._tts = None
        self._interrupter: InterruptionHandler | None = None
        self._barge_in = BargeInDetector()

        # State tracking
        self._transcription_buffer = bytearray()
        self._speaking_text: list[str] = []
        self._text_chunks_received = False
        self._echo_reference = b""
        self._last_barge_in_level = BargeInLevel.NONE
        self._interrupt_requested = False

    async def initialize(self):
        """Start Kilo server, create session, initialize voice components."""
        self.lifecycle.startup()

        self._bridge = KiloBridge()
        await self._bridge.start()
        self.lifecycle.context.session_id = self._bridge.session_id

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
                lambda: asyncio.create_task(self._bridge.interrupt())
            )

        # Wire state machine transitions
        self.state_machine.on_transition(self._on_state_change)

        # Wake word: prefer Porcupine if available, else fallback
        try:
            self._wake = PorcupineWakeWord(self._vad, keyword=self.cfg.wake_word)
            if getattr(self._wake, "_available", False):
                log.info("wake_word_provider_available", extra={"provider": "porcupine"})
            else:
                self._wake = KeywordWakeWord(self._vad)
                log.info("wake_word_provider_available", extra={"provider": "keyword"})
        except Exception as e:
            self._wake = KeywordWakeWord(self._vad)
            log.warning("wake_word_fallback", extra={"error": str(e)})

    def _on_state_change(self, transition):
        """Called on every state transition - can be used for UI updates."""
        log.info("state_transition", extra={
            "from": transition.from_state.value,
            "to": transition.to_state.value,
            "event": transition.event.value,
        })

        # Reset barge-in detector when starting to speak
        if transition.to_state == State.SPEAKING:
            self._barge_in.reset()

    # ── Push-to-talk mode ─────────────────────────────────────────────

    async def run_push_to_talk(self, key: str = "ctrl"):
        """Run in push-to-talk mode. Hold Ctrl to speak."""
        print("\n  Alice Voice Assistant (Push-to-Talk)")
        print(f"  Session: {self.lifecycle.context.session_id[:24]}...")
        print(f"  Hold {key.upper()} to speak, release to send.")
        print("  Type '/quit' in another terminal or press Ctrl+C to exit.\n")

        self._audio = AudioInput(
            device=self.cfg.input_device,
            callback=self._on_audio_chunk,
        )

        if self._tts and self._audio:
            self._tts.set_echo_reference_callback(self._audio.update_echo_reference)

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
        """Wait for a keypress. Returns True if the key was pressed, False to quit."""
        try:
            import keyboard
            keyboard.wait(key)
            return True
        except ImportError:
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

        In continuous mode, the assistant listens for speech and transcribes,
        then sends to Kilo. Kilo decides whether the text is addressing the
        assistant — if not, it responds with "[IGNORED]" and Alice stays silent.
        """
        print("\n  Alice Voice Assistant (Conversation Mode)")
        print(f"  Session: {self.lifecycle.context.session_id[:24]}...")
        print(f"  Speak naturally. Silence > {config.VAD_MIN_SILENCE_MS}ms ends a turn.")
        if self._wake and hasattr(self._wake, "_available") and self._wake._available:
            print("  Wake word: Porcupine active")
        else:
            print("  Wake word: basic VAD fallback")
        print("  Press Ctrl+C to exit.\n")

        self._audio = AudioInput(
            device=self.cfg.input_device,
            callback=self._on_audio_chunk,
        )
        self._audio.start()

        if self._tts and self._audio:
            self._tts.set_echo_reference_callback(self._audio.update_echo_reference)

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

        Handles VAD detection, wake-word activation, barge-in during TTS,
        and audio buffering for STT during listening.
        """
        # Feed VAD for state transitions
        vad_result = self._vad.process(audio)

        # ── Wake-word activation from IDLE ──
        if self.state_machine.state == State.IDLE and self._wake:
            if self._wake.process(audio):
                self.state_machine.fire(Event.WAKE_WORD_DETECTED)
                self.state_machine.fire(Event.START_LISTENING)
                return

        # ── Rule A: speech during SPEAKING → barge-in detection ──
        if self.state_machine.state == State.SPEAKING:
            barge_level = self._barge_in.process(audio)

            if barge_level == BargeInLevel.CONFIRMED:
                # Grab pre-roll before interrupting
                pre_roll = self._audio.get_pre_roll() if self._audio else b""
                self._handle_interruption(pre_roll=pre_roll)
            return

        # ── During INTERRUPTING: capture new speech ──
        if self.state_machine.state == State.INTERRUPTING:
            self._transcription_buffer.extend(audio)

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

    def _handle_interruption(self, pre_roll: bytes = b""):
        """User spoke during TTS — barge-in interruption with pre-roll capture."""
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

        # Clear echo reference when TTS stops
        if self._audio:
            self._audio.clear_echo_reference()

        # Rule E: do NOT destroy Kilo session
        if self._bridge:
            asyncio.create_task(self._bridge.interrupt())

        # Capture pre-roll audio so we don't lose the start of the user's speech
        self._transcription_buffer.clear()
        if pre_roll:
            self._transcription_buffer.extend(pre_roll)
            log.info("pre_roll_captured", extra={"bytes": len(pre_roll)})

        # Continue capturing new speech in INTERRUPTING state
        # Don't fire INTERRUPTION_COMPLETED here — let the VAD flow continue

    async def _handle_transcription(self):
        """Process the transcription buffer: STT → Kilo → TTS."""
        audio_data = bytes(self._transcription_buffer)
        self._transcription_buffer.clear()

        if len(audio_data) < 16000:
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

        # Lightweight correction for common mishearings of the assistant name
        corrections = {
            "adi": "Alice",
            "alisa": "Alice",
            "aliza": "Alice",
            "alis": "Alice",
        }
        lowered = text.strip().lower()
        for wrong, right in corrections.items():
            if lowered == wrong:
                text = right
                break

        print(f"\r  You: {text}\n", flush=True)

        # Add context instruction for continuous mode — Kilo decides if text is addressing Alice
        prompt = text
        if self.cfg.continuous:
            prompt = (
                "[SYSTEM INSTRUCTION: You are Alice, a local voice assistant. "
                "If the user is directly addressing you or asking you something, respond normally in the same language. "
                "If the user is NOT addressing you (just talking to themselves, the computer, or off-topic), "
                "respond with exactly: [IGNORED] and nothing else.]\n\n"
                + text
            )

        # Send to Kilo and speak
        await self._send_and_speak(prompt)

    # ── Kilo response → TTS pipeline ────────────────────────────────

    async def _send_and_speak(self, text: str):
        """Send text to Kilo, stream response, and speak it via TTS."""
        self._interrupt_requested = False

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
        text_buffer = ""

        async for chunk in self._bridge.send_and_stream(text):
            if chunk.type.value == "text":
                text_buffer += chunk.text
                self._text_chunks_received = True

                if text_buffer.strip().upper().startswith("[IGNORED]"):
                    self._interrupt_requested = True
                    continue

                await speech_queue.put(chunk.text)
            elif chunk.type.value == "completion":
                got_completion = True
            elif chunk.type.value == "tool_start":
                print(f"\n[Using tool: {chunk.data.get('tool', '')}]", flush=True)
            elif chunk.type.value == "error":
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

        # If Kilo indicated the message was not for Alice, skip TTS
        if text_buffer.strip().upper().startswith("[IGNORED]"):
            log.info("message_not_for_alice", extra={"transcript": text_buffer[:100]})
            if self.state_machine.state == State.THINKING:
                self.state_machine.fire(Event.THINKING_COMPLETED)
            if self.state_machine.state == State.SPEAKING:
                self.state_machine.fire(Event.TTS_COMPLETED)
            self.state_machine.fire(Event.START_LISTENING)
            print(f"\r  [Ignored - listening...]      ", end="", flush=True)
            return

        # Return to listening state (state transitions happened in _speak_text)
        if self.state_machine.state == State.SPEAKING:
            self.state_machine.fire(Event.TTS_COMPLETED)
        if self.state_machine.state == State.IDLE:
            self.state_machine.fire(Event.START_LISTENING)
        elif self.state_machine.state != State.LISTENING:
            self.state_machine.fire(Event.RECOVER)
            self.state_machine.fire(Event.START_LISTENING)
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

        # Reset barge-in detector before speaking
        self._barge_in.reset()

        # Get detected language from STT for appropriate voice selection
        detected_lang = None
        if self._stt and hasattr(self._stt, "detected_language"):
            detected_lang = self._stt.detected_language

        self.state_machine.fire(Event.TTS_STARTED)
        try:
            await self._tts.speak(text, detected_lang)
        except Exception as e:
            log.error("tts_error", extra={"error": str(e)})
        finally:
            # Clear echo reference when TTS completes
            if self._audio:
                self._audio.clear_echo_reference()
        self.state_machine.fire(Event.TTS_COMPLETED)

    async def cleanup(self):
        """Clean up resources."""
        if self._audio:
            self._audio.stop()
        if self._interrupter:
            self._interrupter.stop_monitoring()
        if hasattr(self, "_bridge"):
            await self._bridge.close()
        self.lifecycle.shutdown()
