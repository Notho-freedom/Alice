"""STT provider using Deepgram (cloud speech recognition).

Uses Deepgram's WebSocket streaming API for real-time transcription.
No SDK required — just websockets for real-time bidirectional communication.
Uses DEEPGRAM_API_KEY from environment.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import struct
import urllib.request
import wave
from typing import AsyncIterator

import websockets

from ... import config

log = logging.getLogger("alice.voice.stt.deepgram")


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Wrap raw 16-bit PCM data in a WAV header."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(pcm_data)
    return buf.getvalue()


class DeepgramSTT:
    """Speech-to-text using Deepgram's REST API.

    No SDK required. Uses urllib HTTP requests with the API key.
    Audio data is expected as raw 16-bit PCM (16kHz mono).
    The provider wraps it in a WAV header before sending to Deepgram.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or config.DEEPGRAM_API_KEY
        self._listening = False
        self.detected_language: str | None = None

        if self.api_key:
            log.info("deepgram_client_ready")
        else:
            log.warning("deepgram_no_api_key")

    async def start(self) -> None:
        self._listening = True
        log.info("stt_deepgram_started")

    async def stop(self) -> None:
        self._listening = False
        log.info("stt_deepgram_stopped")

    async def transcribe(self, audio: bytes) -> str:
        """Transcribe audio bytes via Deepgram REST API.

        Accepts raw 16-bit PCM audio (16kHz, mono) and wraps it
        in a WAV header before sending to the API.
        """
        if not self.api_key:
            return ""

        if len(audio) == 0:
            return ""

        try:
            # Convert audio to raw bytes
            if hasattr(audio, "tobytes"):
                raw_audio = audio.tobytes()
            elif isinstance(audio, (bytes, bytearray)):
                raw_audio = bytes(audio)
            else:
                raw_audio = bytes(audio)

            # Wrap in WAV header for Deepgram API
            wav_data = _pcm_to_wav(
                raw_audio,
                sample_rate=config.AUDIO_SAMPLE_RATE,
                channels=config.AUDIO_CHANNELS,
            )

            # Deepgram API endpoint
            url = "https://api.deepgram.com/v1/listen"
            params = "?model=nova-2&smart_format=true&punctuate=true"

            if config.STT_DETECT_LANGUAGE:
                params += "&detect_language=true"
            elif config.STT_LANGUAGE:
                params += f"&language={config.STT_LANGUAGE}"

            # Hint Deepgram toward Alice/French context to reduce misreads like "Adi"
            params += "&keyterms=Alice"

            request = urllib.request.Request(
                url + params,
                data=wav_data,
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "audio/wav",
                    "Accept": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))

            # Extract transcript
            results = result.get("results", {})
            channels = results.get("channels", [])
            
            # Store detected language for downstream TTS voice selection
            detected = results.get("detected_language")
            if detected:
                self.detected_language = detected
                log.info("language_detected", extra={"language": detected})
                if not config.TTS_LANGUAGE:
                    config._tts_language = detected
            
            if channels:
                alternatives = channels[0].get("alternatives", [])
                if alternatives:
                    transcript = alternatives[0].get("transcript", "")
                    confidence = alternatives[0].get("confidence", 1.0)
                    
                    # Filter only very low-confidence transcripts
                    if confidence < 0.3:
                        log.warning("stt_low_confidence", extra={"confidence": confidence, "transcript": transcript[:100]})
                        return ""
                    
                    cleaned = transcript.strip()
                    # Ignore obviously broken transcripts, but be less aggressive
                    if len(cleaned) < 2:
                        return ""
                    # Only reject if multiple trailing junk chars like "]"
                    if cleaned.endswith("]") and cleaned.count("]") > 1:
                        return ""
                    
                    return cleaned

            return ""

        except Exception as e:
            log.error("deepgram_transcribe_error", extra={"error": str(e), "type": type(e).__name__})
            return ""

    async def stream(self, audio_chunk: bytes) -> AsyncIterator[str]:
        """Stream transcription using Deepgram WebSocket API.

        Connects to Deepgram WebSocket, sends audio chunks as binary data,
        and yields partial transcriptions in real-time.

        This is the primary method for real-time streaming transcription.
        """
        if not self.api_key:
            return

        try:
            # Build WebSocket URI with parameters
            language = config.STT_LANGUAGE or "en-US"
            detect_param = "&detect_language=true" if config.STT_DETECT_LANGUAGE else ""
            uri = f"wss://api.deepgram.com/v1/listen?model=nova-2{detect_param}&smart_format=true&punctuate=true&language={language}&keyterms=Alice"

            async with websockets.connect(
                uri,
                extra_headers={"Authorization": f"Token {self.api_key}"},
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            ) as websocket:
                # Send audio chunk as binary data
                await websocket.send(audio_chunk)
                
                # Signal end of stream
                await websocket.send(json.dumps({
                    "type": "CloseStream",
                    "reason": "completion"
                }))

                # Receive and yield transcriptions
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type")
                        
                        if msg_type == "Results":
                            channel = data.get("channel", {})
                            alternatives = channel.get("alternatives", [])
                            if alternatives:
                                transcript = alternatives[0].get("transcript", "")
                                confidence = alternatives[0].get("confidence", 1.0)
                                is_final = data.get("is_final", True)
                                
                                # Store detected language
                                detected = data.get("detected_language")
                                if detected and self.detected_language != detected:
                                    self.detected_language = detected
                                    log.info("language_detected", extra={"language": detected})
                                    if not config.TTS_LANGUAGE:
                                        config._tts_language = detected
                                
                                # Yield transcript if it's final or has content
                                if transcript and confidence >= 0.3:
                                    cleaned = transcript.strip()
                                    if len(cleaned) >= 2:
                                        if not (cleaned.endswith("]") and cleaned.count("]") > 1):
                                            yield cleaned
                        
                        elif msg_type == "Error":
                            log.error("deepgram_ws_error", extra={"error": data.get("error", {}).get("message", "")})
                            break
                        
                        elif msg_type == "Close":
                            log.info("deepgram_stream_closed")
                            break
                            
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        log.error("deepgram_message_error", extra={"error": str(e)})
                        break
                        
        except websockets.exceptions.ConnectionClosed as e:
            log.info("deepgram_connection_closed", extra={"error": str(e)})
        except Exception as e:
            log.error("deepgram_stream_error", extra={"error": str(e), "type": type(e).__name__})