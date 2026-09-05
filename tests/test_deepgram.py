"""Tests for Deepgram STT provider."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from websockets.exceptions import ConnectionClosed

from alice.voice.stt.deepgram import DeepgramSTT
from alice import config


class TestDeepgramSTT:
    """Test Deepgram speech-to-text provider."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        stt = DeepgramSTT(api_key="test-key")
        assert stt.api_key == "test-key"
        assert stt.detected_language is None
        assert stt._listening is False

    def test_init_without_api_key(self):
        """Test initialization without API key uses config default."""
        stt = DeepgramSTT(api_key=None)
        # When no api_key is passed, it falls back to config
        assert stt.api_key is not None or config.DEEPGRAM_API_KEY == ""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test start and stop methods."""
        stt = DeepgramSTT(api_key="test-key")
        assert stt._listening is False
        
        await stt.start()
        assert stt._listening is True
        
        await stt.stop()
        assert stt._listening is False

    @pytest.mark.asyncio
    async def test_transcribe_empty_audio(self):
        """Test transcribe with empty audio."""
        stt = DeepgramSTT(api_key="test-key")
        result = await stt.transcribe(b"")
        assert result == ""

    @pytest.mark.asyncio
    async def test_transcribe_no_api_key(self):
        """Test transcribe without API key."""
        stt = DeepgramSTT(api_key=None)
        result = await stt.transcribe(b"some audio")
        assert result == ""

    @pytest.mark.asyncio
    async def test_stream_no_api_key(self):
        """Test stream without API key."""
        stt = DeepgramSTT(api_key=None)
        results = []
        async for chunk in stt.stream(b"some audio"):
            results.append(chunk)
        assert results == []

    @pytest.mark.asyncio
    async def test_stream_success(self):
        """Test successful streaming transcription."""
        stt = DeepgramSTT(api_key="test-key")
        
        # Mock websocket
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        
        # Create an async iterator for the mock websocket
        messages = [
            json.dumps({
                "type": "Results",
                "is_final": True,
                "channel": {
                    "alternatives": [{
                        "transcript": "Hello world",
                        "confidence": 0.95
                    }]
                }
            }),
            json.dumps({
                "type": "Close"
            })
        ]
        
        async def mock_aiter(self):
            for msg in messages:
                yield msg
        
        mock_ws.__aiter__ = mock_aiter
        
        # Create mock context manager that returns the websocket
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        
        with patch("alice.voice.stt.deepgram.websockets.connect", return_value=mock_cm):
            results = []
            async for chunk in stt.stream(b"audio data"):
                results.append(chunk)
            
            assert len(results) == 1
            assert results[0] == "Hello world"
            
            # Verify websocket was used correctly
            mock_ws.send.assert_any_call(b"audio data")
            mock_ws.send.assert_any_call(json.dumps({
                "type": "CloseStream",
                "reason": "completion"
            }))

    @pytest.mark.asyncio
    async def test_stream_low_confidence(self):
        """Test stream filters low confidence results."""
        stt = DeepgramSTT(api_key="test-key")
        
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        
        messages = [
            json.dumps({
                "type": "Results",
                "is_final": True,
                "channel": {
                    "alternatives": [{
                        "transcript": "hello",
                        "confidence": 0.2  # Too low
                    }]
                }
            }),
            json.dumps({
                "type": "Close"
            })
        ]
        
        async def mock_aiter(self):
            for msg in messages:
                yield msg
        
        mock_ws.__aiter__ = mock_aiter
        
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        
        with patch("alice.voice.stt.deepgram.websockets.connect", return_value=mock_cm):
            results = []
            async for chunk in stt.stream(b"audio data"):
                results.append(chunk)
            
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_stream_connection_closed(self):
        """Test stream handles connection closed gracefully."""
        stt = DeepgramSTT(api_key="test-key")
        
        # Create a proper ConnectionClosed exception
        exc = ConnectionClosed(rcvd=1000, sent=1000)
        exc.rcvd_then_sent = True  # Avoid assertion error in __str__
        
        with patch("alice.voice.stt.deepgram.websockets.connect", side_effect=exc):
            results = []
            async for chunk in stt.stream(b"audio data"):
                results.append(chunk)
            
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_stream_detects_language(self):
        """Test stream detects and stores language."""
        stt = DeepgramSTT(api_key="test-key")
        
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        
        messages = [
            json.dumps({
                "type": "Results",
                "is_final": True,
                "detected_language": "fr",
                "channel": {
                    "alternatives": [{
                        "transcript": "Bonjour",
                        "confidence": 0.95
                    }]
                }
            }),
            json.dumps({
                "type": "Close"
            })
        ]
        
        async def mock_aiter(self):
            for msg in messages:
                yield msg
        
        mock_ws.__aiter__ = mock_aiter
        
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        
        with patch("alice.voice.stt.deepgram.websockets.connect", return_value=mock_cm):
            results = []
            async for chunk in stt.stream(b"audio data"):
                results.append(chunk)
            
            assert len(results) == 1
            assert results[0] == "Bonjour"
            assert stt.detected_language == "fr"