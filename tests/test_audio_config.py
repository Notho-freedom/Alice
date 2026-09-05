"""Tests for audio device auto-detection."""

from __future__ import annotations

import pytest

from alice import config


class TestAudioDeviceResolution:
    def test_resolve_audio_devices_returns_two_values(self):
        input_index, output_index = config.resolve_audio_devices()
        assert isinstance(input_index, int)
        assert isinstance(output_index, int)

    def test_resolve_audio_devices_uses_valid_indices(self):
        input_index, output_index = config.resolve_audio_devices()
        assert input_index >= 0
        assert output_index >= 0
