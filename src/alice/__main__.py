"""CLI entry point for Alice."""

from __future__ import annotations

import asyncio
import logging
import sys

import click

from .config import validate_environment
from .log_utils import setup_logging
from .terminal.interface import TerminalAssistant


@click.group()
@click.version_option()
def cli():
    """Alice - local voice assistant powered by Kilo Code."""


@cli.command()
@click.option("--ptt", is_flag=True, default=True, help="Use push-to-talk mode (default).")
@click.option("--continuous", "-c", is_flag=True, default=False, help="Continuous conversation mode with wake word.")
def voice(ptt, continuous):
    """Start Alice in voice (audio) mode."""
    # Suppress verbose logging during voice mode
    logging.getLogger().setLevel(logging.WARNING)
    errors = validate_environment()
    for e in errors:
        print(f"  WARNING: {e}")

    from .voice.assistant import VoiceAssistant, VoiceConfig
    vc = VoiceConfig(
        ptt=ptt and not continuous,
        continuous=continuous,
    )
    assistant = VoiceAssistant(voice_config=vc)
    asyncio.run(assistant.initialize())
    if continuous:
        asyncio.run(assistant.run_continuous())
    else:
        asyncio.run(assistant.run_push_to_talk())


@cli.command()
def terminal():
    """Start Alice in terminal (text) mode."""
    # Only log warnings+ during terminal mode
    logging.getLogger("alice").setLevel(logging.WARNING)
    errors = validate_environment()
    for e in errors:
        print(f"  WARNING: {e}")
    assistant = TerminalAssistant()
    asyncio.run(assistant.run())


@cli.command()
def doctor():
    """Check that all components are available."""
    setup_logging()
    errors = validate_environment()
    ok = True
    if errors:
        ok = False
        for e in errors:
            print(f"  FAIL: {e}")
    else:
        print("  Kilo CLI: OK")

    # Check audio devices
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devs = [d for d in devices if d["max_input_channels"] > 0]
        output_devs = [d for d in devices if d["max_output_channels"] > 0]
        if input_devs:
            print(f"  Microphone: OK ({input_devs[0]['name']})")
        else:
            print("  Microphone: NOT FOUND")
            ok = False
        if output_devs:
            print(f"  Speakers: OK ({output_devs[0]['name']})")
        else:
            print("  Speakers: NOT FOUND")
            ok = False
    except Exception as e:
        print(f"  Audio: {e}")
        ok = False

    # Check STT
    from . import config as cfg
    if cfg.STT_PROVIDER == "deepgram" and cfg.DEEPGRAM_API_KEY:
        print("  STT (Deepgram): OK")
    elif cfg.STT_PROVIDER == "openai" and cfg.OPENAI_API_KEY:
        print("  STT (OpenAI): OK")
    else:
        print("  STT: NOT configured")
        ok = False

    # Check TTS
    try:
        if cfg.TTS_PROVIDER == "edge":
            import aiohttp
            print(f"  TTS (Edge backend @ {cfg.TTS_BACKEND_URL}): OK")
        else:
            import pyttsx3
            print("  TTS (pyttsx3): OK")
    except ImportError as e:
        print(f"  TTS: NOT available ({cfg.TTS_PROVIDER})")
        ok = False

    # Check Kilo server
    from .kilo.client import KiloClient
    client = KiloClient()
    if client.health():
        print("  Kilo server: running")
    else:
        print("  Kilo server: not running (will auto-start)")

    print(f"\n  Status: {'ALL OK' if ok else 'ISSUES FOUND'}")
    return 0 if ok else 1


if __name__ == "__main__":
    cli()
