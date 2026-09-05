"""CLI entry point for Alice."""

from __future__ import annotations

import asyncio
import logging
import sys

import click

from .config import validate_environment, resolve_audio_devices
from .log_utils import setup_logging
from .terminal.interface import TerminalAssistant


@click.group()
@click.version_option()
def cli():
    """Alice - local voice assistant powered by Kilo Code."""


@cli.command()
@click.option("--ptt", is_flag=True, default=True, help="Use push-to-talk mode (default).")
@click.option("--continuous", "-c", is_flag=True, default=False, help="Continuous conversation mode with wake word.")
@click.option("--voice", "-v", default=None, help="Voice name or ID (e.g., 'marie', 'victoria').")
@click.option("--language", "-l", default=None, help="Language code (e.g., 'fr', 'en', 'es').")
@click.option("--list-voices", is_flag=True, default=False, help="List available voices and exit.")
def voice(ptt, continuous, voice, language, list_voices):
    """Start Alice in voice (audio) mode."""
    # Suppress verbose logging during voice mode
    logging.getLogger().setLevel(logging.WARNING)
    errors = validate_environment()
    for e in errors:
        print(f"  WARNING: {e}")

    from .voice.assistant import VoiceAssistant, VoiceConfig

    input_dev, output_dev = resolve_audio_devices()

    # List available voices if requested
    if list_voices:
        from .voice.tts import create_tts
        from .voice.tts.elevenlabs import FRENCH_FEMALE_VOICES
        tts = create_tts()
        if tts is None:
            print("  No TTS provider available")
            return
        
        print("\n  Available voices:")
        print("  -- ElevenLabs (French female) --")
        for name, vid in FRENCH_FEMALE_VOICES.items():
            print(f"    {name}: {vid}")
        
        if hasattr(tts, "providers"):
            print(f"\n  Provider chain: {' -> '.join(type(p).__name__ for p in tts.providers)}")
        else:
            print(f"\n  Provider: {type(tts).__name__}")
        return

    vc = VoiceConfig(
        ptt=ptt and not continuous,
        continuous=continuous,
        voice=voice,
        language=language,
        input_device=input_dev,
        output_device=output_dev,
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

        input_index, output_index = resolve_audio_devices()
        print(f"  Audio input device index: {input_index}")
        print(f"  Audio output device index: {output_index}")
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
        from .voice.tts import create_tts
        tts = create_tts()
        if tts is None:
            print("  TTS: NOT available")
            ok = False
        else:
            # List all providers in cascade
            providers = []
            if hasattr(tts, "providers"):
                providers = [type(p).__name__ for p in tts.providers]
            else:
                providers = [type(tts).__name__]
            provider_names = " -> ".join(providers)
            print(f"  TTS ({provider_names}): OK")
    except ImportError as e:
        print(f"  TTS: NOT available ({e})")
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
