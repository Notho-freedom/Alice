"""Rich terminal interface for Alice.

Provides an animated, colorized terminal experience with:
- Animated logo banner on startup
- Color-coded user prompts and Kilo responses
- Tool call indicators with spinner
- Typing animation for streamed responses
- Status bars and progress indicators
- Syntax highlighting for code responses
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.markup import escape

from .. import config
from ..kilo.client import KiloServerError
from ..kilo.bridge import KiloBridge
from ..kilo.stream import ChunkType
from ..core.state_machine import StateMachine, Event
from ..core.lifecycle import Lifecycle, AppConfig

log = logging.getLogger("alice.terminal.rich")

# Console configured for Windows compatibility
_console = Console(
    force_terminal=True,
    color_system="auto",
    width=None,
    highlight=True,
    soft_wrap=False,
    stderr=False,
    legacy_windows=False,
)


def _print_logo():
    """Print animated Alice logo."""
    logo_lines = [
        "  +============================================================+",
        "  |                                                            |",
        ("  |         A    L    I    C    E                           |", "magenta"),
        ("  |          Local Voice Assistant                          |", "cyan"),
        ("  |          Powered by Kilo Code                         |", "cyan"),
        "  |                                                            |",
        "  +============================================================+",
    ]

    for item in logo_lines:
        if isinstance(item, tuple):
            line, color = item
            _console.print(line, style=color)
        else:
            _console.print(item, style="cyan")
        time.sleep(0.03)


class TerminalAssistant:
    """Rich text-based assistant using Kilo bridge."""

    def __init__(self):
        self._lifecycle = Lifecycle(AppConfig(
            directory=config.KILO_DIRECTORY,
            kilo_base_url=config.KILO_BASE_URL,
            auto_approve=config.KILO_AUTO_APPROVE,
        ))
        self.state_machine = StateMachine()
        self._bridge: KiloBridge | None = None
        self._session_id: str | None = None

    async def initialize(self):
        """Start Kilo server and create session."""
        self._lifecycle.startup()

        self._bridge = KiloBridge()
        await self._bridge.start()
        self._session_id = self._bridge.session_id
        self._lifecycle.context.session_id = self._session_id

    async def cleanup(self):
        """Clean up resources."""
        if self._bridge:
            await self._bridge.close()
        self._lifecycle.shutdown()

    def print_banner(self):
        """Print the startup banner."""
        _console.clear()
        _print_logo()

        session_display = self._session_id[:24] + "..." if self._session_id else "None"

        banner = Panel(
            Group(
                Text(f"Session: {session_display}", style="dim cyan"),
                Text("Commands: /quit  /session  /help", style="dim"),
            ),
            border_style="blue",
            padding=(0, 2),
            title="Alice Terminal",
            title_align="left",
        )
        _console.print(banner)
        _console.print()

    async def send_message(self, text: str):
        """Send a message and stream response with visual effects."""
        self.state_machine.fire(Event.PROMPT_SENT)  # IDLE -> THINKING

        print("  Kilo is thinking ", end="", flush=True)
        spinner = itertools.cycle(["|", "/", "-", "\\"])

        response_text = ""
        tokens = {}
        got_completion = False

        async for chunk in self._bridge.send_and_stream(text):
            if chunk.type.value == "text":
                response_text += chunk.text
                print(f"\r{chunk.text}", end="", flush=True)
            elif chunk.type.value == "completion":
                got_completion = True
            elif chunk.type.value == "tool_start":
                print(f"\n  [TOOL] {chunk.data.get('tool', '')}", flush=True)
            elif chunk.type.value == "step_end":
                tokens = chunk.data.get("tokens", {})
            elif chunk.type.value == "error":
                print(f"\n  [ERROR] {chunk.text}", flush=True)
                got_completion = True

        print("\r  " + " " * 20 + "\r", end="", flush=True)

        self.state_machine.fire(Event.THINKING_COMPLETED)
        self.state_machine.fire(Event.TTS_COMPLETED)

        if tokens:
            total = tokens.get("total", sum(tokens.get(k, 0) for k in ("input", "output", "reasoning")))
            print(f"  [dim]tokens: {total}[/dim]")

    async def run(self):
        """Main REPL loop."""
        await self.initialize()
        self.print_banner()

        while self._lifecycle.context.running:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, input, "  > "
                )
            except (EOFError, KeyboardInterrupt):
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.lower() in ("/quit", "/exit", "/q"):
                break
            elif user_input.lower() == "/session":
                await self._show_sessions()
                continue
            elif user_input.lower() == "/help":
                self._print_help()
                continue
            elif user_input.lower().startswith("/session "):
                await self._switch_session(user_input.split(None, 1)[1])
                continue

            # Print user message with color
            _console.print(f"\n  {escape(user_input)}", style="green")
            _console.print()

            try:
                await self.send_message(user_input)
            except KiloServerError as e:
                _console.print(f"  [red]Error: {escape(str(e))}[/red]")
            except Exception as e:
                log.error("terminal_error", extra={"error": str(e)})
                _console.print(f"  [red]Error: {escape(str(e))}[/red]")

            _console.print()

        _console.print("\n[dim]Goodbye![/dim]")
        await self.cleanup()

    async def _show_sessions(self):
        if not self._sessions:
            return
        sessions = self._sessions.list_all()
        _console.print(f"\n[cyan]Sessions ({len(sessions)})[/cyan]:")
        for s in sessions:
            _console.print(f"  {s['id'][:24]}... {s.get('title', '')}")
        _console.print()

    async def _switch_session(self, sid: str):
        if not self._sessions:
            return
        if self._sessions.client.get_session(sid):
            self._session_id = sid
            self._lifecycle.context.session_id = sid
            self._sessions.touch(sid)
            _console.print(f"[yellow]Switched to session: {sid}[/yellow]")
        else:
            _console.print(f"[red]Session not found: {sid}[/red]")

    def _print_help(self):
        help_text = """
[bold cyan]Alice Terminal Assistant[/bold cyan]
[green]Commands:[/green]
  > message    Send a message to Kilo
  /session     List sessions
  /session <id>  Switch to a session
  /help        Show this help
  /quit        Exit
"""
        _console.print(help_text)


def cli():
    """Entry point for the `alice` command."""
    import click
    from ..config import validate_environment
    from ..log_utils import setup_logging

    @click.command()
    def terminal():
        """Start Alice in terminal (text) mode."""
        logging.getLogger("alice").setLevel(logging.WARNING)
        errors = validate_environment()
        if errors:
            for e in errors:
                print(f"  WARNING: {e}")

        assistant = TerminalAssistant()
        asyncio.run(assistant.run())

    terminal()