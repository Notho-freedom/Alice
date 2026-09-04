"""Terminal text interface for Alice.

Uses the Kilo bridge directly (spec section 18).
Provides a simple REPL:  > message
"""

from __future__ import annotations

import asyncio
import logging

from .. import config
from ..kilo.bridge import KiloBridge, BridgeConfig
from ..kilo.session import SessionManager
from ..kilo.stream import ChunkType
from ..kilo.client import KiloClient
from ..core.lifecycle import Lifecycle, AppConfig
from ..core.state_machine import StateMachine, Event

log = logging.getLogger("alice.terminal")


class TerminalAssistant:
    """Text-based assistant using the Kilo bridge."""

    def __init__(self):
        self._lifecycle = Lifecycle(AppConfig(
            directory=config.KILO_DIRECTORY,
            kilo_base_url=config.KILO_BASE_URL,
            auto_approve=config.KILO_AUTO_APPROVE,
        ))
        self.state_machine = StateMachine()
        self._bridge: KiloBridge | None = None
        self._sessions: SessionManager | None = None

    async def initialize(self):
        """Start Kilo server and create a session via the bridge."""
        self._lifecycle.startup()

        client = KiloClient()
        if not client.health():
            log.info("kilo_starting")
            client.start_server()

        self._sessions = SessionManager(client)
        session_id = self._sessions.get_or_create(title="Alice Terminal Session")
        self._lifecycle.context.session_id = session_id

        # Bridge uses the same client + session
        self._bridge = KiloBridge(
            client=client,
            config=BridgeConfig(
                auto_start_server=False,
                auto_resume_session=False,
                directory=config.KILO_DIRECTORY,
            ),
        )
        self._bridge._session_id = session_id
        self._bridge.state_machine = self.state_machine

        log.info("terminal_ready", extra={"session_id": session_id})

    async def send_message(self, text: str):
        """Send a text message to Kilo and stream the response."""
        if not self._bridge:
            print("Error: Not initialized. Call initialize() first.")
            return

        print(f"\nAssistant:", end="", flush=True)

        async for chunk in self._bridge.send_and_stream(text):
            if chunk.type == ChunkType.TEXT and chunk.text:
                print(chunk.text, end="", flush=True)
            elif chunk.type == ChunkType.TOOL_START:
                tool_name = chunk.data.get("tool", "")
                print(f"\n[Using tool: {tool_name}]", flush=True)
                print("Assistant:", end="", flush=True)
            elif chunk.type == ChunkType.STEP_END:
                reason = chunk.data.get("reason", "")
                tokens = chunk.data.get("tokens", {})
                total = tokens.get("total", sum(tokens.get(k, 0) for k in ("input", "output", "reasoning")))
                print(f"\n[took {reason}, tokens: {total}]", flush=True)
            elif chunk.type == ChunkType.ERROR:
                print(f"\n[ERROR: {chunk.text}]", flush=True)

        print()

    async def cleanup(self):
        """Clean up resources."""
        if self._bridge:
            pass  # Don't close session — Rule E preserves it
        if self._lifecycle:
            self._lifecycle.shutdown()

    def print_banner(self):
        print("\n" + "=" * 60)
        print("  Alice - Local Assistant (Terminal Mode)")
        print("  Powered by Kilo Code")
        print("=" * 60)
        if self._lifecycle.context.session_id:
            print(f"  Session: {self._lifecycle.context.session_id[:24]}...")
        print("  Type '/quit' or '/exit' to quit.")
        print("  Type '/session' to see sessions.")
        print("=" * 60 + "\n")

    async def run(self):
        """Main REPL loop."""
        await self.initialize()
        self.print_banner()

        while self._lifecycle.context.running:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, input, "> "
                )
            except EOFError:
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

            await self.send_message(user_input)

        await self.cleanup()

    async def _show_sessions(self):
        if not self._sessions:
            return
        sessions = self._sessions.list_all()
        print(f"\nSessions ({len(sessions)}):")
        for s in sessions:
            print(f"  {s['id'][:24]}... {s.get('title', '')}")

    async def _switch_session(self, sid: str):
        if not self._sessions:
            return
        if self._sessions.client.get_session(sid):
            self._lifecycle.context.session_id = sid
            self._bridge._session_id = sid
            self._sessions.touch(sid)
            print(f"Switched to session: {sid}")
        else:
            print(f"Session not found: {sid}")

    def _print_help(self):
        print("""
Alice Terminal Assistant - Commands:
  > message      Send a message to Kilo
  /session       List sessions
  /session <id>  Switch to a session
  /help          Show this help
  /quit          Exit
""")