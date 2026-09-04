"""Kilo session lifecycle management.

Tracks the Kilo session ID across the conversation, supports
continuation, forking, and reconnection per the spec.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import config
from .client import KiloClient

log = logging.getLogger("alice.kilo.session")


@dataclass
class SessionState:
    """Local record of a Kilo session, persisted across restarts."""
    session_id: str
    title: str
    started_at: float
    updated_at: float
    directory: str
    active: bool = True


class SessionManager:
    """Manages Kilo sessions: create, resume, persist, list."""

    STATE_FILE = Path.home() / ".alice" / "sessions.json"

    def __init__(self, client: KiloClient):
        self.client = client
        self._sessions: dict[str, SessionState] = {}
        self._load_state()

    def _load_state(self):
        if self.STATE_FILE.exists():
            try:
                data = json.loads(self.STATE_FILE.read_text(encoding="utf-8"))
                for entry in data:
                    s = SessionState(**entry)
                    self._sessions[s.session_id] = s
                log.info("session_state_loaded",
                         extra={"count": len(self._sessions)})
            except Exception as e:
                log.warning("session_state_load_failed", extra={"error": str(e)})

    def _save_state(self):
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = [s.__dict__ for s in self._sessions.values()]
        self.STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_or_create(self, title: str = "Alice Voice Session") -> str:
        """Return an existing active session, or create a new one.

        Per the spec (Rule E), closing the voice interface does NOT
        destroy the Kilo session. We reuse the last session.
        """
        active = self.client.get_active_session()
        if active:
            log.info("session_resumed", extra={"session_id": active})
            return active

        # Try to reuse last session from our state file
        last = self._get_last_session()
        if last:
            # Validate it still exists
            if self.client.get_session(last):
                log.info("session_restored", extra={"session_id": last})
                return last

        # Create new session
        session_id = self.client.create_session(title=title)
        self._sessions[session_id] = SessionState(
            session_id=session_id,
            title=title,
            started_at=asyncio.get_event_loop().time(),
            updated_at=asyncio.get_event_loop().time(),
            directory=self.client.directory,
            active=True,
        )
        self._save_state()
        return session_id

    def _get_last_session(self) -> str | None:
        """Return the most recently used session ID from local state."""
        if not self._sessions:
            return None
        latest = max(self._sessions.values(), key=lambda s: s.updated_at)
        return latest.session_id

    def create_new(self, title: str = "Alice Voice Session") -> str:
        """Force-create a new session."""
        session_id = self.client.create_session(title=title)
        self._sessions[session_id] = SessionState(
            session_id=session_id,
            title=title,
            started_at=asyncio.get_event_loop().time(),
            updated_at=asyncio.get_event_loop().time(),
            directory=self.client.directory,
            active=True,
        )
        self._save_state()
        return session_id

    def touch(self, session_id: str):
        """Mark a session as recently used."""
        if session_id in self._sessions:
            self._sessions[session_id].updated_at = asyncio.get_event_loop().time()
            self._save_state()

    def list_all(self) -> list[dict[str, Any]]:
        """List sessions from Kilo (not local cache)."""
        return self.client.list_sessions()

    def close(self, session_id: str):
        """Close (delete) a Kilo session."""
        self.client.delete_session(session_id)
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._save_state()
