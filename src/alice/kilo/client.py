"""HTTP client for Kilo serve: REST API + SSE event streaming.

Uses Kilo's headless server (`kilo serve`) as the sole integration surface.
This is the thin bridge between Alice and the Kilo brain.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from typing import Any

import aiohttp
import requests

from .. import config
from .protocol import KiloEvent, parse_event_line

log = logging.getLogger("alice.kilo.client")


class KiloServerError(Exception):
    """Raised when the Kilo server is unreachable or returns an error."""


class KiloClient:
    """Thin HTTP client wrapping Kilo's REST API and SSE event stream."""

    def __init__(
        self,
        base_url: str = config.KILO_BASE_URL,
        directory: str = config.KILO_DIRECTORY,
    ):
        self.base_url = base_url.rstrip("/")
        self.directory = directory
        self._process: subprocess.Popen | None = None
        self._server_started = False
        self._sem = asyncio.Semaphore(1)

    # ── Server lifecycle ──────────────────────────────────────────────

    def health(self) -> bool:
        """Return True if the Kilo server is reachable."""
        try:
            r = requests.get(
                f"{self.base_url}/global/health",
                timeout=5,
            )
            return r.status_code == 200 and r.json().get("healthy") is True
        except Exception:
            return False

    def start_server(self) -> bool:
        """Start `kilo serve` if not already running. Returns True if ready."""
        if self.health():
            self._server_started = True
            return True

        log.info("kilo_server_start", extra={"server": self.base_url})
        cmd = ["kilo", "serve", "--port", str(config.KILO_PORT),
               "--hostname", config.KILO_HOST, "--print-logs"]
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.directory,
            shell=False,
        )
        self._server_started = True

        # Wait for server to become healthy with exponential backoff
        delay = 0.5
        for attempt in range(30):
            if self.health():
                log.info("kilo_server_ready")
                return True

            # Check if process crashed
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode("utf-8", errors="replace") if self._process.stderr else ""
                raise KiloServerError(
                    f"Kilo server exited with code {self._process.returncode}: {stderr[:500]}"
                )

            time.sleep(delay)
            delay = min(delay * 1.5, 5.0)

        raise KiloServerError("Kilo server did not become healthy within 15s")

    def stop_server(self):
        """Stop the Kilo server if we started it."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            self._server_started = False

    # ── Session management (v1 + v2 API) ──────────────────────────────

    def create_session(self, title: str | None = None) -> str:
        """Create a new Kilo session and return its ID."""
        params = {"directory": self.directory}
        body: dict[str, Any] = {}
        if title:
            body["title"] = title

        r = requests.post(
            f"{self.base_url}/session",
            params=params,
            json=body,
            timeout=30,
            headers={"Content-Type": "application/json"},
        )
        if r.status_code != 200:
            raise KiloServerError(f"Create session failed: {r.status_code} {r.text}")
        data = r.json()
        session_id = data.get("id") or data.get("data", {}).get("id")
        if not session_id:
            raise KiloServerError(f"No session ID in response: {data}")
        log.info("kilo_session_created", extra={"session_id": session_id})
        return session_id

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Get session details."""
        r = requests.get(
            f"{self.base_url}/api/session/{session_id}",
            params={"directory": self.directory},
            timeout=10,
        )
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        return r.json().get("data", r.json())

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions for the current directory."""
        r = requests.get(
            f"{self.base_url}/api/session",
            params={"directory": self.directory, "roots": True},
            timeout=10,
        )
        if r.status_code != 200:
            return []
        return r.json().get("data", [])

    def get_active_session(self) -> str | None:
        """Return the ID of the first active session, or None."""
        r = requests.get(f"{self.base_url}/api/session/active", timeout=5)
        if r.status_code != 200:
            return None
        data = r.json().get("data", {})
        for session_id in data:
            return session_id
        return None

    def delete_session(self, session_id: str):
        """Delete a session."""
        r = requests.delete(
            f"{self.base_url}/api/session/{session_id}",
            params={"directory": self.directory},
            timeout=10,
        )
        if r.status_code in (200, 204, 404):
            log.info("kilo_session_deleted", extra={"session_id": session_id})
        else:
            raise KiloServerError(f"Delete session failed: {r.status_code}")

    # ── Prompt / send ─────────────────────────────────────────────────

    def send_prompt(
        self,
        session_id: str,
        text: str,
        model: dict | None = None,
        agent: str | None = None,
    ) -> dict[str, Any]:
        """Send a prompt to a session. Returns the admission info."""
        params = {"directory": self.directory}
        body: dict[str, Any] = {"prompt": {"text": text}}
        if model:
            body["model"] = model
        if agent:
            body["agent"] = agent

        r = requests.post(
            f"{self.base_url}/api/session/{session_id}/prompt",
            params=params,
            json=body,
            timeout=30,
            headers={"Content-Type": "application/json"},
        )
        if r.status_code != 200:
            raise KiloServerError(f"Prompt failed: {r.status_code} {r.text}")
        return r.json().get("data", {})

    # ── Interrupt ────────────────────────────────────────────────────

    def interrupt(self, session_id: str):
        """Abort a running session."""
        r = requests.post(
            f"{self.base_url}/api/session/{session_id}/interrupt",
            params={"directory": self.directory},
            timeout=10,
        )
        if r.status_code in (200, 204):
            log.info("kilo_session_interrupted", extra={"session_id": session_id})
        else:
            log.warning(
                "kilo_interrupt_failed",
                extra={"session_id": session_id, "status": r.status_code},
            )

    # ── SSE event streaming ──────────────────────────────────────────

    async def subscribe_events(
        self,
        session_id: str | None = None,
    ) -> "KiloEventStream":
        """Subscribe to SSE events from the global event stream.

        If session_id is given, only events for that session are yielded.
        """
        url = f"{self.base_url}/global/event"
        return KiloEventStream(url, session_id_filter=session_id)


class KiloEventStream:
    """Async SSE stream consumer for Kilo events."""

    def __init__(self, url: str, session_id_filter: str | None = None):
        self.url = url
        self.session_id_filter = session_id_filter
        self._session: aiohttp.ClientSession | None = None
        self._queue: asyncio.Queue[KiloEvent | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._consume())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        if self._session and not self._session.closed:
            await self._session.close()

    async def _consume(self):
        headers = {"Accept": "text/event-stream", "Cache-Control": "no-cache"}
        try:
            self._session = aiohttp.ClientSession()
            async with self._session.get(self.url, headers=headers) as resp:
                async for line_bytes in resp.content:
                    if not self._running:
                        break
                    line = line_bytes.decode("utf-8").strip()
                    if not line:
                        continue
                    if not line.startswith("data: "):
                        continue
                    raw = parse_event_line(line)
                    if raw is None:
                        continue

                    payload = raw.get("payload", raw)
                    event = KiloEvent.from_payload(payload)
                    if event is None:
                        continue  # skip sync events

                    if (
                        self.session_id_filter
                        and event.session_id != self.session_id_filter
                    ):
                        continue

                    await self._queue.put(event)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error("kilo_stream_error", extra={"error": str(e)})
            await self._queue.put(None)

    async def next_event(self) -> KiloEvent | None:
        """Return the next event, or None if the stream ended."""
        return await self._queue.get()

    async def __aiter__(self):
        while True:
            event = await self.next_event()
            if event is None:
                break
            yield event
