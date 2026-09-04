# AGENTS.md

Local voice assistant ("Alice") that pilots Kilo Code as its brain.

## Architecture Summary (M0 Audit)

### Kilo Integration Strategy

Kilo Code is the **brain**. This application is the **sensory + control layer**.

Integration is via `kilo serve` (headless HTTP server) + HTTP REST API + SSE event stream.

**Primary integration surfaces (ranked by preference per spec):**

1. **`kilo serve`** — Headless HTTP server on `127.0.0.1:4096`.
   - `POST /session?directory=...` — create session (v1 API, no prompt needed)
   - `GET /api/session` — list sessions (v2 API)
   - `GET /api/session/{sessionID}` — get session info (v2)
   - `GET /api/session/active` — active sessions (v2)
   - `DELETE /api/session/{sessionID}?directory=...` — delete session (v2)
   - `POST /api/session/{sessionID}/interrupt?directory=...` — abort (v2, returns 204)

2. **SSE Event Stream** — `GET /global/event` (SSE, no auth needed locally)
   Events arrive as `{"directory": "...", "payload": {"id": "...", "type": "...", "properties": {...}}}`
   Native events (skip `sync` duplicates):
   - `session.next.prompted` — user prompt received
   - `session.next.step.started` — agent started (has agent, model info)
   - `session.next.step.ended` — step ended (has finish reason, cost, tokens)
   - `session.next.text.started` — text generation started
   - `session.next.text.delta` — incremental text (properties.delta)
   - `session.next.text.ended` — full text (properties.text)
   - `session.next.tool.started` — tool call started
   - `session.next.tool.ended` — tool result
   - `session.error` — error (properties.error)
   - `server.connected` / `server.heartbeat` — lifecycle
   - `sync` — duplicate normalized events, IGNORE these

3. `kilo acp` — ACP over ndjson (has reported issues with headless clients, not used)
4. `kilo run --format json` — one-shot subprocess (no persistent session, fallback only)

### Architecture Rules

**Rule 1 — No duplication of Kilo capabilities.**
Before implementing any feature in Alice, verify whether Kilo Code already
provides that capability (sessions, context, tools, MCP, tasks, permissions,
agent management, etc.). If Kilo provides it, use the Kilo API — do NOT
reimplement it in Alice.

Rationale: Kilo is the brain. Alice is the sensory + control layer.
Duplicating brain functionality leads to inconsistency and maintenance debt.

**Rule 2 — The bridge is the contract.**
All interaction with Kilo happens through `alice/kilo/bridge.py`.
This module defines the complete lifecycle contract:

```
create_session → send_prompt → stream_response → interrupt → continue → close
```

Tests for this contract live in `tests/test_bridge.py`.

**Rule 3 — Session persistence (Rule E).**
Interrupting speech ≠ closing a session.
The Kilo session persists across voice turns for context continuity.
Alice reuses the last session on restart (via `alice/sessions.json`).

### Project Structure

```
src/alice/
    core/      — state machine, events, lifecycle
    kilo/      — HTTP client, session, SSE stream, protocol/types
    voice/     — STT, TTS, VAD, wake word, input, interruption
    terminal/  — text interface (uses same bridge as voice)
    providers/ — concrete provider implementations
config.py      — environment configuration
log_utils.py   — structured JSON logging
__main__.py    — CLI entry point
```

### Milestone Progress

- M0: Audit complete
- M1: Kilo Bridge — complete
- STT: Deepgram with language auto-detection (multilingual)
- TTS: Provider cascade (ElevenLabs → VAPI → Edge → Pi TTS → pyttsx3)
- M2-M10: pending

### Environment

- Python 3.11.9 on Windows (win32)
- Kilo CLI 7.5.9
- Audio devices: Microphone Array (idx 1), Speakers (idx 3) via sounddevice
- Packages: aiohttp, numpy, scipy, sounddevice, webrtcvad, rich, pydantic, click, pyttsx3, openai, requests, soundfile
