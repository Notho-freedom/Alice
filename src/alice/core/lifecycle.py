"""Application lifecycle and shared types."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("alice.core.lifecycle")


@dataclass
class AppConfig:
    """Application configuration passed to the lifecycle manager."""
    directory: str = "."
    kilo_base_url: str = "http://127.0.0.1:4096"
    auto_approve: bool = False
    voice_enabled: bool = False


@dataclass
class AppContext:
    """Shared context passed between components."""
    config: AppConfig
    session_id: str | None = None
    running: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class Lifecycle:
    """Manages application startup and shutdown."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.context = AppContext(config=config)

    def startup(self):
        log.info("app_startup")
        self.context.running = True

    def shutdown(self):
        log.info("app_shutdown")
        self.context.running = False
