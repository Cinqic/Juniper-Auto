"""Structured logging foundation.

Emits one JSON object per log line so future tooling can parse logs
mechanically. Deliberately minimal for Phase 0 -- this is not an
observability platform, just a consistent event shape:
timestamp, level, event, phase, run_id, experiment_id, git_commit, config_id,
architecture_id, seed, env_id, plus arbitrary extra fields.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any


def current_git_commit(short: bool = False) -> str:
    """Best-effort current git commit hash. Returns 'unknown' outside a git
    checkout rather than raising, since logging must not crash on this."""
    try:
        args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        out = subprocess.run(args, capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class LogContext:
    """Fields every structured log line should carry when known. Any field
    left as None is simply omitted from the emitted JSON."""

    phase: str | None = None
    run_id: str | None = None
    experiment_id: str | None = None
    git_commit: str | None = None
    config_id: str | None = None
    architecture_id: str | None = None
    seed: int | None = None
    env_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = {
            "phase": self.phase,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "git_commit": self.git_commit,
            "config_id": self.config_id,
            "architecture_id": self.architecture_id,
            "seed": self.seed,
            "env_id": self.env_id,
        }
        d = {k: v for k, v in d.items() if v is not None}
        d.update(self.extra)
        return d


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if context:
            payload.update(context)
        return json.dumps(payload, sort_keys=True, default=str)


def get_logger(name: str, *, stream=None) -> logging.Logger:
    """Return a stdlib logger configured to emit structured JSON lines.
    Safe to call repeatedly -- does not duplicate handlers."""
    logger = logging.getLogger(name)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, level: int, event: str, context: LogContext) -> None:
    logger.log(level, event, extra={"context": context.as_dict()})
