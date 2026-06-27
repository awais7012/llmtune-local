"""Server configuration."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PORT = 8765


def get_port() -> int:
    return int(os.getenv("LLMTUNE_PORT", str(DEFAULT_PORT)))


def static_dir() -> Path | None:
    """Return the directory containing the built frontend, if it exists."""
    candidates = [
        Path(__file__).parent / "static",
        Path(__file__).resolve().parents[3] / "frontend" / "dist",
    ]
    for path in candidates:
        if (path / "index.html").exists():
            return path
    return None
