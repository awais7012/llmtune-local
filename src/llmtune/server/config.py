"""Server configuration."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PORT = 8765

# ---------------------------------------------------------------------------
# Load the frontend .env file so the server picks up AUTH0_DOMAIN /
# AUTH0_CLIENT_ID in development without requiring users to export them.
# (No effect for a pip-installed copy — the project root path won't exist.)
# ---------------------------------------------------------------------------

def _load_frontend_env() -> None:
    """
    Read key=value pairs from frontend/.env and set them as environment
    variables if they are not already set. This is a minimal dotenv loader
    so we don't need python-dotenv as a dependency.
    """
    here = Path(__file__).resolve()
    # Walk up from src/llmtune/server/config.py to the project root
    project_root = here.parents[3]  # .../fine-tuning
    env_file = project_root / "frontend" / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_frontend_env()


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
