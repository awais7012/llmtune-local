"""Local, on-device record of who signs in to llmtune.

The only purpose of auth is so the product owner can keep a record of which
accounts use llmtune. That record stays on the user's own machine — nothing is
sent anywhere — at ~/.llmtune/logins.jsonl (one JSON object per line).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

_LOG_PATH = Path.home() / ".llmtune" / "logins.jsonl"


def record_login(user_id: str, email: str | None) -> None:
    """Append a single login event. Never raises — recording is best-effort."""
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "user_id": user_id,
            "email": email or "",
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def list_logins() -> list[dict]:
    """Return all recorded logins, newest first. Empty if none/unreadable."""
    if not _LOG_PATH.is_file():
        return []
    out: list[dict] = []
    try:
        for line in _LOG_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except Exception:
        return []
    return list(reversed(out))
