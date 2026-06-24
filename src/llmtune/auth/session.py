"""Local session auth: starts the device flow on the local server, opens the
browser for sign-in, polls for the token, and stores it in the OS keyring.

Provider-agnostic — it only ever talks to the local llmtune server, which
brokers the actual identity provider (Auth0)."""

from __future__ import annotations

import json
import os
import time
import webbrowser
from pathlib import Path

import httpx
import keyring

SERVICE_NAME = "llmtune"
KEYRING_USER = "session"
TOKEN_FILE = Path.home() / ".llmtune" / "auth.json"

LLMTUNE_API_BASE = os.getenv("LLMTUNE_API_BASE", "http://127.0.0.1:8765")


def _save_token(token: str) -> None:
    try:
        keyring.set_password(SERVICE_NAME, KEYRING_USER, token)
    except Exception:
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(json.dumps({"token": token}))


def _load_token() -> str | None:
    try:
        token = keyring.get_password(SERVICE_NAME, KEYRING_USER)
        if token:
            return token
    except Exception:
        pass
    if TOKEN_FILE.exists():
        data = json.loads(TOKEN_FILE.read_text())
        return data.get("token")
    return None


def _delete_token() -> None:
    try:
        keyring.delete_password(SERVICE_NAME, KEYRING_USER)
    except Exception:
        pass
    if TOKEN_FILE.exists():
        TOKEN_FILE.unlink()


def is_authenticated() -> bool:
    token = _load_token()
    if not token:
        return False
    try:
        r = httpx.get(
            f"{LLMTUNE_API_BASE}/auth/verify",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5,
        )
        return r.status_code == 200
    except Exception:
        # Offline — trust the stored token.
        return True


def login(open_browser: bool = True) -> str:
    """Start device auth flow. Returns the session token on success."""
    r = httpx.post(f"{LLMTUNE_API_BASE}/auth/device/start", timeout=15)
    r.raise_for_status()
    data = r.json()
    login_url: str = data["login_url"]
    device_code: str = data["device_code"]
    interval: int = data.get("interval", 5)
    user_code: str = data.get("user_code", "")

    if user_code:
        print(f"\nTo sign in, confirm this code in your browser: {user_code}")
    print(f"Opening: {login_url}\n")

    if open_browser:
        webbrowser.open(login_url)

    # Poll until token arrives or timeout (5 min).
    deadline = time.time() + 300
    while time.time() < deadline:
        time.sleep(interval)
        poll = httpx.post(
            f"{LLMTUNE_API_BASE}/auth/device/poll",
            json={"device_code": device_code},
            timeout=15,
        )
        if poll.status_code == 200:
            token = poll.json()["token"]
            _save_token(token)
            return token
        if poll.status_code != 202:
            raise RuntimeError(f"Auth failed: {poll.text}")

    raise TimeoutError("Login timed out. Please try again.")


def logout() -> None:
    _delete_token()


def get_token() -> str | None:
    return _load_token()
