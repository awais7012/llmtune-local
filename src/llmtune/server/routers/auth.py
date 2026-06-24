"""Auth routes — Auth0 device authorization flow, brokered by the local server.

The browser UI and the CLI both call /device/start then poll /device/poll. The
server proxies Auth0's device flow, so no auth SDK ships in the frontend and the
user signs in on Auth0's own hosted page.
"""

from __future__ import annotations

import os
import platform
import subprocess
import webbrowser

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llmtune.server.auth import (
    auth0_domain,
    exchange_device_code,
    list_logins,
    record_login,
    request_device_code,
    verify_token,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class DeviceStartResponse(BaseModel):
    device_code: str
    login_url: str
    interval: int = 5
    user_code: str = ""


class DevicePollRequest(BaseModel):
    device_code: str


class OpenBrowserRequest(BaseModel):
    url: str


def _open_system_browser(url: str) -> bool:
    """
    Open URL in the system's default browser, never inside pywebview.

    Python's webbrowser module is unreliable when we're running inside the
    pywebview Cocoa/GTK host process, so try the OS-native launcher first
    (macOS `open`, Windows `os.startfile`, Linux `xdg-open`) and fall back to
    webbrowser only if that fails. Returns True if a launcher was invoked
    without error — the user still has the manual link as a backup.
    """
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", url])
            return True
        if system == "Windows":
            os.startfile(url)  # type: ignore[attr-defined]
            return True
        if system == "Linux":
            subprocess.Popen(["xdg-open", url])
            return True
    except Exception:
        pass
    try:
        return webbrowser.open(url, new=2)
    except Exception:
        return False


@router.post("/device/start", response_model=DeviceStartResponse)
def device_start():
    try:
        data = request_device_code()
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    return DeviceStartResponse(
        device_code=data["device_code"],
        # Auth0's hosted sign-in page, with the user code pre-filled.
        login_url=data.get("verification_uri_complete") or data["verification_uri"],
        interval=data.get("interval", 5),
        user_code=data.get("user_code", ""),
    )


@router.post("/device/open-browser")
def device_open_browser(body: OpenBrowserRequest):
    # Allow the local server and our configured Auth0 tenant only.
    allowed = (
        body.url.startswith("http://127.0.0.1:")
        or body.url.startswith("http://localhost:")
        or body.url.startswith(f"https://{auth0_domain()}/")
    )
    if not allowed:
        raise HTTPException(status_code=400, detail="Invalid login URL")
    opened = _open_system_browser(body.url)
    # "ok" even if opened is False — the UI shows the manual link as a fallback.
    return {"status": "ok", "opened": opened}


@router.post("/device/poll")
def device_poll(body: DevicePollRequest):
    try:
        tokens = exchange_device_code(body.device_code)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if tokens is None:
        return JSONResponse(status_code=202, content={"status": "pending"})

    id_token = tokens.get("id_token")
    if not id_token:
        raise HTTPException(status_code=400, detail="Auth0 did not return an ID token.")

    try:
        claims = verify_token(id_token)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e

    # Keep a local, on-device record of who signed in.
    record_login(claims.get("sub", ""), claims.get("email"))

    return {"token": id_token}


@router.get("/logins")
def logins():
    """Return the local record of accounts that have signed in on this machine."""
    return {"logins": list_logins()}


@router.get("/verify")
def verify(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")

    token = auth.removeprefix("Bearer ").strip()
    try:
        claims = verify_token(token)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e

    return {"valid": True, "user_id": claims["sub"], "email": claims.get("email")}
