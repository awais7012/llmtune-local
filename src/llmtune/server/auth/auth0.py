"""Auth0 Device Authorization Flow + ID-token verification.

The local server brokers Auth0's device flow so neither the CLI nor the browser
UI needs to embed an auth SDK:

    1. /auth/device/start  → we ask Auth0 for a device + user code and hand back
       Auth0's hosted verification URL (the user signs in there).
    2. /auth/device/poll   → we exchange the device code at Auth0's token endpoint
       until the user finishes, then verify and return the ID token.

Only the tenant **domain** and a **Native application Client ID** are needed —
both are public values (the client ID is not a secret for native/device apps),
so they can be baked in like a publishable key.
"""

from __future__ import annotations

import os

import httpx
from jose import JWTError, jwt

# ── Configuration ───────────────────────────────────────────────────────────
# Override any of these with environment variables. Defaults are baked in so a
# pip-installed copy works out of the box; swap them for your own tenant.
DEFAULT_DOMAIN = "dev-ljg0w3p0ceo6ovzh.us.auth0.com"

# Client ID of an Auth0 **Native** application with the "Device Code" grant
# enabled. Public value — safe to ship (like a publishable key). Override with
# AUTH0_CLIENT_ID for a different tenant/app.
DEFAULT_CLIENT_ID = "xeC7LgYLC6tLJ2wd39Rxjm4dsVc4Wpj4"

DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"

# Auth0 returns these two token-endpoint errors while the user hasn't finished
# yet — they mean "keep polling", not "failed".
_PENDING_ERRORS = {"authorization_pending", "slow_down"}

_JWKS_CACHE: dict | None = None


def auth0_domain() -> str:
    return os.getenv("AUTH0_DOMAIN", DEFAULT_DOMAIN).strip().removeprefix("https://").removeprefix("http://").rstrip("/")


def auth0_client_id() -> str:
    return os.getenv("AUTH0_CLIENT_ID", DEFAULT_CLIENT_ID).strip()


def _auth0_scope() -> str:
    return os.getenv("AUTH0_SCOPE", "openid profile email").strip()


def _auth0_audience() -> str:
    # Optional — only set if you want access tokens scoped to a custom API.
    return os.getenv("AUTH0_AUDIENCE", "").strip()


def _require_config() -> tuple[str, str]:
    domain, client_id = auth0_domain(), auth0_client_id()
    if not domain or not client_id:
        raise ValueError(
            "Auth0 is not configured. Set AUTH0_DOMAIN and AUTH0_CLIENT_ID "
            "(create a Native app in the Auth0 dashboard and enable the "
            "Device Code grant)."
        )
    return domain, client_id


# ── Device flow ───────────────────────────────────────────────────────────--
def request_device_code() -> dict:
    """Ask Auth0 to start a device authorization. Returns Auth0's JSON response
    (device_code, user_code, verification_uri, verification_uri_complete,
    expires_in, interval)."""
    domain, client_id = _require_config()
    data = {"client_id": client_id, "scope": _auth0_scope()}
    audience = _auth0_audience()
    if audience:
        data["audience"] = audience

    resp = httpx.post(f"https://{domain}/oauth/device/code", data=data, timeout=15)
    if resp.status_code != 200:
        raise ValueError(f"Auth0 device/code failed: {resp.text}")
    return resp.json()


def exchange_device_code(device_code: str) -> dict | None:
    """Poll Auth0's token endpoint once.

    Returns the token response dict when the user has approved, ``None`` while
    still pending, and raises ``ValueError`` on a terminal error (expired code,
    access denied, …).
    """
    domain, client_id = _require_config()
    resp = httpx.post(
        f"https://{domain}/oauth/token",
        data={"grant_type": DEVICE_GRANT, "device_code": device_code, "client_id": client_id},
        timeout=15,
    )

    if resp.status_code == 200:
        return resp.json()

    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    error = body.get("error", "")
    if error in _PENDING_ERRORS:
        return None
    raise ValueError(body.get("error_description") or f"Auth0 token error: {resp.text}")


# ── Token verification ───────────────────────────────────────────────────────
def _get_jwks() -> dict:
    global _JWKS_CACHE
    if _JWKS_CACHE:
        return _JWKS_CACHE
    domain = auth0_domain()
    if not domain:
        raise ValueError("Auth0 is not configured. Set AUTH0_DOMAIN.")
    resp = httpx.get(f"https://{domain}/.well-known/jwks.json", timeout=10)
    resp.raise_for_status()
    _JWKS_CACHE = resp.json()
    return _JWKS_CACHE


def verify_token(token: str) -> dict:
    """Verify an Auth0 ID token (RS256) against the tenant JWKS, audience and
    issuer. Returns the decoded claims (includes ``sub`` and, with the email
    scope, ``email``)."""
    domain, client_id = _require_config()
    try:
        jwks = _get_jwks()
    except Exception as e:
        raise ValueError(f"Could not fetch Auth0 JWKS: {e}") from e

    try:
        header = jwt.get_unverified_header(token)
        key = next((k for k in jwks["keys"] if k.get("kid") == header.get("kid")), None)
        if key is None:
            raise ValueError("No matching key found in JWKS")
        return jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=client_id,
            issuer=f"https://{domain}/",
        )
    except JWTError as e:
        raise ValueError(f"Invalid Auth0 token: {e}") from e
