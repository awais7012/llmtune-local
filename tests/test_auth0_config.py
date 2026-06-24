"""Unit tests for Auth0 config resolution and token-verification guards.

No network calls — these only check configuration logic and error handling.
"""

import importlib

import pytest


def _fresh(monkeypatch, **env):
    """Reload the auth0 module with a controlled environment."""
    for k in ("AUTH0_DOMAIN", "AUTH0_CLIENT_ID", "AUTH0_SCOPE", "AUTH0_AUDIENCE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import llmtune.server.auth.auth0 as a0
    return importlib.reload(a0)


def test_domain_strips_scheme_and_slash(monkeypatch):
    a0 = _fresh(monkeypatch, AUTH0_DOMAIN="https://my-tenant.us.auth0.com/")
    assert a0.auth0_domain() == "my-tenant.us.auth0.com"


def test_env_overrides_default_client_id(monkeypatch):
    a0 = _fresh(monkeypatch, AUTH0_CLIENT_ID="custom_client_id")
    assert a0.auth0_client_id() == "custom_client_id"


def test_missing_client_id_raises_clear_error(monkeypatch):
    a0 = _fresh(monkeypatch, AUTH0_DOMAIN="t.us.auth0.com", AUTH0_CLIENT_ID="")
    with pytest.raises(ValueError, match="Auth0 is not configured"):
        a0.request_device_code()


def test_verify_token_rejects_garbage(monkeypatch):
    a0 = _fresh(monkeypatch, AUTH0_DOMAIN="t.us.auth0.com", AUTH0_CLIENT_ID="cid")
    # No network: an obviously-malformed token must raise ValueError, not crash.
    with pytest.raises(ValueError):
        a0.verify_token("not.a.jwt")
