"""Tests for HTTP Basic auth middleware on the UI server."""

from __future__ import annotations

import base64
import importlib

import pytest
from fastapi.testclient import TestClient


def _make_client(username: str, password: str) -> TestClient:
    """Reimport server.py with auth env vars set."""

    import orinode.ui.server as srv_module

    # Patch the module-level auth variables
    orig_user = srv_module.UI_USERNAME
    orig_pass = srv_module.UI_PASSWORD
    orig_enabled = srv_module._AUTH_ENABLED

    srv_module.UI_USERNAME = username
    srv_module.UI_PASSWORD = password
    srv_module._AUTH_ENABLED = bool(username and password)

    try:
        client = TestClient(srv_module.app, raise_server_exceptions=False)
        yield client
    finally:
        srv_module.UI_USERNAME = orig_user
        srv_module.UI_PASSWORD = orig_pass
        srv_module._AUTH_ENABLED = orig_enabled


@pytest.fixture()
def auth_client(monkeypatch):
    """TestClient with ORINODE_UI_USER=orinode, ORINODE_UI_PASS=testpass."""
    monkeypatch.setenv("ORINODE_UI_USER", "orinode")
    monkeypatch.setenv("ORINODE_UI_PASS", "testpass")

    import importlib

    import orinode.ui.server as srv

    importlib.reload(srv)
    client = TestClient(srv.app, raise_server_exceptions=False)
    yield client

    # Reload with no auth so other tests are unaffected
    monkeypatch.delenv("ORINODE_UI_USER", raising=False)
    monkeypatch.delenv("ORINODE_UI_PASS", raising=False)
    importlib.reload(srv)


def _basic(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return f"Basic {token}"


# ── unauthenticated ────────────────────────────────────────────────────────────


def test_unauthenticated_request_returns_401(auth_client):
    resp = auth_client.get("/api/v1/stats")
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate", "").startswith("Basic")


# ── correct credentials ────────────────────────────────────────────────────────


def test_correct_credentials_returns_200(auth_client):
    resp = auth_client.get(
        "/api/v1/stats",
        headers={"Authorization": _basic("orinode", "testpass")},
    )
    assert resp.status_code == 200


# ── wrong credentials ─────────────────────────────────────────────────────────


def test_wrong_password_returns_401(auth_client):
    resp = auth_client.get(
        "/api/v1/stats",
        headers={"Authorization": _basic("orinode", "wrongpassword")},
    )
    assert resp.status_code == 401


def test_wrong_username_returns_401(auth_client):
    resp = auth_client.get(
        "/api/v1/stats",
        headers={"Authorization": _basic("hacker", "testpass")},
    )
    assert resp.status_code == 401


def test_empty_credentials_returns_401(auth_client):
    resp = auth_client.get(
        "/api/v1/stats",
        headers={"Authorization": _basic("", "")},
    )
    assert resp.status_code == 401


# ── no auth configured — open access ──────────────────────────────────────────


def test_no_auth_configured_allows_unauthenticated(monkeypatch):
    monkeypatch.delenv("ORINODE_UI_USER", raising=False)
    monkeypatch.delenv("ORINODE_UI_PASS", raising=False)

    import orinode.ui.server as srv

    importlib.reload(srv)
    client = TestClient(srv.app, raise_server_exceptions=False)

    resp = client.get("/api/v1/stats")
    assert resp.status_code == 200
