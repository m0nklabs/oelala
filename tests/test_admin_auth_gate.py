"""Behavioral tests for the admin auth gate (src/backend/admin_api.py).

check_admin and get_admin_user are the only thing between any authenticated
user and every admin endpoint (user lists, credit adjustments, tier updates,
suspensions). The existing admin test file asserts documentation; these tests
drive the gate itself through a real route.
"""
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

import admin_api
from auth import User


@pytest.fixture()
def gate_env(monkeypatch):
    """Fresh gate: bypass off, service key present, empty cache."""
    monkeypatch.setattr(admin_api, "ADMIN_BYPASS", False)
    monkeypatch.setattr(admin_api, "SUPABASE_SERVICE_KEY", "service-key")
    monkeypatch.setattr(admin_api, "_admin_cache", {})
    client = Mock()
    monkeypatch.setattr(admin_api, "_get_admin_client", lambda: client)
    yield client


def make_app():
    app = FastAPI()

    @app.get("/probe", response_model=dict)
    async def probe(user: User = Depends(admin_api.get_admin_user)):
        return {"user": user.id}

    return app


def call_gate(monkeypatch, gate_env, user_id="user-1"):
    user = User(id=user_id, email="u@_.test.invalid", role="user")
    app = make_app()
    # get_admin_user resolves get_current_user from admin_api's namespace, so
    # the override must be registered on the app against that exact function.
    app.dependency_overrides[admin_api.get_current_user] = lambda: user
    return TestClient(app)


def test_bypass_grants_everyone_when_enabled(monkeypatch, gate_env):
    monkeypatch.setattr(admin_api, "ADMIN_BYPASS", True)
    r = call_gate(monkeypatch, gate_env).get("/probe")
    assert r.status_code == 200
    assert r.json() == {"user": "user-1"}
    gate_env.get.assert_not_called()          # bypass skips the DB entirely


def test_no_service_key_denies_without_a_db_call(monkeypatch, gate_env):
    monkeypatch.setattr(admin_api, "SUPABASE_SERVICE_KEY", "")
    r = call_gate(monkeypatch, gate_env).get("/probe")
    assert r.status_code == 403
    gate_env.get.assert_not_called()


def test_non_admin_row_is_denied(monkeypatch, gate_env):
    resp = Mock(status_code=200, json=Mock(return_value=[{"is_admin": False}]))
    gate_env.get = AsyncMock(return_value=resp)
    assert call_gate(monkeypatch, gate_env).get("/probe").status_code == 403


def test_admin_row_is_admitted(monkeypatch, gate_env):
    resp = Mock(status_code=200, json=Mock(return_value=[{"is_admin": True}]))
    gate_env.get = AsyncMock(return_value=resp)
    r = call_gate(monkeypatch, gate_env).get("/probe")
    assert r.status_code == 200


def test_result_is_cached_for_second_call(monkeypatch, gate_env):
    resp = Mock(status_code=200, json=Mock(return_value=[{"is_admin": True}]))
    gate_env.get = AsyncMock(return_value=resp)
    c = call_gate(monkeypatch, gate_env)
    assert c.get("/probe").status_code == 200
    assert c.get("/probe").status_code == 200
    assert gate_env.get.await_count == 1          # second call served from cache


def test_db_error_fails_closed(monkeypatch, gate_env):
    gate_env.get = AsyncMock(side_effect=RuntimeError("db down"))
    assert call_gate(monkeypatch, gate_env).get("/probe").status_code == 403
