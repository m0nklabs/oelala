"""Behavioral tests for the admin endpoints (src/backend/admin_api.py).

The endpoints behind the (separately tested) auth gate are the only path an
admin has to adjust credits, change tiers, suspend users or flip admin flags.
They are thin wrappers over Supabase RPC calls with a common error contract:
500 on a failing call, 400 when the RPC reports success:false. These tests
drive them through a real route with the admin client mocked — no Supabase,
no network.
"""
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import admin_api
from auth import User

ADMIN = User(id="admin-1", email="a@_.test.invalid", role="user")


@pytest.fixture()
def app(monkeypatch):
    """Gate admits everyone (bypass); the admin client is mocked per test."""
    monkeypatch.setattr(admin_api, "ADMIN_BYPASS", True)
    a = FastAPI()
    a.include_router(admin_api.router)
    a.dependency_overrides[admin_api.get_current_user] = lambda: ADMIN
    return a


@pytest.fixture()
def client(app):
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# adjust_credits — the money endpoint
# ---------------------------------------------------------------------------


def test_adjust_rejects_zero_amount(client):
    r = client.post("/api/admin/credits/adjust",
                    json={"user_id": "u1", "amount": 0, "reason": "gift"})
    assert r.status_code == 422          # pydantic validator


def test_adjust_rejects_over_limit(client):
    r = client.post("/api/admin/credits/adjust",
                    json={"user_id": "u1", "amount": 100001, "reason": "gift"})
    assert r.status_code == 422


def test_adjust_rejects_short_reason(client):
    r = client.post("/api/admin/credits/adjust",
                    json={"user_id": "u1", "amount": 5, "reason": "x"})
    assert r.status_code == 422


def test_adjust_success_returns_new_balance(client, monkeypatch):
    c = Mock()
    resp = Mock(status_code=200, text="ok")
    resp.json = Mock(return_value=[{"success": True, "new_balance": 105}])
    c.post = AsyncMock(return_value=resp)
    monkeypatch.setattr(admin_api, "_get_admin_client", lambda: c)

    r = client.post("/api/admin/credits/adjust",
                    json={"user_id": "u1", "amount": 5, "reason": "gift"})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["new_balance"] == 105
    # the rpc must carry the admin id and the reason
    payload = c.post.call_args.kwargs["json"]
    assert payload["p_admin_id"] == "admin-1"
    assert payload["p_description"] == "gift"
    assert payload["p_amount"] == 5


def test_adjust_rpc_failure_is_500(client, monkeypatch):
    c = Mock()
    resp = Mock(status_code=502, text="bad gateway")
    c.post = AsyncMock(return_value=resp)
    monkeypatch.setattr(admin_api, "_get_admin_client", lambda: c)
    r = client.post("/api/admin/credits/adjust",
                    json={"user_id": "u1", "amount": 5, "reason": "gift"})
    assert r.status_code == 500


def test_adjust_rpc_success_false_is_400(client, monkeypatch):
    c = Mock()
    resp = Mock(status_code=200, text="ok")
    resp.json = Mock(return_value=[{"success": False, "error": "user not found"}])
    c.post = AsyncMock(return_value=resp)
    monkeypatch.setattr(admin_api, "_get_admin_client", lambda: c)
    r = client.post("/api/admin/credits/adjust",
                    json={"user_id": "u1", "amount": 5, "reason": "gift"})
    assert r.status_code == 400
    assert r.json()["detail"] == "user not found"


# ---------------------------------------------------------------------------
# update_tier
# ---------------------------------------------------------------------------


def test_tier_rejects_invalid_value(client):
    r = client.post("/api/admin/tier/update",
                    json={"user_id": "u1", "tier": "enterprise"})
    assert r.status_code == 400


def test_tier_success(client, monkeypatch):
    c = Mock()
    resp = Mock(status_code=200, text="ok")
    resp.json = Mock(return_value=[{"success": True}])
    c.post = AsyncMock(return_value=resp)
    monkeypatch.setattr(admin_api, "_get_admin_client", lambda: c)
    r = client.post("/api/admin/tier/update",
                    json={"user_id": "u1", "tier": "pro"})
    assert r.status_code == 200
    assert c.post.call_args.kwargs["json"]["p_tier"] == "pro"


# ---------------------------------------------------------------------------
# toggle_status / toggle_suspension
# ---------------------------------------------------------------------------


def test_toggle_status_sends_both_flags(client, monkeypatch):
    c = Mock()
    resp = Mock(status_code=200, text="ok")
    resp.json = Mock(return_value=[{"success": True}])
    c.post = AsyncMock(return_value=resp)
    monkeypatch.setattr(admin_api, "_get_admin_client", lambda: c)
    r = client.post("/api/admin/status/toggle",
                    json={"user_id": "u1", "is_admin": True, "is_vip": False})
    assert r.status_code == 200
    payload = c.post.call_args.kwargs["json"]
    assert payload["p_is_admin"] is True
    assert payload["p_is_vip"] is False


def test_toggle_suspension_success(client, monkeypatch):
    c = Mock()
    resp = Mock(status_code=200, text="ok")
    resp.json = Mock(return_value=[{"success": True}])
    c.post = AsyncMock(return_value=resp)
    monkeypatch.setattr(admin_api, "_get_admin_client", lambda: c)
    r = client.post("/api/admin/suspension/toggle",
                    json={"user_id": "u1", "is_suspended": True, "reason": "abuse"})
    assert r.status_code == 200
    # PR-Piet (2026-11-26): assert the rpc payload, verified against
    # admin_api.toggle_suspension (line ~536): p_user_id, p_is_suspended, p_reason.
    payload = c.post.call_args.kwargs["json"]
    assert payload["p_user_id"] == "u1"
    assert payload["p_is_suspended"] is True
    assert payload["p_reason"] == "abuse"


# ---------------------------------------------------------------------------
# list_users — paging contract
# ---------------------------------------------------------------------------


def fake_list_client():
    """A mock admin client with the attributes list_users actually touches."""
    c = Mock()
    c.headers = {"apikey": "test"}            # list_users spreads it into headers
    resp = Mock(status_code=200, text="ok")
    resp.json = Mock(return_value=[])
    resp.headers = {"Content-Range": "0-0/0"}
    c.get = AsyncMock(return_value=resp)
    return c


def test_list_users_paging_params(client, monkeypatch):
    c = fake_list_client()
    monkeypatch.setattr(admin_api, "_get_admin_client", lambda: c)

    r = client.get("/api/admin/users", params={"page": 3, "per_page": 20})
    assert r.status_code == 200
    params = c.get.call_args.kwargs["params"]
    assert params["offset"] == 40        # (3-1) * 20
    assert params["limit"] == 20


def test_list_users_rejects_per_page_over_100(client):
    r = client.get("/api/admin/users", params={"per_page": 500})
    assert r.status_code == 422          # Query(le=100)


def test_list_users_tier_filter(client, monkeypatch):
    c = fake_list_client()
    monkeypatch.setattr(admin_api, "_get_admin_client", lambda: c)
    client.get("/api/admin/users", params={"tier": "pro"})
    params = c.get.call_args.kwargs["params"]
    assert params["tier"] == "eq.pro"


# ---------------------------------------------------------------------------
# check endpoint
# ---------------------------------------------------------------------------


def test_check_reports_bypass_as_admin(client):
    r = client.get("/api/admin/check")
    assert r.status_code == 200
    assert r.json() == {"is_admin": True}     # bypass is on in this fixture


def test_check_follows_the_gate_when_bypass_is_off(client, monkeypatch):
    # PR-Piet (2026-11-26): with bypass off, /check must consult the gate, not
    # answer True unconditionally. The gate's Supabase lookup is mocked here;
    # the gate's own behavior lives in test_admin_auth_gate.py.
    monkeypatch.setattr(admin_api, "ADMIN_BYPASS", False)
    monkeypatch.setattr(admin_api, "SUPABASE_SERVICE_KEY", "service-key")
    monkeypatch.setattr(admin_api, "_admin_cache", {})
    c = Mock()
    resp = Mock(status_code=200)
    resp.json = Mock(return_value=[{"is_admin": False}])
    c.get = AsyncMock(return_value=resp)
    monkeypatch.setattr(admin_api, "_get_admin_client", lambda: c)

    r = client.get("/api/admin/check")
    assert r.status_code == 200
    assert r.json() == {"is_admin": False}    # gate consulted and obeyed
    # PR-Piet (2026-11-26, deny-side): the gate lookup must actually happen for
    # this user - an unconditional {"is_admin": false} would otherwise also pass.
    assert c.get.await_count == 1
    params = c.get.call_args.kwargs["params"]
    assert params["user_id"] == f"eq.{ADMIN.id}"
