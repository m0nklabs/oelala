"""Behavioral tests for the credits API endpoints (src/backend/credits_api.py).

The endpoints are the only path through which credits are bought, granted and
charged, so the suite pins the money behavior with a real FastAPI TestClient:
dependency overrides replace auth, get_credit_manager and Stripe are patched.
No Stripe keys, no network, no Supabase.
"""
import json
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import credits_api
from auth import User


@pytest.fixture()
def client(monkeypatch):
    app = FastAPI()
    app.include_router(credits_api.router)
    app.include_router(credits_api.stripe_router)

    user = User(id="user-1", email="u@_.test.invalid", role="user")
    app.dependency_overrides[credits_api.get_current_user] = lambda: user

    manager = Mock()
    monkeypatch.setattr(credits_api, "get_credit_manager", lambda: manager)
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_dummy")
    yield TestClient(app, raise_server_exceptions=False), manager
    app.dependency_overrides.clear()


def make_user_manager(balance=25):
    m = Mock()
    m.get_balance = Mock(return_value=Mock(balance=balance))
    return m


# ---------------------------------------------------------------------------
# get_balance
# ---------------------------------------------------------------------------


def test_balance_without_service_key_returns_welcome_credits(client):
    c, manager = client
    manager.service_key = ""
    r = c.get("/api/credits")
    assert r.status_code == 200
    assert r.json()["balance"] == 25


def test_balance_manager_error_maps_to_500(client):
    c, manager = client
    manager.service_key = "service"
    manager.get_balance = AsyncMock(side_effect=RuntimeError("db down"))
    assert c.get("/api/credits").status_code == 500


# ---------------------------------------------------------------------------
# get_packages
# ---------------------------------------------------------------------------


def test_packages_without_service_key_returns_defaults(client):
    c, manager = client
    manager.service_key = ""
    r = c.get("/api/credits/packages")
    assert r.status_code == 200
    assert len(r.json()) >= 1


def test_packages_manager_error_falls_back_to_defaults(client):
    c, manager = client
    manager.service_key = "service"
    manager.get_packages = AsyncMock(side_effect=RuntimeError("db down"))
    r = c.get("/api/credits/packages")
    assert r.status_code == 200
    assert len(r.json()) >= 1          # defaults, not an error


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------


def test_estimate_flags_hd_and_duration_and_sufficiency(client):
    c, manager = client
    manager.service_key = "service"
    manager.get_balance = AsyncMock(return_value=Mock(balance=3))

    r = c.post("/api/credits/estimate", json={
        "generation_type": "video_i2v",
        "width": 1920, "height": 1080,
        "duration_seconds": 12,
    })
    body = r.json()
    assert r.status_code == 200
    assert body["breakdown"]["hd_upcharge"] is True
    assert body["breakdown"]["duration_multiplier"] is True
    assert body["current_balance"] == 3
    # PR-Piet (2026-11-23): the flags alone would pass at any price. Pin the
    # exact calculation: MEDIUM (8) tier x 2.0 duration = 16.
    assert body["estimated_credits"] == 16
    assert body["sufficient"] is False          # 16 needed vs 3 held


def test_estimate_sufficient_when_balance_covers(client):
    c, manager = client
    manager.service_key = "service"
    manager.get_balance = AsyncMock(return_value=Mock(balance=100))
    r = c.post("/api/credits/estimate", json={"generation_type": "sdxl"})
    assert r.json()["sufficient"] is True


# ---------------------------------------------------------------------------
# get_transaction_history
# ---------------------------------------------------------------------------


def test_history_caps_limit_at_100(client):
    c, manager = client
    manager.service_key = "service"
    manager.get_transactions = AsyncMock(return_value=[])
    c.get("/api/credits/history", params={"limit": 500})
    args = manager.get_transactions.call_args
    assert args.args[1] == 100                  # capped


def test_history_error_returns_empty_list_not_500(client):
    c, manager = client
    manager.service_key = "service"
    manager.get_transactions = AsyncMock(side_effect=RuntimeError("db down"))
    r = c.get("/api/credits/history")
    assert r.status_code == 200
    assert r.json() == []


# ---------------------------------------------------------------------------
# initiate_purchase
# ---------------------------------------------------------------------------


def test_purchase_without_stripe_key_is_503(client, monkeypatch):
    c, _ = client
    monkeypatch.delenv("STRIPE_SECRET_KEY")
    r = c.post("/api/credits/purchase", json={"package_id": "starter"})
    assert r.status_code == 503


def test_purchase_unknown_package_is_404(client):
    c, manager = client
    manager.service_key = "service"
    manager.get_packages = AsyncMock(return_value=[Mock(id="pro", name="Pro", credits=100,
                                                   price_cents=900, currency="eur")])
    assert c.post("/api/credits/purchase", json={"package_id": "nope"}).status_code == 404


def test_purchase_happy_path_builds_checkout_session(client, monkeypatch):
    c, manager = client
    manager.service_key = "service"
    manager.get_packages = AsyncMock(return_value=[Mock(id="pro", name="Pro", credits=100,
                                                   price_cents=900, currency="eur")])

    created = {}

    def fake_create(**kwargs):
        created.update(kwargs)
        return Mock(id="cs_123", url="https://checkout.stripe.test/pay/cs_123")

    monkeypatch.setattr("stripe.checkout.Session.create", fake_create)

    r = c.post("/api/credits/purchase", json={"package_id": "pro"})
    body = r.json()
    assert r.status_code == 200
    assert body["session_id"] == "cs_123"
    assert body["checkout_url"].endswith("/pay/cs_123")
    # the success URL must carry the session placeholder for the return page
    assert "{CHECKOUT_SESSION_ID}" in created["success_url"]
    assert created["client_reference_id"] == "user-1"
    assert created["metadata"]["credits"] == "100"


def test_purchase_stripe_error_maps_to_500(client, monkeypatch):
    c, manager = client
    manager.service_key = "service"
    manager.get_packages = AsyncMock(return_value=[Mock(id="pro", name="Pro", credits=100,
                                                   price_cents=900, currency="eur")])
    monkeypatch.setattr("stripe.checkout.Session.create",
                        Mock(side_effect=__import__("stripe").error.StripeError("down")))
    assert c.post("/api/credits/purchase", json={"package_id": "pro"}).status_code == 500


# ---------------------------------------------------------------------------
# stripe_webhook — the money path
# ---------------------------------------------------------------------------


def webhook_payload(session_object):
    return {"type": "checkout.session.completed", "data": {"object": session_object}}


def post_webhook(client, monkeypatch, event, *, verify=True):
    if verify:
        monkeypatch.setattr("stripe.Webhook.construct_event", Mock(return_value=event))
    headers = {"stripe-signature": "t=1,v1=x"} if verify else {}
    return c_post(client, headers, event if not verify else None)


def c_post(client, headers, raw):
    if raw is None:
        return client.post("/api/stripe/webhook", headers=headers, content=b"{}")
    return client.post("/api/stripe/webhook", headers=headers, content=json.dumps(raw).encode())


def test_webhook_without_stripe_key_is_503(client, monkeypatch):
    c, _ = client
    monkeypatch.delenv("STRIPE_SECRET_KEY")
    assert c.post("/api/stripe/webhook", content=b"{}").status_code == 503


def test_webhook_bad_signature_is_400(client, monkeypatch):
    import stripe
    c, _ = client
    monkeypatch.setattr("stripe.Webhook.construct_event",
                        Mock(side_effect=stripe.error.SignatureVerificationError("bad", "sig")))
    r = c.post("/api/stripe/webhook", headers={"stripe-signature": "x"}, content=b"{}")
    assert r.status_code == 400


def test_webhook_completed_paid_event_credits_the_user(client, monkeypatch):
    c, manager = client
    manager.service_key = "service"
    manager.add_credits = AsyncMock()

    event = webhook_payload({
        "id": "cs_1", "payment_status": "paid",
        "client_reference_id": "user-9",
        "metadata": {"package_id": "pro", "credits": "100"},
        "payment_intent": "pi_1",
    })
    monkeypatch.setattr("stripe.Webhook.construct_event", Mock(return_value=event))

    r = c.post("/api/stripe/webhook",
               headers={"stripe-signature": "t=1,v1=x"},
               content=json.dumps({"ignored": True}).encode())

    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    # PR-Piet (2026-11-23): call_args alone would pass even if the webhook
    # called add_credits without awaiting it — assert the await itself.
    assert manager.add_credits.await_count == 1
    kwargs = manager.add_credits.await_args.kwargs
    assert kwargs["user_id"] == "user-9"
    assert kwargs["amount"] == 100
    assert kwargs["type"] == "purchase"
    assert kwargs["reference_id"] == "pi_1"


def test_webhook_unpaid_event_does_not_credit(client, monkeypatch):
    c, manager = client
    manager.service_key = "service"
    manager.add_credits = AsyncMock()

    event = webhook_payload({
        "id": "cs_1", "payment_status": "unpaid",
        "client_reference_id": "user-9",
        "metadata": {"package_id": "pro", "credits": "100"},
    })
    monkeypatch.setattr("stripe.Webhook.construct_event", Mock(return_value=event))

    r = c.post("/api/stripe/webhook",
               headers={"stripe-signature": "t=1,v1=x"}, content=b"{}")
    assert r.json()["status"] == "pending"
    manager.add_credits.assert_not_called()


def test_webhook_missing_user_id_is_an_error_not_a_credit(client, monkeypatch):
    # PR-Piet (2026-11-23): exercise each guard with the other field valid, so
    # an implementation that drops either check still fails a test.
    c, manager = client
    manager.service_key = "service"
    manager.add_credits = AsyncMock()

    event = webhook_payload({
        "id": "cs_1", "payment_status": "paid",
        "client_reference_id": None,
        "metadata": {"package_id": "pro", "credits": "100"},
    })
    monkeypatch.setattr("stripe.Webhook.construct_event", Mock(return_value=event))

    r = c.post("/api/stripe/webhook",
               headers={"stripe-signature": "t=1,v1=x"}, content=b"{}")
    assert r.json()["status"] == "error"
    manager.add_credits.assert_not_called()


def test_webhook_zero_credits_is_an_error_not_a_credit(client, monkeypatch):
    c, manager = client
    manager.service_key = "service"
    manager.add_credits = AsyncMock()

    event = webhook_payload({
        "id": "cs_1", "payment_status": "paid",
        "client_reference_id": "user-9",
        "metadata": {"package_id": "pro", "credits": "0"},
    })
    monkeypatch.setattr("stripe.Webhook.construct_event", Mock(return_value=event))

    r = c.post("/api/stripe/webhook",
               headers={"stripe-signature": "t=1,v1=x"}, content=b"{}")
    assert r.json()["status"] == "error"
    manager.add_credits.assert_not_called()


def test_webhook_add_credits_failure_is_500_so_stripe_retries(client, monkeypatch):
    c, manager = client
    manager.service_key = "service"
    manager.add_credits = AsyncMock(side_effect=RuntimeError("db down"))

    event = webhook_payload({
        "id": "cs_1", "payment_status": "paid",
        "client_reference_id": "user-9",
        "metadata": {"package_id": "pro", "credits": "100"},
        "payment_intent": "pi_1",
    })
    monkeypatch.setattr("stripe.Webhook.construct_event", Mock(return_value=event))

    r = c.post("/api/stripe/webhook",
               headers={"stripe-signature": "t=1,v1=x"}, content=b"{}")
    assert r.status_code == 500


# ---------------------------------------------------------------------------
# check_credits — the 402 envelope the frontend consumes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_credits_passes_when_sufficient(monkeypatch):
    manager = Mock()
    manager.get_balance = AsyncMock(return_value=Mock(balance=50))
    manager.get_packages = AsyncMock(return_value=[])
    # check_credits resolves get_credit_manager from credits_api's globals
    monkeypatch.setattr(credits_api, "get_credit_manager", lambda: manager)
    assert await credits_api.check_credits(User(id="u", email="e@_.test.invalid", role="user"), 10) is True


@pytest.mark.asyncio
async def test_check_credits_raises_402_with_the_frontend_envelope(monkeypatch):
    from fastapi import HTTPException
    manager = Mock()
    manager.get_balance = AsyncMock(return_value=Mock(balance=2))
    pkg = Mock(id="pro", name="Pro", credits=100, price_cents=900, currency="eur")
    pkg.dict = Mock(return_value={"id": "pro", "name": "Pro", "credits": 100,
                                  "price_cents": 900, "currency": "eur"})
    manager.get_packages = AsyncMock(return_value=[pkg])
    monkeypatch.setattr(credits_api, "get_credit_manager", lambda: manager)

    with pytest.raises(HTTPException) as exc:
        await credits_api.check_credits(User(id="u", email="e@_.test.invalid", role="user"), 10)
    # PR-Piet (2026-11-23): assert the status explicitly — a 500 with the same
    # envelope would otherwise satisfy the detail assertions and silently break
    # the frontend-facing 402 contract.
    assert exc.value.status_code == 402
    detail = exc.value.detail
    assert detail["error"] == "insufficient_credits"
    assert detail["required"] == 10
    assert detail["available"] == 2
    assert detail["packages"][0]["id"] == "pro"
