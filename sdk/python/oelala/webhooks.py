"""Webhook signature verification utilities."""

from __future__ import annotations

import hashlib
import hmac
from typing import Union

from oelala.exceptions import AuthenticationError
from oelala.models import WebhookEvent


def verify_webhook_signature(
    payload: Union[str, bytes],
    signature: str,
    secret: str,
) -> bool:
    """Verify an Oelala webhook signature.

    Args:
        payload: The raw request body (string or bytes).
        signature: The ``X-Webhook-Signature`` header value
            (format: ``sha256=<hex>``).
        secret: Your webhook secret (format: ``whsec_...``).

    Returns:
        True if the signature is valid.

    Raises:
        AuthenticationError: If the signature is invalid or malformed.

    Example::

        from oelala import verify_webhook_signature

        # In your webhook handler (e.g. Flask, FastAPI, Django)
        is_valid = verify_webhook_signature(
            payload=request.body,
            signature=request.headers["X-Webhook-Signature"],
            secret="whsec_your_secret",
        )
    """
    if isinstance(payload, str):
        payload = payload.encode("utf-8")

    if not signature.startswith("sha256="):
        raise AuthenticationError("Invalid signature format: must start with 'sha256='")

    expected_hex = signature[7:]  # strip "sha256="

    computed = hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(computed, expected_hex):
        raise AuthenticationError("Invalid webhook signature")

    return True


def parse_webhook_event(payload: dict) -> WebhookEvent:
    """Parse a webhook payload dict into a WebhookEvent.

    Args:
        payload: The parsed JSON body of the webhook request.

    Returns:
        :class:`WebhookEvent` instance.
    """
    return WebhookEvent(
        event=payload["event"],
        event_id=payload.get("event_id", ""),
        timestamp=payload.get("timestamp", ""),
        data=payload.get("data", {}),
    )
