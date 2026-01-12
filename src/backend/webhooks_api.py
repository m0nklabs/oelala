#!/usr/bin/env python3
"""
Webhooks API - CRUD operations for webhook management.

Endpoints:
- GET    /webhooks              - List user's webhooks
- POST   /webhooks              - Create a new webhook
- GET    /webhooks/{id}         - Get webhook details
- PATCH  /webhooks/{id}         - Update webhook
- DELETE /webhooks/{id}         - Delete webhook
- GET    /webhooks/{id}/deliveries - Get delivery history
- POST   /webhooks/{id}/test    - Send test webhook
"""

import logging
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, HttpUrl, field_validator

from auth import User, get_current_user
from webhook_service import (
    WebhookEvent,
    WebhookPayload,
    WebhookService,
    webhook_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# Supabase config
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")


def _supabase_headers() -> Dict[str, str]:
    """Headers for Supabase requests."""
    if not SUPABASE_KEY:
        return {}
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


# ==============================================================================
# Pydantic Models
# ==============================================================================


class WebhookCreate(BaseModel):
    """Request model for creating a webhook."""

    name: str = Field(..., min_length=1, max_length=255, description="Friendly name")
    url: str = Field(..., description="Webhook endpoint URL (HTTPS recommended)")
    events: List[str] = Field(
        default=["job.completed", "job.failed"],
        description="Event types to subscribe to",
    )
    description: Optional[str] = Field(None, max_length=500)
    headers: Optional[Dict[str, str]] = Field(default_factory=dict)
    enabled: bool = Field(default=True)

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate webhook URL."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        if not v.startswith("https://") and "localhost" not in v and "127.0.0.1" not in v:
            logger.warning(f"Non-HTTPS webhook URL configured: {v}")
        return v

    @field_validator("events")
    @classmethod
    def validate_events(cls, v: List[str]) -> List[str]:
        """Validate event types."""
        if not v:
            raise ValueError("At least one event type is required")
        invalid = set(v) - set(WebhookEvent.ALL_EVENTS)
        if invalid:
            raise ValueError(
                f"Invalid event types: {invalid}. "
                f"Valid types: {WebhookEvent.ALL_EVENTS}"
            )
        return v


class WebhookUpdate(BaseModel):
    """Request model for updating a webhook."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    url: Optional[str] = None
    events: Optional[List[str]] = None
    description: Optional[str] = Field(None, max_length=500)
    headers: Optional[Dict[str, str]] = None
    enabled: Optional[bool] = None
    regenerate_secret: bool = Field(default=False, description="Generate a new secret")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate webhook URL."""
        if v is None:
            return v
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    @field_validator("events")
    @classmethod
    def validate_events(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate event types."""
        if v is None:
            return v
        if not v:
            raise ValueError("At least one event type is required")
        invalid = set(v) - set(WebhookEvent.ALL_EVENTS)
        if invalid:
            raise ValueError(
                f"Invalid event types: {invalid}. "
                f"Valid types: {WebhookEvent.ALL_EVENTS}"
            )
        return v


class WebhookResponse(BaseModel):
    """Response model for webhook."""

    id: str
    name: str
    url: str
    events: List[str]
    enabled: bool
    description: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    secret: Optional[str] = None  # Only included on create
    last_delivery_at: Optional[str] = None
    last_delivery_status: Optional[str] = None
    total_deliveries: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    created_at: str
    updated_at: str


class WebhookDeliveryResponse(BaseModel):
    """Response model for webhook delivery."""

    id: str
    event_type: str
    event_id: str
    status: str
    attempt_count: int
    max_attempts: int
    response_status: Optional[int] = None
    response_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    created_at: str
    delivered_at: Optional[str] = None


class TestWebhookResponse(BaseModel):
    """Response model for test webhook."""

    success: bool
    status_code: Optional[int] = None
    response_time_ms: Optional[int] = None
    error: Optional[str] = None


# ==============================================================================
# API Endpoints
# ==============================================================================


@router.get("", response_model=List[WebhookResponse])
async def list_webhooks(
    current_user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """
    List all webhooks for the current user.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        async with httpx.AsyncClient() as client:
            url = f"{SUPABASE_URL}/rest/v1/webhooks"
            params = {
                "user_id": f"eq.{current_user.id}",
                "select": "*",
                "order": "created_at.desc",
            }

            resp = await client.get(
                url,
                headers=_supabase_headers(),
                params=params,
                timeout=10.0,
            )

            if resp.status_code != 200:
                logger.error(f"Supabase error: {resp.status_code} - {resp.text}")
                raise HTTPException(status_code=500, detail="Failed to fetch webhooks")

            webhooks = resp.json()

            # Remove secret from list response
            for webhook in webhooks:
                webhook.pop("secret", None)

            return webhooks

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing webhooks: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("", response_model=WebhookResponse, status_code=201)
async def create_webhook(
    data: WebhookCreate,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Create a new webhook.

    Returns the created webhook including the secret (only shown once).
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(status_code=503, detail="Database not configured")

    # Generate a secure secret
    secret = f"whsec_{secrets.token_urlsafe(32)}"

    try:
        async with httpx.AsyncClient() as client:
            url = f"{SUPABASE_URL}/rest/v1/webhooks"

            insert_data = {
                "user_id": current_user.id,
                "name": data.name,
                "url": data.url,
                "secret": secret,
                "events": data.events,
                "description": data.description,
                "headers": data.headers or {},
                "enabled": data.enabled,
            }

            resp = await client.post(
                url,
                headers=_supabase_headers(),
                json=insert_data,
                timeout=10.0,
            )

            if resp.status_code not in (200, 201):
                logger.error(f"Supabase error: {resp.status_code} - {resp.text}")
                raise HTTPException(status_code=500, detail="Failed to create webhook")

            webhooks = resp.json()
            if not webhooks:
                raise HTTPException(status_code=500, detail="Failed to create webhook")

            webhook = webhooks[0]
            # Include secret in create response (only time it's shown)
            webhook["secret"] = secret

            logger.info(f"Created webhook {webhook['id']} for user {current_user.id}")
            return webhook

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating webhook: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{webhook_id}", response_model=WebhookResponse)
async def get_webhook(
    webhook_id: str,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get a specific webhook by ID.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        async with httpx.AsyncClient() as client:
            url = f"{SUPABASE_URL}/rest/v1/webhooks"
            params = {
                "id": f"eq.{webhook_id}",
                "user_id": f"eq.{current_user.id}",
                "select": "*",
            }

            resp = await client.get(
                url,
                headers=_supabase_headers(),
                params=params,
                timeout=10.0,
            )

            if resp.status_code != 200:
                logger.error(f"Supabase error: {resp.status_code} - {resp.text}")
                raise HTTPException(status_code=500, detail="Failed to fetch webhook")

            webhooks = resp.json()
            if not webhooks:
                raise HTTPException(status_code=404, detail="Webhook not found")

            webhook = webhooks[0]
            # Remove secret from response
            webhook.pop("secret", None)

            return webhook

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting webhook: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch("/{webhook_id}", response_model=WebhookResponse)
async def update_webhook(
    webhook_id: str,
    data: WebhookUpdate,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Update a webhook.

    If regenerate_secret is true, a new secret will be generated and returned.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        async with httpx.AsyncClient() as client:
            # First verify ownership
            check_url = f"{SUPABASE_URL}/rest/v1/webhooks"
            check_params = {
                "id": f"eq.{webhook_id}",
                "user_id": f"eq.{current_user.id}",
                "select": "id",
            }

            check_resp = await client.get(
                check_url,
                headers=_supabase_headers(),
                params=check_params,
                timeout=10.0,
            )

            if check_resp.status_code != 200 or not check_resp.json():
                raise HTTPException(status_code=404, detail="Webhook not found")

            # Build update data
            update_data: Dict[str, Any] = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            if data.name is not None:
                update_data["name"] = data.name
            if data.url is not None:
                update_data["url"] = data.url
            if data.events is not None:
                update_data["events"] = data.events
            if data.description is not None:
                update_data["description"] = data.description
            if data.headers is not None:
                update_data["headers"] = data.headers
            if data.enabled is not None:
                update_data["enabled"] = data.enabled

            new_secret = None
            if data.regenerate_secret:
                new_secret = f"whsec_{secrets.token_urlsafe(32)}"
                update_data["secret"] = new_secret

            # Update
            url = f"{SUPABASE_URL}/rest/v1/webhooks"
            params = {
                "id": f"eq.{webhook_id}",
                "user_id": f"eq.{current_user.id}",
            }

            resp = await client.patch(
                url,
                headers=_supabase_headers(),
                params=params,
                json=update_data,
                timeout=10.0,
            )

            if resp.status_code != 200:
                logger.error(f"Supabase error: {resp.status_code} - {resp.text}")
                raise HTTPException(status_code=500, detail="Failed to update webhook")

            webhooks = resp.json()
            if not webhooks:
                raise HTTPException(status_code=404, detail="Webhook not found")

            webhook = webhooks[0]

            # Include new secret if regenerated
            if new_secret:
                webhook["secret"] = new_secret
            else:
                webhook.pop("secret", None)

            logger.info(f"Updated webhook {webhook_id}")
            return webhook

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating webhook: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{webhook_id}", status_code=204)
async def delete_webhook(
    webhook_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Delete a webhook.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        async with httpx.AsyncClient() as client:
            url = f"{SUPABASE_URL}/rest/v1/webhooks"
            params = {
                "id": f"eq.{webhook_id}",
                "user_id": f"eq.{current_user.id}",
            }

            resp = await client.delete(
                url,
                headers=_supabase_headers(),
                params=params,
                timeout=10.0,
            )

            if resp.status_code not in (200, 204):
                logger.error(f"Supabase error: {resp.status_code} - {resp.text}")
                raise HTTPException(status_code=500, detail="Failed to delete webhook")

            logger.info(f"Deleted webhook {webhook_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting webhook: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{webhook_id}/deliveries", response_model=List[WebhookDeliveryResponse])
async def list_deliveries(
    webhook_id: str,
    current_user: User = Depends(get_current_user),
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
) -> List[Dict[str, Any]]:
    """
    List delivery history for a webhook.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        async with httpx.AsyncClient() as client:
            # First verify ownership
            check_url = f"{SUPABASE_URL}/rest/v1/webhooks"
            check_params = {
                "id": f"eq.{webhook_id}",
                "user_id": f"eq.{current_user.id}",
                "select": "id",
            }

            check_resp = await client.get(
                check_url,
                headers=_supabase_headers(),
                params=check_params,
                timeout=10.0,
            )

            if check_resp.status_code != 200 or not check_resp.json():
                raise HTTPException(status_code=404, detail="Webhook not found")

            # Fetch deliveries
            url = f"{SUPABASE_URL}/rest/v1/webhook_deliveries"
            params = {
                "webhook_id": f"eq.{webhook_id}",
                "select": "id,event_type,event_id,status,attempt_count,max_attempts,response_status,response_time_ms,error_message,created_at,delivered_at",
                "order": "created_at.desc",
                "limit": str(limit),
                "offset": str(offset),
            }

            resp = await client.get(
                url,
                headers=_supabase_headers(),
                params=params,
                timeout=10.0,
            )

            if resp.status_code != 200:
                logger.error(f"Supabase error: {resp.status_code} - {resp.text}")
                raise HTTPException(status_code=500, detail="Failed to fetch deliveries")

            return resp.json()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing deliveries: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{webhook_id}/test", response_model=TestWebhookResponse)
async def test_webhook(
    webhook_id: str,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Send a test webhook to verify the endpoint is working.

    Sends a job.completed event with test data.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        async with httpx.AsyncClient() as client:
            # Fetch webhook (including secret for signing)
            url = f"{SUPABASE_URL}/rest/v1/webhooks"
            params = {
                "id": f"eq.{webhook_id}",
                "user_id": f"eq.{current_user.id}",
                "select": "id,url,secret,headers,enabled",
            }

            resp = await client.get(
                url,
                headers=_supabase_headers(),
                params=params,
                timeout=10.0,
            )

            if resp.status_code != 200 or not resp.json():
                raise HTTPException(status_code=404, detail="Webhook not found")

            webhook = resp.json()[0]

            if not webhook.get("enabled"):
                raise HTTPException(status_code=400, detail="Webhook is disabled")

            # Create test payload
            payload = WebhookPayload(
                event_type="test",
                data={
                    "message": "This is a test webhook from Oelala",
                    "job_id": "test_job_123",
                    "job_type": "test",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Deliver test webhook
            result = await webhook_service.deliver_webhook(
                webhook=webhook,
                payload=payload,
            )

            return {
                "success": result["status"] == "success",
                "status_code": result.get("response_status"),
                "response_time_ms": result.get("response_time_ms"),
                "error": result.get("error_message"),
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing webhook: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/events/types", response_model=List[str])
async def list_event_types(
    current_user: User = Depends(get_current_user),
) -> List[str]:
    """
    List all available webhook event types.
    """
    return WebhookEvent.ALL_EVENTS
