#!/usr/bin/env python3
"""
Webhook Delivery Service - Async notifications with HMAC signing and retry logic.

Handles:
- Webhook event delivery with HMAC-SHA256 signing
- Exponential backoff retry logic (5 attempts)
- Delivery logging to Supabase
- Integration with job queue events
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

DEBUG_ENABLED = os.getenv("WEBHOOK_DEBUG", "false").lower() == "true"


def debug_log(message: str):
    """Emit debug logs when DEBUG_ENABLED is true."""
    if DEBUG_ENABLED:
        logger.info(f"🐛 [Webhook] {message}")


# Webhook event types
class WebhookEvent:
    """Webhook event type constants."""

    JOB_QUEUED = "job.queued"
    JOB_STARTED = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"

    ALL_EVENTS = [JOB_QUEUED, JOB_STARTED, JOB_COMPLETED, JOB_FAILED]


# Retry configuration (exponential backoff)
RETRY_DELAYS = [
    10,  # Retry 1: 10 seconds
    60,  # Retry 2: 1 minute
    300,  # Retry 3: 5 minutes
    1800,  # Retry 4: 30 minutes
    3600,  # Retry 5: 1 hour
]
MAX_ATTEMPTS = len(RETRY_DELAYS) + 1  # Initial attempt + retries

# HTTP timeout for webhook delivery
DELIVERY_TIMEOUT = 10.0  # seconds


class WebhookPayload:
    """Represents a webhook payload to be delivered."""

    def __init__(
        self,
        event_type: str,
        data: Dict[str, Any],
        event_id: Optional[str] = None,
    ):
        self.event_type = event_type
        self.event_id = event_id or str(uuid.uuid4())
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.data = data

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event": self.event_type,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "data": self.data,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), separators=(",", ":"))


class WebhookService:
    """
    Manages webhook delivery with HMAC signing and retry logic.

    Features:
    - HMAC-SHA256 signature verification
    - Exponential backoff retry (5 attempts)
    - Async delivery with logging
    - Supabase integration for persistence
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
    ):
        """
        Initialize webhook service.

        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase service key
        """
        self.supabase_url = (supabase_url or os.getenv("SUPABASE_URL", "")).rstrip("/")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_KEY")

        # Background retry task
        self._retry_task: Optional[asyncio.Task] = None
        self._running = False

        # In-memory cache of active webhooks (refreshed periodically)
        self._webhook_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._cache_expires_at: float = 0

        logger.info("🪝 WebhookService initialized")

    def _supabase_headers(self) -> Dict[str, str]:
        """Headers for Supabase requests."""
        if not self.supabase_key:
            return {}
        return {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    @staticmethod
    def generate_signature(payload: str, secret: str) -> str:
        """
        Generate HMAC-SHA256 signature for webhook payload.

        Args:
            payload: JSON string payload
            secret: Webhook secret key

        Returns:
            Hex-encoded signature prefixed with 'sha256='
        """
        signature = hmac.new(
            secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"sha256={signature}"

    @staticmethod
    def verify_signature(payload: str, signature: str, secret: str) -> bool:
        """
        Verify HMAC-SHA256 signature.

        Args:
            payload: JSON string payload
            signature: Received signature (with 'sha256=' prefix)
            secret: Webhook secret key

        Returns:
            True if signature is valid
        """
        expected = WebhookService.generate_signature(payload, secret)
        return hmac.compare_digest(expected, signature)

    async def get_webhooks_for_user(
        self, user_id: str, event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get enabled webhooks for a user, optionally filtered by event type.

        Args:
            user_id: User UUID
            event_type: Optional event type to filter by

        Returns:
            List of webhook configurations
        """
        if not self.supabase_url or not self.supabase_key:
            debug_log("Supabase not configured, returning empty webhooks")
            return []

        try:
            async with httpx.AsyncClient() as client:
                # Query enabled webhooks for user
                url = f"{self.supabase_url}/rest/v1/webhooks"
                params = {
                    "user_id": f"eq.{user_id}",
                    "enabled": "eq.true",
                    "select": "id,name,url,secret,events,headers",
                }

                resp = await client.get(
                    url,
                    headers=self._supabase_headers(),
                    params=params,
                    timeout=5.0,
                )

                if resp.status_code != 200:
                    logger.warning(f"Failed to fetch webhooks: {resp.status_code}")
                    return []

                webhooks = resp.json()

                # Filter by event type if specified
                if event_type:
                    webhooks = [
                        w for w in webhooks if event_type in (w.get("events") or [])
                    ]

                debug_log(f"Found {len(webhooks)} webhooks for user {user_id}")
                return webhooks

        except Exception as e:
            logger.error(f"Error fetching webhooks: {e}")
            return []

    async def deliver_webhook(
        self,
        webhook: Dict[str, Any],
        payload: WebhookPayload,
        delivery_id: Optional[str] = None,
        attempt: int = 1,
    ) -> Dict[str, Any]:
        """
        Deliver a webhook with HMAC signing.

        Args:
            webhook: Webhook configuration
            payload: Payload to deliver
            delivery_id: Optional existing delivery ID (for retries)
            attempt: Current attempt number (1-based)

        Returns:
            Delivery result dict with status, response_status, etc.
        """
        webhook_id = webhook["id"]
        url = webhook["url"]
        secret = webhook["secret"]
        custom_headers = webhook.get("headers") or {}

        # Generate JSON payload
        payload_json = payload.to_json()
        signature = self.generate_signature(payload_json, secret)

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Signature": signature,
            "X-Webhook-Event": payload.event_type,
            "X-Webhook-Id": payload.event_id,
            "X-Webhook-Timestamp": payload.timestamp,
            "User-Agent": "Oelala-Webhook/1.0",
            **custom_headers,
        }

        result = {
            "webhook_id": webhook_id,
            "event_type": payload.event_type,
            "event_id": payload.event_id,
            "attempt": attempt,
            "status": "pending",
            "response_status": None,
            "response_body": None,
            "response_time_ms": None,
            "error_message": None,
        }

        start_time = time.time()

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    content=payload_json,
                    headers=headers,
                    timeout=DELIVERY_TIMEOUT,
                )

                result["response_time_ms"] = int((time.time() - start_time) * 1000)
                result["response_status"] = resp.status_code

                # Truncate response body
                body = resp.text[:1000] if resp.text else None
                result["response_body"] = body

                # Check success (2xx status codes)
                if 200 <= resp.status_code < 300:
                    result["status"] = "success"
                    logger.info(
                        f"✅ Webhook delivered: {webhook_id} -> {url} "
                        f"(status={resp.status_code}, time={result['response_time_ms']}ms)"
                    )
                else:
                    result["status"] = "failed"
                    result["error_message"] = f"HTTP {resp.status_code}: {body}"
                    logger.warning(
                        f"⚠️ Webhook delivery failed: {webhook_id} -> {url} "
                        f"(status={resp.status_code})"
                    )

        except httpx.TimeoutException:
            result["response_time_ms"] = int((time.time() - start_time) * 1000)
            result["status"] = "failed"
            result["error_message"] = f"Timeout after {DELIVERY_TIMEOUT}s"
            logger.warning(f"⏱️ Webhook timeout: {webhook_id} -> {url}")

        except httpx.RequestError as e:
            result["response_time_ms"] = int((time.time() - start_time) * 1000)
            result["status"] = "failed"
            result["error_message"] = f"Connection error: {str(e)}"
            logger.warning(f"🔌 Webhook connection error: {webhook_id} -> {url}: {e}")

        except Exception as e:
            result["response_time_ms"] = int((time.time() - start_time) * 1000)
            result["status"] = "failed"
            result["error_message"] = f"Unexpected error: {str(e)}"
            logger.error(f"❌ Webhook error: {webhook_id} -> {url}: {e}")

        # Log delivery to database
        await self._log_delivery(
            webhook_id=webhook_id,
            delivery_id=delivery_id,
            payload=payload,
            result=result,
            attempt=attempt,
        )

        return result

    async def _log_delivery(
        self,
        webhook_id: str,
        delivery_id: Optional[str],
        payload: WebhookPayload,
        result: Dict[str, Any],
        attempt: int,
    ):
        """Log delivery attempt to database."""
        if not self.supabase_url or not self.supabase_key:
            debug_log("Supabase not configured, skipping delivery log")
            return

        try:
            async with httpx.AsyncClient() as client:
                if delivery_id:
                    # Update existing delivery record
                    url = f"{self.supabase_url}/rest/v1/webhook_deliveries"
                    params = {"id": f"eq.{delivery_id}"}

                    update_data = {
                        "status": result["status"],
                        "attempt_count": attempt,
                        "response_status": result["response_status"],
                        "response_body": result["response_body"],
                        "response_time_ms": result["response_time_ms"],
                        "error_message": result["error_message"],
                    }

                    if result["status"] == "success":
                        update_data["delivered_at"] = datetime.now(
                            timezone.utc
                        ).isoformat()
                        update_data["next_retry_at"] = None
                    elif result["status"] == "failed" and attempt < MAX_ATTEMPTS:
                        # Schedule retry
                        delay = RETRY_DELAYS[attempt - 1]
                        next_retry = datetime.now(timezone.utc) + timedelta(
                            seconds=delay
                        )
                        update_data["status"] = "retrying"
                        update_data["next_retry_at"] = next_retry.isoformat()

                    resp = await client.patch(
                        url,
                        headers=self._supabase_headers(),
                        params=params,
                        json=update_data,
                        timeout=5.0,
                    )

                    if resp.status_code != 200:
                        logger.warning(f"Failed to update delivery: {resp.status_code}")

                else:
                    # Create new delivery record
                    url = f"{self.supabase_url}/rest/v1/webhook_deliveries"

                    insert_data = {
                        "webhook_id": webhook_id,
                        "event_type": payload.event_type,
                        "event_id": payload.event_id,
                        "payload": payload.to_dict(),
                        "status": result["status"],
                        "attempt_count": attempt,
                        "max_attempts": MAX_ATTEMPTS,
                        "response_status": result["response_status"],
                        "response_body": result["response_body"],
                        "response_time_ms": result["response_time_ms"],
                        "error_message": result["error_message"],
                    }

                    if result["status"] == "success":
                        insert_data["delivered_at"] = datetime.now(
                            timezone.utc
                        ).isoformat()
                    elif result["status"] == "failed" and attempt < MAX_ATTEMPTS:
                        # Schedule retry
                        delay = RETRY_DELAYS[attempt - 1]
                        next_retry = datetime.now(timezone.utc) + timedelta(
                            seconds=delay
                        )
                        insert_data["status"] = "retrying"
                        insert_data["next_retry_at"] = next_retry.isoformat()

                    resp = await client.post(
                        url,
                        headers=self._supabase_headers(),
                        json=insert_data,
                        timeout=5.0,
                    )

                    if resp.status_code not in (200, 201):
                        logger.warning(f"Failed to create delivery: {resp.status_code}")

        except Exception as e:
            logger.error(f"Error logging delivery: {e}")

    async def dispatch_event(
        self,
        user_id: str,
        event_type: str,
        data: Dict[str, Any],
    ):
        """
        Dispatch an event to all relevant webhooks for a user.

        This is the main entry point for triggering webhooks.

        Args:
            user_id: User UUID
            event_type: Event type (job.queued, job.completed, etc.)
            data: Event data payload
        """
        debug_log(f"Dispatching {event_type} for user {user_id}")

        # Get webhooks for this user and event type
        webhooks = await self.get_webhooks_for_user(user_id, event_type)

        if not webhooks:
            debug_log(f"No webhooks configured for {event_type}")
            return

        # Create payload
        payload = WebhookPayload(event_type=event_type, data=data)

        # Deliver to all webhooks concurrently
        tasks = [self.deliver_webhook(webhook, payload) for webhook in webhooks]

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(
                1
                for r in results
                if isinstance(r, dict) and r.get("status") == "success"
            )
            logger.info(
                f"🪝 Dispatched {event_type} to {len(webhooks)} webhooks "
                f"({success_count} successful)"
            )

    async def process_pending_retries(self):
        """
        Process pending webhook retries.
        Called periodically by the background task.
        """
        if not self.supabase_url or not self.supabase_key:
            return

        try:
            async with httpx.AsyncClient() as client:
                # Get deliveries ready for retry
                now = datetime.now(timezone.utc).isoformat()
                url = f"{self.supabase_url}/rest/v1/webhook_deliveries"
                params = {
                    "status": "eq.retrying",
                    "next_retry_at": f"lte.{now}",
                    "select": "id,webhook_id,event_type,payload,attempt_count",
                    "limit": "10",
                }

                resp = await client.get(
                    url,
                    headers=self._supabase_headers(),
                    params=params,
                    timeout=5.0,
                )

                if resp.status_code != 200:
                    logger.warning(
                        f"Failed to fetch pending retries: {resp.status_code}"
                    )
                    return

                deliveries = resp.json()

                if not deliveries:
                    debug_log("No pending retries")
                    return

                logger.info(f"🔄 Processing {len(deliveries)} pending retries")

                for delivery in deliveries:
                    await self._process_single_retry(client, delivery)

        except Exception as e:
            logger.error(f"Error processing retries: {e}")

    async def _process_single_retry(
        self,
        client: httpx.AsyncClient,
        delivery: Dict[str, Any],
    ):
        """Process a single retry delivery."""
        try:
            delivery_id = delivery["id"]
            webhook_id = delivery["webhook_id"]
            attempt = delivery["attempt_count"] + 1

            # Fetch webhook config
            url = f"{self.supabase_url}/rest/v1/webhooks"
            params = {
                "id": f"eq.{webhook_id}",
                "enabled": "eq.true",
                "select": "id,url,secret,headers",
            }

            resp = await client.get(
                url,
                headers=self._supabase_headers(),
                params=params,
                timeout=5.0,
            )

            if resp.status_code != 200:
                logger.warning(f"Failed to fetch webhook for retry: {resp.status_code}")
                return

            webhooks = resp.json()
            if not webhooks:
                # Webhook was disabled or deleted, mark as failed
                await self._mark_delivery_failed(
                    client, delivery_id, "Webhook disabled"
                )
                return

            webhook = webhooks[0]

            # Recreate payload from stored data
            stored_payload = delivery["payload"]
            payload = WebhookPayload(
                event_type=stored_payload["event"],
                data=stored_payload["data"],
                event_id=stored_payload["event_id"],
            )

            # Attempt delivery
            await self.deliver_webhook(
                webhook=webhook,
                payload=payload,
                delivery_id=delivery_id,
                attempt=attempt,
            )

        except Exception as e:
            logger.error(f"Error processing retry {delivery.get('id')}: {e}")

    async def _mark_delivery_failed(
        self,
        client: httpx.AsyncClient,
        delivery_id: str,
        error: str,
    ):
        """Mark a delivery as permanently failed."""
        try:
            url = f"{self.supabase_url}/rest/v1/webhook_deliveries"
            params = {"id": f"eq.{delivery_id}"}

            await client.patch(
                url,
                headers=self._supabase_headers(),
                params=params,
                json={
                    "status": "failed",
                    "error_message": error,
                    "next_retry_at": None,
                },
                timeout=5.0,
            )

        except Exception as e:
            logger.error(f"Error marking delivery failed: {e}")

    async def start_retry_worker(self, interval: float = 30.0):
        """
        Start background worker for processing retries.

        Args:
            interval: Check interval in seconds (default 30s)
        """
        if self._running:
            logger.warning("Retry worker already running")
            return

        self._running = True
        logger.info(f"🔄 Starting webhook retry worker (interval: {interval}s)")

        async def worker_loop():
            while self._running:
                try:
                    await self.process_pending_retries()
                except Exception as e:
                    logger.error(f"Error in retry worker: {e}")
                await asyncio.sleep(interval)

        self._retry_task = asyncio.create_task(worker_loop())

    async def stop_retry_worker(self):
        """Stop background retry worker."""
        self._running = False
        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                debug_log("Retry worker cancellation acknowledged")
            self._retry_task = None
        logger.info("🛑 Stopped webhook retry worker")


# Global webhook service instance
webhook_service = WebhookService()


# ==============================================================================
# Helper functions for job event integration
# ==============================================================================


async def trigger_job_queued(
    user_id: str,
    job_id: str,
    job_type: str,
    queue_position: int,
    total_pending: int,
    eta_seconds: Optional[int] = None,
):
    """Trigger job.queued webhook event."""
    data = {
        "job_id": job_id,
        "job_type": job_type,
        "queue_position": queue_position,
        "total_pending": total_pending,
    }
    if eta_seconds is not None:
        data["eta_seconds"] = eta_seconds

    await webhook_service.dispatch_event(
        user_id=user_id,
        event_type=WebhookEvent.JOB_QUEUED,
        data=data,
    )


async def trigger_job_started(
    user_id: str,
    job_id: str,
    job_type: str,
):
    """Trigger job.started webhook event."""
    await webhook_service.dispatch_event(
        user_id=user_id,
        event_type=WebhookEvent.JOB_STARTED,
        data={
            "job_id": job_id,
            "job_type": job_type,
        },
    )


async def trigger_job_completed(
    user_id: str,
    job_id: str,
    job_type: str,
    output_url: Optional[str] = None,
    processing_time_seconds: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Trigger job.completed webhook event."""
    data = {
        "job_id": job_id,
        "job_type": job_type,
    }
    if output_url:
        data["output_url"] = output_url
    if processing_time_seconds is not None:
        data["processing_time_seconds"] = processing_time_seconds
    if metadata:
        data["metadata"] = metadata

    await webhook_service.dispatch_event(
        user_id=user_id,
        event_type=WebhookEvent.JOB_COMPLETED,
        data=data,
    )


async def trigger_job_failed(
    user_id: str,
    job_id: str,
    job_type: str,
    error: str,
):
    """Trigger job.failed webhook event."""
    await webhook_service.dispatch_event(
        user_id=user_id,
        event_type=WebhookEvent.JOB_FAILED,
        data={
            "job_id": job_id,
            "job_type": job_type,
            "error": error,
        },
    )
