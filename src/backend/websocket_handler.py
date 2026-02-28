#!/usr/bin/env python3
"""
WebSocket Event Handler for Real-time Progress Updates
Manages client connections and broadcasts queue/progress events
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set, Any, Optional
from fastapi import WebSocket
from collections import defaultdict

logger = logging.getLogger(__name__)

DEBUG_ENABLED = False  # Set to True for verbose logging


def debug_log(message: str):
    """Emit debug logs when DEBUG_ENABLED is true."""
    if DEBUG_ENABLED:
        logger.info(f"🐛 {message}")


# Lazy import to avoid circular dependency
_webhook_triggers = None


def _get_webhook_triggers():
    """Lazy load webhook trigger functions to avoid circular imports."""
    global _webhook_triggers
    if _webhook_triggers is None:
        try:
            from webhook_service import (
                trigger_job_queued,
                trigger_job_started,
                trigger_job_completed,
                trigger_job_failed,
            )

            _webhook_triggers = {
                "queued": trigger_job_queued,
                "started": trigger_job_started,
                "completed": trigger_job_completed,
                "failed": trigger_job_failed,
            }
            debug_log("Webhook triggers loaded")
        except ImportError as e:
            logger.warning(f"Webhook service not available: {e}")
            _webhook_triggers = {}
    return _webhook_triggers


class WebSocketManager:
    """
    Manages WebSocket connections for real-time progress updates.
    Supports multiple clients per user and broadcasts queue/progress events.
    """

    def __init__(self):
        # WebSocket connections grouped by user_id
        self.connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        # Track job ownership: job_id -> {user_id, job_type, started_at}
        self.job_ownership: Dict[str, Dict[str, Any]] = {}
        # Last broadcast timestamps to avoid spam
        self.last_broadcast: Dict[str, float] = {}

    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None):
        """Register a new WebSocket connection (must be already accepted)"""
        user_key = user_id or "anonymous"
        self.connections[user_key].add(websocket)
        logger.info(
            f"📡 WebSocket connected for user {user_key} (total: {len(self.connections[user_key])})"
        )
        debug_log(f"Active users: {list(self.connections.keys())}")

    def disconnect(self, websocket: WebSocket, user_id: Optional[str] = None):
        """Unregister a WebSocket connection"""
        user_key = user_id or "anonymous"
        if websocket in self.connections[user_key]:
            self.connections[user_key].discard(websocket)
            logger.info(
                f"📡 WebSocket disconnected for user {user_key} (remaining: {len(self.connections[user_key])})"
            )
            # Clean up empty user sets
            if not self.connections[user_key]:
                del self.connections[user_key]
                debug_log(f"Removed empty connection set for user {user_key}")

    def register_job(
        self, job_id: str, user_id: Optional[str] = None, job_type: str = "generation"
    ):
        """Register a job for a specific user"""
        user_key = user_id or "anonymous"
        self.job_ownership[job_id] = {
            "user_id": user_key,
            "job_type": job_type,
            "started_at": None,
        }
        debug_log(f"Registered job {job_id} for user {user_key} (type: {job_type})")

    def unregister_job(self, job_id: str):
        """Unregister a completed/failed job"""
        if job_id in self.job_ownership:
            job_info = self.job_ownership.pop(job_id)
            debug_log(f"Unregistered job {job_id} for user {job_info.get('user_id')}")

            # Clean up rate limiting cache for this job to prevent memory leak
            keys_to_remove = [k for k in self.last_broadcast.keys() if job_id in k]
            for key in keys_to_remove:
                del self.last_broadcast[key]
            if keys_to_remove:
                debug_log(
                    f"Cleaned up {len(keys_to_remove)} rate limit entries for job {job_id}"
                )

    async def broadcast_to_user(
        self, user_id: Optional[str], event_type: str, data: Dict[str, Any]
    ):
        """
        Broadcast an event to all connections for a specific user.

        Args:
            user_id: User ID (None for anonymous)
            event_type: Event type ('queue_update', 'progress', 'job_complete', 'job_failed')
            data: Event payload
        """
        user_key = user_id or "anonymous"
        if user_key not in self.connections:
            debug_log(f"No connections for user {user_key}, skipping broadcast")
            return

        # Rate limiting: avoid spamming updates faster than 100ms
        now = asyncio.get_running_loop().time()
        cache_key = f"{user_key}:{event_type}:{data.get('job_id', 'unknown')}"
        last_time = self.last_broadcast.get(cache_key, 0)
        if now - last_time < 0.1 and event_type == "progress":
            debug_log(f"Rate limiting: skipping duplicate event {cache_key}")
            return
        self.last_broadcast[cache_key] = now

        message = json.dumps(
            {
                "type": event_type,
                "timestamp": datetime.now().isoformat(),
                "data": data,
            }
        )

        disconnected = set()
        for ws in self.connections[user_key]:
            try:
                await ws.send_text(message)
                debug_log(f"Sent {event_type} to user {user_key}")
            except Exception as e:
                logger.warning(f"Failed to send to websocket: {e}")
                disconnected.add(ws)

        # Clean up disconnected clients
        if disconnected:
            self.connections[user_key].difference_update(disconnected)
            logger.info(
                f"📡 Removed {len(disconnected)} dead connections for user {user_key}"
            )

    async def broadcast_queue_update(
        self,
        job_id: str,
        queue_position: int,
        total_pending: int,
        eta_seconds: Optional[int] = None,
    ):
        """
        Broadcast queue position update to job owner.

        Args:
            job_id: Job/prompt ID
            queue_position: Current position in queue (0 = running)
            total_pending: Total number of pending jobs
            eta_seconds: Estimated time to start (optional)
        """
        job_info = self.job_ownership.get(job_id)
        if not job_info:
            debug_log(f"Job {job_id} not registered, cannot broadcast queue update")
            return

        user_id = job_info.get("user_id")
        job_type = job_info.get("job_type", "generation")
        if not user_id:
            debug_log(f"Job {job_id} not registered, cannot broadcast queue update")
            return

        data = {
            "job_id": job_id,
            "queue_position": queue_position,
            "total_pending": total_pending,
            "status": "running" if queue_position == 0 else "queued",
        }
        if eta_seconds is not None:
            data["eta_seconds"] = eta_seconds
            data["eta_human"] = self._format_eta(eta_seconds)

        await self.broadcast_to_user(user_id, "queue_update", data)
        logger.info(
            f"📊 Queue update: job {job_id} at position {queue_position}/{total_pending}"
        )

        # Trigger job.queued webhook (only when entering queue, not when running)
        if queue_position > 0:
            triggers = _get_webhook_triggers()
            if triggers.get("queued") and user_id and user_id != "anonymous":
                try:
                    asyncio.create_task(
                        triggers["queued"](
                            user_id=user_id,
                            job_id=job_id,
                            job_type=job_type,
                            queue_position=queue_position,
                            total_pending=total_pending,
                            eta_seconds=eta_seconds,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to trigger job.queued webhook: {e}")

    async def broadcast_progress(
        self,
        job_id: str,
        progress: int,
        message: Optional[str] = None,
        node_name: Optional[str] = None,
    ):
        """
        Broadcast generation progress update to job owner.

        Args:
            job_id: Job/prompt ID
            progress: Progress percentage (0-100)
            message: Optional status message
            node_name: Optional current processing node name
        """
        job_info = self.job_ownership.get(job_id)
        if not job_info:
            debug_log(f"Job {job_id} not registered, cannot broadcast progress")
            return

        user_id = job_info.get("user_id")
        job_type = job_info.get("job_type", "generation")

        # Track when job starts running (for job.started webhook)
        if progress > 0 and not job_info.get("started_at"):
            import time

            job_info["started_at"] = time.time()
            self.job_ownership[job_id] = job_info

            # Trigger job.started webhook
            triggers = _get_webhook_triggers()
            if triggers.get("started") and user_id and user_id != "anonymous":
                try:
                    asyncio.create_task(
                        triggers["started"](
                            user_id=user_id,
                            job_id=job_id,
                            job_type=job_type,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to trigger job.started webhook: {e}")

        data = {
            "job_id": job_id,
            "progress": min(100, max(0, progress)),
            "status": "running",
        }
        if message:
            data["message"] = message
        if node_name:
            data["node_name"] = node_name

        await self.broadcast_to_user(user_id, "progress", data)
        debug_log(f"Progress: job {job_id} at {progress}% ({node_name or 'unknown'})")

    async def broadcast_job_complete(
        self,
        job_id: str,
        output_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Broadcast job completion to job owner.

        Args:
            job_id: Job/prompt ID
            output_url: URL to generated output
            metadata: Additional job metadata
        """
        job_info = self.job_ownership.get(job_id)
        if not job_info:
            debug_log(f"Job {job_id} not registered, cannot broadcast completion")
            return

        user_id = job_info.get("user_id")
        job_type = job_info.get("job_type", "generation")
        started_at = job_info.get("started_at")

        # Calculate processing time
        processing_time = None
        if started_at:
            import time

            processing_time = time.time() - started_at

        data = {
            "job_id": job_id,
            "status": "completed",
            "progress": 100,
        }
        if output_url:
            data["output_url"] = output_url
        if metadata:
            data["metadata"] = metadata

        await self.broadcast_to_user(user_id, "job_complete", data)
        logger.info(f"✅ Job complete: {job_id}")

        # Trigger job.completed webhook
        triggers = _get_webhook_triggers()
        if triggers.get("completed") and user_id and user_id != "anonymous":
            try:
                asyncio.create_task(
                    triggers["completed"](
                        user_id=user_id,
                        job_id=job_id,
                        job_type=job_type,
                        output_url=output_url,
                        processing_time_seconds=processing_time,
                        metadata=metadata,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to trigger job.completed webhook: {e}")

        # Trigger email notification (fire-and-forget)
        if user_id and user_id != "anonymous":
            try:
                from email_service import notify_job_completed

                asyncio.create_task(
                    notify_job_completed(
                        user_id=user_id,
                        job_id=job_id,
                        job_type=job_type,
                        output_url=output_url,
                        processing_time_seconds=processing_time,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to trigger email notification: {e}")

        self.unregister_job(job_id)

    async def broadcast_job_failed(
        self, job_id: str, error: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Broadcast job failure to job owner.

        Args:
            job_id: Job/prompt ID
            error: Error message
            metadata: Additional job metadata
        """
        job_info = self.job_ownership.get(job_id)
        if not job_info:
            debug_log(f"Job {job_id} not registered, cannot broadcast failure")
            return

        user_id = job_info.get("user_id")
        job_type = job_info.get("job_type", "generation")

        data = {
            "job_id": job_id,
            "status": "failed",
            "error": error,
        }
        if metadata:
            data["metadata"] = metadata

        await self.broadcast_to_user(user_id, "job_failed", data)
        logger.error(f"❌ Job failed: {job_id} - {error}")

        # Trigger job.failed webhook
        triggers = _get_webhook_triggers()
        if triggers.get("failed") and user_id and user_id != "anonymous":
            try:
                asyncio.create_task(
                    triggers["failed"](
                        user_id=user_id,
                        job_id=job_id,
                        job_type=job_type,
                        error=error,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to trigger job.failed webhook: {e}")

        # Trigger email notification for failure (fire-and-forget)
        if user_id and user_id != "anonymous":
            try:
                from email_service import notify_job_failed

                asyncio.create_task(
                    notify_job_failed(
                        user_id=user_id,
                        job_id=job_id,
                        job_type=job_type,
                        error=error,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to trigger failure email notification: {e}")

        self.unregister_job(job_id)

    @staticmethod
    def _format_eta(seconds: int) -> str:
        """Format ETA seconds as human-readable string"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"


# Global WebSocket manager instance
ws_manager = WebSocketManager()
