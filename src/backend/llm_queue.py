#!/usr/bin/env python3
"""
LLM Request Queue

Serializes LLM prompt enhancement requests to prevent concurrent VRAM usage.
Jobs are processed one at a time in the background, results delivered via polling.

Architecture:
  - Frontend POSTs to /generate-prompt → gets back {status: "queued", job_id}
  - Frontend polls /llm-job/{job_id} every 1-2s for status/result
  - Backend processes jobs FIFO with asyncio.Lock (one at a time)
  - VRAM coordination: waits for ComfyUI idle, frees VRAM, runs LLM, done
"""

import asyncio
import logging
import os
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)
DEBUG_ENABLED = os.getenv("OELALA_DEBUG", "0") == "1"


def _debug(msg: str) -> None:
    if DEBUG_ENABLED:
        logger.info(f"🐛 [llm_queue] {msg}")


class LLMJob:
    """Represents a single LLM prompt enhancement job."""

    __slots__ = (
        "job_id",
        "status",
        "created_at",
        "started_at",
        "completed_at",
        "queue_position",
        "request_data",
        "result",
        "error",
        "user_id",
    )

    def __init__(
        self,
        job_id: str,
        request_data: Dict[str, Any],
        user_id: Optional[str] = None,
    ):
        self.job_id = job_id
        self.status = "queued"  # queued → processing → completed | failed
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.queue_position: int = 0
        self.request_data = request_data
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.user_id = user_id

    def to_dict(self) -> Dict[str, Any]:
        """Serialize job state for API response."""
        d: Dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status,
            "queue_position": self.queue_position,
            "created_at": self.created_at,
        }
        if self.started_at:
            d["started_at"] = self.started_at
        if self.completed_at:
            d["completed_at"] = self.completed_at
            d["processing_time"] = round(self.completed_at - (self.started_at or self.created_at), 2)
        if self.result is not None:
            d["result"] = self.result
        if self.error is not None:
            d["error"] = self.error
        return d


class LLMQueueManager:
    """
    Manages LLM prompt enhancement jobs with FIFO queue + serialization lock.

    - One job processes at a time (asyncio.Lock)
    - Jobs are tracked in an OrderedDict for position calculation
    - Completed jobs are kept for 5 minutes for polling, then cleaned up
    """

    # How long to keep completed/failed jobs for polling (seconds)
    RESULT_TTL = 300  # 5 minutes

    def __init__(self):
        self._lock = asyncio.Lock()
        self._jobs: OrderedDict[str, LLMJob] = OrderedDict()
        self._pending_queue: list[str] = []  # job_ids in FIFO order
        self._cleanup_task: Optional[asyncio.Task] = None

    def submit(
        self,
        request_data: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> LLMJob:
        """
        Submit a new LLM job to the queue. Returns the job immediately.

        Args:
            request_data: The parsed prompt generation request
            user_id: Optional user ID for tracking

        Returns:
            LLMJob with status="queued" and queue_position set
        """
        job_id = f"llm_{uuid.uuid4().hex[:12]}"
        job = LLMJob(job_id=job_id, request_data=request_data, user_id=user_id)

        # Calculate queue position (number of pending + processing jobs)
        job.queue_position = len(self._pending_queue)
        self._jobs[job_id] = job
        self._pending_queue.append(job_id)

        logger.info(
            f"📝 LLM job {job_id} queued (position: {job.queue_position}, "
            f"total pending: {len(self._pending_queue)})"
        )
        _debug(f"Queue state: {[j for j in self._pending_queue]}")

        return job

    def get_job(self, job_id: str) -> Optional[LLMJob]:
        """Get job by ID (returns None if not found or expired)."""
        return self._jobs.get(job_id)

    def _update_positions(self) -> None:
        """Recalculate queue positions for all pending jobs."""
        for idx, jid in enumerate(self._pending_queue):
            if jid in self._jobs:
                self._jobs[jid].queue_position = idx

    async def process_next(self, processor_fn) -> None:
        """
        Process the next job in the queue using the provided async function.
        Called from the background worker loop.

        Args:
            processor_fn: async (request_data: dict) -> Optional[dict]
                          Returns result dict on success, None on failure.
        """
        if not self._pending_queue:
            return

        async with self._lock:
            if not self._pending_queue:
                return

            job_id = self._pending_queue.pop(0)
            job = self._jobs.get(job_id)
            if not job:
                return

            # Update positions for remaining jobs
            self._update_positions()

            # Mark as processing
            job.status = "processing"
            job.started_at = time.time()
            job.queue_position = -1  # -1 = currently processing
            logger.info(f"⚡ LLM job {job_id} processing...")

            try:
                result = await processor_fn(job.request_data)
                if result is not None:
                    job.status = "completed"
                    job.result = result
                    logger.info(
                        f"✅ LLM job {job_id} completed in "
                        f"{time.time() - job.started_at:.1f}s"
                    )
                else:
                    job.status = "failed"
                    job.error = "LLM generation returned no result (Guardian unavailable?)"
                    logger.warning(f"❌ LLM job {job_id} failed: no result")
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)
                logger.error(f"❌ LLM job {job_id} failed: {exc}")
            finally:
                job.completed_at = time.time()

    @property
    def pending_count(self) -> int:
        """Number of jobs waiting to be processed."""
        return len(self._pending_queue)

    @property
    def has_pending(self) -> bool:
        """Whether there are jobs waiting."""
        return len(self._pending_queue) > 0

    async def start_worker(self, processor_fn, poll_interval: float = 0.5) -> None:
        """
        Start the background worker loop that processes jobs.

        Args:
            processor_fn: async function that processes a job's request_data
            poll_interval: How often to check for new jobs (seconds)
        """
        logger.info("🔄 LLM queue worker started")

        # Also start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        while True:
            try:
                if self.has_pending:
                    await self.process_next(processor_fn)
                else:
                    await asyncio.sleep(poll_interval)
            except asyncio.CancelledError:
                logger.info("🛑 LLM queue worker stopped")
                break
            except Exception as exc:
                logger.error(f"❌ LLM queue worker error: {exc}")
                await asyncio.sleep(1.0)

    async def _cleanup_loop(self) -> None:
        """Periodically remove old completed/failed jobs."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                now = time.time()
                expired = [
                    jid
                    for jid, job in self._jobs.items()
                    if job.status in ("completed", "failed")
                    and job.completed_at
                    and (now - job.completed_at) > self.RESULT_TTL
                ]
                for jid in expired:
                    del self._jobs[jid]
                if expired:
                    _debug(f"Cleaned up {len(expired)} expired LLM jobs")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(f"LLM queue cleanup error: {exc}")

    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status for debugging/admin."""
        return {
            "pending": len(self._pending_queue),
            "processing": sum(
                1 for j in self._jobs.values() if j.status == "processing"
            ),
            "completed": sum(
                1 for j in self._jobs.values() if j.status == "completed"
            ),
            "failed": sum(
                1 for j in self._jobs.values() if j.status == "failed"
            ),
            "total_tracked": len(self._jobs),
        }


# Module-level singleton
llm_queue_manager = LLMQueueManager()
