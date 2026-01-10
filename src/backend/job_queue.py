#!/usr/bin/env python3
"""
Job Queue Manager with Position Tracking and ETA Estimation
Integrates with ComfyUI queue and broadcasts real-time updates
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Any
from collections import deque
import httpx

logger = logging.getLogger(__name__)

DEBUG_ENABLED = False


def debug_log(message: str):
    """Emit debug logs when DEBUG_ENABLED is true."""
    if DEBUG_ENABLED:
        logger.info(f"🐛 {message}")


class JobQueueManager:
    """
    Manages job queue state and position tracking.
    Polls ComfyUI queue and broadcasts updates via WebSocket.
    """

    def __init__(self, comfyui_host: str = "localhost", comfyui_port: int = 8188):
        self.comfyui_url = f"http://{comfyui_host}:{comfyui_port}"
        # Job metadata: prompt_id -> {user_id, created_at, type, ...}
        self.jobs: Dict[str, Dict[str, Any]] = {}
        # Historical completion times for ETA estimation (last 20 jobs)
        self.completion_times: deque = deque(maxlen=20)
        # Last known queue state for change detection
        self.last_queue_state: Dict[str, int] = {}  # prompt_id -> position
        # Polling task
        self._poll_task: Optional[asyncio.Task] = None
        self._running = False

    def register_job(
        self,
        prompt_id: str,
        user_id: Optional[str] = None,
        job_type: str = "generation",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Register a new job in the queue.

        Args:
            prompt_id: ComfyUI prompt ID
            user_id: User who submitted the job
            job_type: Type of job (generation, upscale, etc.)
            metadata: Additional metadata
        """
        self.jobs[prompt_id] = {
            "prompt_id": prompt_id,
            "user_id": user_id or "anonymous",
            "job_type": job_type,
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "status": "queued",
            "metadata": metadata or {},
        }
        debug_log(f"Registered job {prompt_id} for user {user_id}")

    def get_job(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get job metadata"""
        return self.jobs.get(prompt_id)

    def update_job_status(self, prompt_id: str, status: str, **kwargs):
        """Update job status and additional fields"""
        if prompt_id in self.jobs:
            self.jobs[prompt_id]["status"] = status
            self.jobs[prompt_id].update(kwargs)
            debug_log(f"Updated job {prompt_id} status to {status}")

    def complete_job(self, prompt_id: str):
        """Mark job as completed and record completion time for ETA"""
        if prompt_id in self.jobs:
            job = self.jobs[prompt_id]
            job["status"] = "completed"
            job["completed_at"] = time.time()

            # Calculate execution time for ETA estimation
            if job.get("started_at"):
                execution_time = job["completed_at"] - job["started_at"]
                self.completion_times.append(execution_time)
                debug_log(
                    f"Job {prompt_id} completed in {execution_time:.1f}s (avg: {self.get_average_execution_time():.1f}s)"
                )

    def fail_job(self, prompt_id: str, error: str):
        """Mark job as failed"""
        if prompt_id in self.jobs:
            self.jobs[prompt_id]["status"] = "failed"
            self.jobs[prompt_id]["error"] = error
            self.jobs[prompt_id]["completed_at"] = time.time()
            debug_log(f"Job {prompt_id} failed: {error}")

    def get_average_execution_time(self) -> float:
        """Get average execution time from recent jobs (for ETA estimation)"""
        if not self.completion_times:
            return 120.0  # Default 2 minutes if no history
        return sum(self.completion_times) / len(self.completion_times)

    def estimate_eta(self, queue_position: int) -> int:
        """
        Estimate time until job starts based on queue position and historical data.

        Args:
            queue_position: Position in queue (1-based)

        Returns:
            Estimated seconds until job starts
        """
        if queue_position <= 0:
            return 0
        avg_time = self.get_average_execution_time()
        # Account for jobs ahead in queue
        return int(avg_time * queue_position)

    async def get_comfyui_queue(self) -> Optional[Dict[str, Any]]:
        """Fetch current queue state from ComfyUI"""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.comfyui_url}/queue", timeout=5)
                if resp.status_code == 200:
                    return resp.json()
                else:
                    logger.warning(f"ComfyUI queue request failed: {resp.status_code}")
                    return None
        except Exception as e:
            logger.warning(f"Failed to get ComfyUI queue: {e}")
            return None

    async def poll_queue_updates(self, ws_manager):
        """
        Poll ComfyUI queue and broadcast position updates.
        Should be called periodically (every 1-2 seconds).

        Args:
            ws_manager: WebSocketManager instance for broadcasting
        """
        queue_data = await self.get_comfyui_queue()
        if not queue_data:
            return

        current_state = {}
        running_jobs = queue_data.get("queue_running", [])
        pending_jobs = queue_data.get("queue_pending", [])

        # Process running jobs (position 0)
        for item in running_jobs:
            if len(item) >= 2:
                prompt_id = item[1]
                current_state[prompt_id] = 0

                # Update job status if we're tracking it
                if prompt_id in self.jobs:
                    job = self.jobs[prompt_id]
                    if job["status"] != "running":
                        job["status"] = "running"
                        if not job.get("started_at"):
                            job["started_at"] = time.time()
                        debug_log(f"Job {prompt_id} started running")

                    # Broadcast queue update
                    await ws_manager.broadcast_queue_update(
                        job_id=prompt_id,
                        queue_position=0,
                        total_pending=len(pending_jobs),
                        eta_seconds=0,
                    )

        # Process pending jobs (position 1, 2, 3, ...)
        for idx, item in enumerate(pending_jobs):
            if len(item) >= 2:
                prompt_id = item[1]
                position = idx + 1
                current_state[prompt_id] = position

                # Update job status if we're tracking it
                if prompt_id in self.jobs:
                    job = self.jobs[prompt_id]
                    if job["status"] != "queued":
                        job["status"] = "queued"

                    # Broadcast only if position changed or first time
                    last_position = self.last_queue_state.get(prompt_id, -1)
                    if last_position != position:
                        eta = self.estimate_eta(position)
                        await ws_manager.broadcast_queue_update(
                            job_id=prompt_id,
                            queue_position=position,
                            total_pending=len(pending_jobs),
                            eta_seconds=eta,
                        )
                        debug_log(
                            f"Queue position change: {prompt_id} from {last_position} to {position} (ETA: {eta}s)"
                        )

        # Detect completed/failed jobs (no longer in queue)
        for prompt_id in list(self.last_queue_state.keys()):
            if prompt_id not in current_state and prompt_id in self.jobs:
                # Job disappeared from queue - check history
                await self._check_job_completion(prompt_id, ws_manager)

        self.last_queue_state = current_state

    async def _check_job_completion(self, prompt_id: str, ws_manager):
        """Check if a job completed successfully via ComfyUI history"""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.comfyui_url}/history/{prompt_id}", timeout=5
                )
                if resp.status_code == 200:
                    history = resp.json().get(prompt_id, {})
                    if history:
                        # Job completed successfully
                        self.complete_job(prompt_id)

                        # Extract output URL if available
                        output_url = self._extract_output_url(history)

                        await ws_manager.broadcast_job_complete(
                            job_id=prompt_id,
                            output_url=output_url,
                            metadata={"history": history},
                        )
                        # Clean up job after completion
                        if prompt_id in self.jobs:
                            del self.jobs[prompt_id]
                    else:
                        # Job failed or cancelled
                        self.fail_job(prompt_id, "Job cancelled or failed")
                        await ws_manager.broadcast_job_failed(
                            job_id=prompt_id,
                            error="Job cancelled or not found in history",
                        )
                        if prompt_id in self.jobs:
                            del self.jobs[prompt_id]
        except Exception as e:
            logger.warning(f"Failed to check job completion for {prompt_id}: {e}")

    def _extract_output_url(self, history: Dict[str, Any]) -> Optional[str]:
        """Extract output URL from ComfyUI history"""
        try:
            outputs = history.get("outputs", {})
            for node_id, node_output in outputs.items():
                # Video output
                if "gifs" in node_output:
                    for gif in node_output["gifs"]:
                        if gif.get("type") == "output":
                            return f"/comfyui-output/{gif['filename']}"
                # Image output
                if "images" in node_output:
                    for img in node_output["images"]:
                        if img.get("type") == "output":
                            return f"/comfyui-output/{img['filename']}"
                # Audio output
                if "audio" in node_output:
                    for audio in node_output["audio"]:
                        if audio.get("type") == "output":
                            return f"/comfyui-output/{audio['filename']}"
            return None
        except Exception as e:
            logger.warning(f"Failed to extract output URL: {e}")
            return None

    async def start_polling(self, ws_manager, interval: float = 2.0):
        """
        Start background polling task.

        Args:
            ws_manager: WebSocketManager instance
            interval: Polling interval in seconds (default 2s)
        """
        if self._running:
            logger.warning("Queue polling already running")
            return

        self._running = True
        logger.info(f"🔄 Starting queue polling (interval: {interval}s)")

        async def poll_loop():
            while self._running:
                try:
                    await self.poll_queue_updates(ws_manager)
                except Exception as e:
                    logger.error(f"Error in queue polling: {e}")
                await asyncio.sleep(interval)

        self._poll_task = asyncio.create_task(poll_loop())

    async def stop_polling(self):
        """Stop background polling task"""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        logger.info("🛑 Stopped queue polling")


# Global job queue manager instance
job_queue_manager = JobQueueManager()
