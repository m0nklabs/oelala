#!/usr/bin/env python3
"""
RunPod Serverless Client for Oelala Backend

Routes heavy ComfyUI jobs to RunPod cloud GPUs when local VRAM is insufficient
or when burst capacity is needed.

Usage:
    from runpod_client import RunPodClient, get_runpod_client

    client = get_runpod_client()
    if client.is_available():
        job = await client.submit_workflow(workflow_json, webhook_url="...")
        status = await client.get_job_status(job["id"])
"""

import os
import asyncio
import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

import httpx

logger = logging.getLogger(__name__)

# Debug flag — controlled by env var
DEBUG = os.getenv("RUNPOD_DEBUG", "false").lower() in ("true", "1", "yes")


def debug_log(msg: str):
    if DEBUG:
        logger.info(f"🐛 [RunPod] {msg}")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUNPOD_API_BASE = "https://api.runpod.ai/v2"
RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"

# Job status mapping from RunPod → internal
class RunPodJobStatus(str, Enum):
    IN_QUEUE = "IN_QUEUE"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"


@dataclass
class RunPodEndpoint:
    """Represents a deployed RunPod serverless endpoint."""
    id: str
    name: str
    gpu_type: str = "UNKNOWN"
    workers_min: int = 0
    workers_max: int = 1
    idle_timeout: int = 5


@dataclass
class RunPodJob:
    """Represents a submitted RunPod job."""
    id: str
    endpoint_id: str
    status: RunPodJobStatus = RunPodJobStatus.IN_QUEUE
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    execution_time_ms: Optional[int] = None


# ---------------------------------------------------------------------------
# RunPod Client
# ---------------------------------------------------------------------------

class RunPodClient:
    """
    Async client for RunPod Serverless API.

    Manages endpoints, submits ComfyUI workflows, and polls for results.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY", "")
        self._http: Optional[httpx.AsyncClient] = None
        # Endpoint cache — endpoint_id -> RunPodEndpoint
        self._endpoints: Dict[str, RunPodEndpoint] = {}
        # Active jobs — job_id -> RunPodJob
        self._active_jobs: Dict[str, RunPodJob] = {}
        # Default endpoint (set via env var, configure, or auto-detected)
        self.default_endpoint_id: Optional[str] = os.getenv("RUNPOD_ENDPOINT_ID")

    @property
    def http(self) -> httpx.AsyncClient:
        """Lazy-init async HTTP client."""
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._http

    def is_available(self) -> bool:
        """Check if RunPod is configured (API key present)."""
        return bool(self.api_key and len(self.api_key) > 10)

    def has_endpoint(self) -> bool:
        """Check if at least one endpoint is configured."""
        return bool(self.default_endpoint_id or self._endpoints)

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def get_account_info(self) -> Dict[str, Any]:
        """Get RunPod account info (balance, spend, etc.)."""
        query = '{ myself { id email clientBalance spendLimit currentSpendPerHr } }'
        resp = await self.http.post(RUNPOD_GRAPHQL, json={"query": query})
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            raise RuntimeError(f"RunPod GraphQL error: {data['errors']}")
        return data.get("data", {}).get("myself", {})

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    async def list_endpoints(self) -> List[Dict[str, Any]]:
        """List all serverless endpoints on the account."""
        query = """
        {
            myself {
                endpoints {
                    id
                    name
                    gpuIds
                    idleTimeout
                    workersMin
                    workersMax
                    templateId
                }
            }
        }
        """
        resp = await self.http.post(RUNPOD_GRAPHQL, json={"query": query})
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            raise RuntimeError(f"RunPod GraphQL error: {data['errors']}")
        endpoints = data.get("data", {}).get("myself", {}).get("endpoints", [])

        # Cache endpoints
        for ep in endpoints:
            self._endpoints[ep["id"]] = RunPodEndpoint(
                id=ep["id"],
                name=ep.get("name", ""),
                gpu_type=ep.get("gpuIds", "UNKNOWN"),
                workers_min=ep.get("workersMin", 0),
                workers_max=ep.get("workersMax", 1),
                idle_timeout=ep.get("idleTimeout", 5),
            )

        # Auto-set default if not set
        if not self.default_endpoint_id and endpoints:
            self.default_endpoint_id = endpoints[0]["id"]
            debug_log(f"Auto-selected default endpoint: {self.default_endpoint_id}")

        return endpoints

    # ------------------------------------------------------------------
    # Job Submission
    # ------------------------------------------------------------------

    async def submit_workflow(
        self,
        workflow: Dict[str, Any],
        endpoint_id: Optional[str] = None,
        webhook_url: Optional[str] = None,
        extra_input: Optional[Dict[str, Any]] = None,
    ) -> RunPodJob:
        """
        Submit a ComfyUI workflow to RunPod serverless.

        Args:
            workflow: ComfyUI API-format workflow JSON
            endpoint_id: Specific endpoint (or uses default)
            webhook_url: Optional webhook for completion callback
            extra_input: Additional input fields (e.g. images as base64)

        Returns:
            RunPodJob with the job ID
        """
        ep_id = endpoint_id or self.default_endpoint_id
        if not ep_id:
            raise RuntimeError("No RunPod endpoint configured. Deploy one first.")

        payload: Dict[str, Any] = {
            "input": {
                "workflow": workflow,
                **(extra_input or {}),
            }
        }
        if webhook_url:
            payload["webhook"] = webhook_url

        url = f"{RUNPOD_API_BASE}/{ep_id}/run"
        debug_log(f"Submitting workflow to {url}")

        resp = await self.http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        job = RunPodJob(
            id=data.get("id", ""),
            endpoint_id=ep_id,
            status=RunPodJobStatus(data.get("status", "IN_QUEUE")),
        )
        self._active_jobs[job.id] = job
        debug_log(f"Job submitted: {job.id} status={job.status}")
        return job

    async def submit_workflow_sync(
        self,
        workflow: Dict[str, Any],
        endpoint_id: Optional[str] = None,
        extra_input: Optional[Dict[str, Any]] = None,
        timeout: int = 600,
    ) -> RunPodJob:
        """
        Submit workflow and wait for completion (blocking).
        Use /runsync endpoint for shorter jobs (< 30s expected).
        Falls back to polling for longer jobs.

        Args:
            workflow: ComfyUI API-format workflow
            endpoint_id: Specific endpoint
            extra_input: Additional input
            timeout: Max wait time in seconds

        Returns:
            RunPodJob with output
        """
        ep_id = endpoint_id or self.default_endpoint_id
        if not ep_id:
            raise RuntimeError("No RunPod endpoint configured.")

        payload: Dict[str, Any] = {
            "input": {
                "workflow": workflow,
                **(extra_input or {}),
            }
        }

        url = f"{RUNPOD_API_BASE}/{ep_id}/runsync"
        debug_log(f"Submitting sync workflow to {url} (timeout={timeout}s)")

        resp = await self.http.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status", "IN_QUEUE")
        job = RunPodJob(
            id=data.get("id", ""),
            endpoint_id=ep_id,
            status=RunPodJobStatus(status),
            output=data.get("output"),
            execution_time_ms=data.get("executionTime"),
        )

        # If still in queue/progress, fall back to polling
        if status not in ("COMPLETED", "FAILED"):
            debug_log(f"Sync request returned {status}, switching to polling")
            return await self.poll_job(job.id, ep_id, timeout=timeout)

        self._active_jobs[job.id] = job
        return job

    # ------------------------------------------------------------------
    # Job Status & Polling
    # ------------------------------------------------------------------

    async def get_job_status(
        self,
        job_id: str,
        endpoint_id: Optional[str] = None,
    ) -> RunPodJob:
        """Get current status of a RunPod job."""
        ep_id = endpoint_id or self.default_endpoint_id
        if not ep_id:
            raise RuntimeError("No endpoint configured.")

        url = f"{RUNPOD_API_BASE}/{ep_id}/status/{job_id}"
        resp = await self.http.get(url)
        resp.raise_for_status()
        data = resp.json()

        status = RunPodJobStatus(data.get("status", "IN_QUEUE"))
        job = self._active_jobs.get(job_id, RunPodJob(id=job_id, endpoint_id=ep_id))
        job.status = status
        job.output = data.get("output")
        job.execution_time_ms = data.get("executionTime")

        if status == RunPodJobStatus.FAILED:
            job.error = data.get("error", "Unknown error")
        if status in (RunPodJobStatus.COMPLETED, RunPodJobStatus.FAILED):
            job.completed_at = time.time()

        self._active_jobs[job_id] = job
        return job

    async def poll_job(
        self,
        job_id: str,
        endpoint_id: Optional[str] = None,
        interval: float = 3.0,
        timeout: int = 600,
    ) -> RunPodJob:
        """
        Poll a job until completion or timeout.

        Args:
            job_id: RunPod job ID
            endpoint_id: Endpoint that owns the job
            interval: Seconds between polls
            timeout: Max wait time

        Returns:
            RunPodJob with final status
        """
        start = time.time()
        while (time.time() - start) < timeout:
            job = await self.get_job_status(job_id, endpoint_id)
            debug_log(f"Poll {job_id}: status={job.status} elapsed={time.time()-start:.0f}s")

            if job.status in (
                RunPodJobStatus.COMPLETED,
                RunPodJobStatus.FAILED,
                RunPodJobStatus.CANCELLED,
                RunPodJobStatus.TIMED_OUT,
            ):
                return job

            await asyncio.sleep(interval)

        # Timeout
        job = self._active_jobs.get(job_id, RunPodJob(id=job_id, endpoint_id=endpoint_id or ""))
        job.status = RunPodJobStatus.TIMED_OUT
        job.error = f"Polling timed out after {timeout}s"
        return job

    async def cancel_job(
        self,
        job_id: str,
        endpoint_id: Optional[str] = None,
    ) -> bool:
        """Cancel a running/queued job."""
        ep_id = endpoint_id or self.default_endpoint_id
        if not ep_id:
            return False

        url = f"{RUNPOD_API_BASE}/{ep_id}/cancel/{job_id}"
        try:
            resp = await self.http.post(url)
            resp.raise_for_status()
            debug_log(f"Cancelled job {job_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel job {job_id}: {e}")
            return False

    # ------------------------------------------------------------------
    # Health & Metrics
    # ------------------------------------------------------------------

    async def get_endpoint_health(
        self,
        endpoint_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get health/worker status of an endpoint."""
        ep_id = endpoint_id or self.default_endpoint_id
        if not ep_id:
            return {"status": "no_endpoint"}

        url = f"{RUNPOD_API_BASE}/{ep_id}/health"
        try:
            resp = await self.http.get(url)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self):
        """Close the HTTP client."""
        if self._http and not self._http.is_closed:
            await self._http.aclose()
            self._http = None

    def get_active_jobs(self) -> List[RunPodJob]:
        """Get all tracked active jobs."""
        return [
            j for j in self._active_jobs.values()
            if j.status in (RunPodJobStatus.IN_QUEUE, RunPodJobStatus.IN_PROGRESS)
        ]

    def get_job_stats(self) -> Dict[str, int]:
        """Get job count by status."""
        stats: Dict[str, int] = {}
        for job in self._active_jobs.values():
            stats[job.status.value] = stats.get(job.status.value, 0) + 1
        return stats


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_client: Optional[RunPodClient] = None


def get_runpod_client() -> RunPodClient:
    """Get or create the global RunPod client singleton."""
    global _client
    if _client is None:
        _client = RunPodClient()
        if _client.is_available():
            logger.info("✅ RunPod client initialized (API key found)")
        else:
            logger.info("⚠️ RunPod client: no API key configured (RUNPOD_API_KEY)")
    return _client
