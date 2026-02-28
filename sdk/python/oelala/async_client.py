"""Async Oelala API client."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Union

import httpx

from oelala.client import DEFAULT_BASE_URL, DEFAULT_TIMEOUT, SDK_VERSION
from oelala.exceptions import (
    AuthenticationError,
    InsufficientCreditsError,
    NotFoundError,
    OelalaError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ValidationError,
)
from oelala.models import (
    CreditsResponse,
    DeepHealthResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    JobStatus,
)


class AsyncOelalaClient:
    """Async client for the Oelala AI generation API.

    Usage::

        import asyncio
        from oelala import AsyncOelalaClient

        async def main():
            async with AsyncOelalaClient(api_key="oelala_your_key") as client:
                job = await client.generate(
                    type="text-to-video",
                    prompt="a timelapse of flowers blooming",
                    duration_seconds=10,
                )
                result = await client.wait_for_job(job.job_id)
                if result.succeeded:
                    await client.download(job.job_id, "flowers.mp4")

        asyncio.run(main())
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        if not api_key or not api_key.startswith("oelala_"):
            raise ValueError("API key must start with 'oelala_'")

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "X-API-Key": api_key,
                "User-Agent": f"oelala-python/{SDK_VERSION}",
                "Accept": "application/json",
            },
            timeout=timeout,
        )

    async def __aenter__(self) -> "AsyncOelalaClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        await self._client.aclose()

    # ── API Methods ──────────────────────────────────────────────

    async def generate(
        self,
        type: str,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        cfg: Optional[float] = None,
        seed: Optional[int] = None,
        duration_seconds: Optional[int] = None,
        image_url: Optional[str] = None,
    ) -> GenerateResponse:
        """Submit a generation job (async version)."""
        req = GenerateRequest(
            type=type,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
            duration_seconds=duration_seconds,
            image_url=image_url,
        )
        data = await self._post("/api/v1/generate", json=req.to_dict())
        return GenerateResponse(**data)

    async def get_job(self, job_id: str) -> JobStatus:
        """Get the current status of a generation job (async version)."""
        data = await self._get(f"/api/v1/jobs/{job_id}")
        return JobStatus(**data)

    async def wait_for_job(
        self,
        job_id: str,
        *,
        poll_interval: float = 5.0,
        timeout: float = 600.0,
        on_progress: Optional[Any] = None,
    ) -> JobStatus:
        """Poll a job until it reaches a terminal state (async version).

        Args:
            job_id: The job ID to monitor.
            poll_interval: Seconds between status checks (default 5).
            timeout: Maximum wait time in seconds (default 600).
            on_progress: Optional async callback ``async fn(JobStatus)``.

        Returns:
            Final :class:`JobStatus` when job is completed or failed.

        Raises:
            TimeoutError: If the job doesn't finish within *timeout* seconds.
        """
        elapsed = 0.0
        while True:
            status = await self.get_job(job_id)
            if on_progress:
                result = on_progress(status)
                if asyncio.iscoroutine(result):
                    await result
            if status.is_done:
                return status
            if elapsed + poll_interval > timeout:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {timeout}s (last status: {status.status})"
                )
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

    async def download(self, job_id: str, dest: Union[str, Path, BinaryIO]) -> Path:
        """Download the result of a completed job (async version)."""
        response = await self._client.get(f"/api/v1/jobs/{job_id}/download")
        self._raise_for_status(response)

        if isinstance(dest, (str, Path)):
            path = Path(dest)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            return path
        else:
            dest.write(response.content)
            return Path("<stream>")

    async def get_credits(self) -> CreditsResponse:
        """Get current credit balance (async version)."""
        data = await self._get("/api/v1/credits")
        return CreditsResponse(**data)

    async def health(self) -> HealthResponse:
        """Check API health (async, no auth required)."""
        resp = await self._client.get("/api/v1/health")
        self._raise_for_status(resp)
        return HealthResponse(**resp.json())

    async def health_deep(self) -> DeepHealthResponse:
        """Deep health check (async, no auth required)."""
        resp = await self._client.get("/health/deep")
        self._raise_for_status(resp)
        return DeepHealthResponse(**resp.json())

    # ── Convenience Methods ──────────────────────────────────────

    async def text_to_image(self, prompt: str, **kwargs: Any) -> GenerateResponse:
        """Shortcut for text-to-image generation."""
        return await self.generate(type="text-to-image", prompt=prompt, **kwargs)

    async def text_to_video(self, prompt: str, **kwargs: Any) -> GenerateResponse:
        """Shortcut for text-to-video generation."""
        return await self.generate(type="text-to-video", prompt=prompt, **kwargs)

    async def image_to_video(self, prompt: str, *, image_url: str, **kwargs: Any) -> GenerateResponse:
        """Shortcut for image-to-video generation."""
        return await self.generate(type="image-to-video", prompt=prompt, image_url=image_url, **kwargs)

    # ── Internal ─────────────────────────────────────────────────

    async def _get(self, path: str) -> Dict[str, Any]:
        resp = await self._client.get(path)
        self._raise_for_status(resp)
        return resp.json()

    async def _post(self, path: str, json: Any = None) -> Dict[str, Any]:
        resp = await self._client.post(path, json=json)
        self._raise_for_status(resp)
        return resp.json()

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        body = None
        try:
            body = resp.json()
        except Exception:
            pass
        detail = (body or {}).get("detail", resp.text)
        msg = f"HTTP {resp.status_code}: {detail}"

        if resp.status_code == 401:
            raise AuthenticationError(msg, status_code=401, body=body)
        elif resp.status_code == 402:
            raise InsufficientCreditsError(msg, status_code=402, body=body)
        elif resp.status_code == 404:
            raise NotFoundError(msg, status_code=404, body=body)
        elif resp.status_code == 422:
            raise ValidationError(msg, status_code=422, body=body)
        elif resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            raise RateLimitError(
                msg,
                status_code=429,
                body=body,
                retry_after=float(retry_after) if retry_after else None,
            )
        elif resp.status_code >= 500:
            raise ServerError(msg, status_code=resp.status_code, body=body)
        else:
            raise OelalaError(msg, status_code=resp.status_code, body=body)
