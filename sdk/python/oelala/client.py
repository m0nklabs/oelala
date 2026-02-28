"""Synchronous Oelala API client."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Union

import httpx

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

DEFAULT_BASE_URL = "https://api.oelala.xyz"
DEFAULT_TIMEOUT = 30.0
SDK_VERSION = "0.1.0"


class OelalaClient:
    """Synchronous client for the Oelala AI generation API.

    Usage::

        from oelala import OelalaClient

        client = OelalaClient(api_key="oelala_your_key_here")

        # Generate an image
        job = client.generate(
            type="text-to-image",
            prompt="a cat riding a unicorn through space",
        )
        print(f"Job {job.job_id} started, estimated {job.estimated_time_seconds}s")

        # Wait for completion
        result = client.wait_for_job(job.job_id)
        if result.succeeded:
            client.download(job.job_id, "output.png")
            print("Downloaded!")
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """Initialize the Oelala client.

        Args:
            api_key: Your Oelala API key (starts with ``oelala_``).
            base_url: API base URL. Defaults to ``https://api.oelala.xyz``.
            timeout: HTTP request timeout in seconds. Defaults to 30.
        """
        if not api_key or not api_key.startswith("oelala_"):
            raise ValueError("API key must start with 'oelala_'")

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={
                "X-API-Key": api_key,
                "User-Agent": f"oelala-python/{SDK_VERSION}",
                "Accept": "application/json",
            },
            timeout=timeout,
        )

    def __enter__(self) -> "OelalaClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client and release resources."""
        self._client.close()

    # ── API Methods ──────────────────────────────────────────────

    def generate(
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
        """Submit a generation job.

        Args:
            type: One of ``"text-to-image"``, ``"text-to-video"``, ``"image-to-video"``.
            prompt: Text description of the desired output.
            negative_prompt: What to avoid.
            width: Output width (256-2048).
            height: Output height (256-2048).
            steps: Diffusion steps (1-100).
            cfg: Guidance scale (1.0-20.0).
            seed: Random seed (-1 for random).
            duration_seconds: Video duration (1-30).
            image_url: Source image URL (required for image-to-video).

        Returns:
            :class:`GenerateResponse` with the job ID and status.

        Raises:
            InsufficientCreditsError: Not enough credits.
            ValidationError: Invalid parameters.
        """
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
        data = self._post("/api/v1/generate", json=req.to_dict())
        return GenerateResponse(**data)

    def get_job(self, job_id: str) -> JobStatus:
        """Get the current status of a generation job.

        Args:
            job_id: The job ID returned from :meth:`generate`.

        Returns:
            :class:`JobStatus` with current progress and result info.
        """
        data = self._get(f"/api/v1/jobs/{job_id}")
        return JobStatus(**data)

    def wait_for_job(
        self,
        job_id: str,
        *,
        poll_interval: float = 5.0,
        timeout: float = 600.0,
        on_progress: Optional[Any] = None,
    ) -> JobStatus:
        """Poll a job until it reaches a terminal state.

        Args:
            job_id: The job ID to monitor.
            poll_interval: Seconds between status checks (default 5).
            timeout: Maximum wait time in seconds (default 600).
            on_progress: Optional callback ``fn(JobStatus)`` called on each poll.

        Returns:
            Final :class:`JobStatus` when job is completed or failed.

        Raises:
            TimeoutError: If the job doesn't finish within *timeout* seconds.
        """
        start = time.monotonic()
        while True:
            status = self.get_job(job_id)
            if on_progress:
                on_progress(status)
            if status.is_done:
                return status
            elapsed = time.monotonic() - start
            if elapsed + poll_interval > timeout:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {timeout}s (last status: {status.status})"
                )
            time.sleep(poll_interval)

    def download(self, job_id: str, dest: Union[str, Path, BinaryIO]) -> Path:
        """Download the result of a completed job.

        Args:
            job_id: The completed job ID.
            dest: File path or writable binary stream to save to.

        Returns:
            :class:`Path` to the saved file (if *dest* was a string/Path).

        Raises:
            NotFoundError: If the job or result doesn't exist.
        """
        response = self._client.get(f"/api/v1/jobs/{job_id}/download")
        self._raise_for_status(response)

        if isinstance(dest, (str, Path)):
            path = Path(dest)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            return path
        else:
            dest.write(response.content)
            return Path("<stream>")

    def get_credits(self) -> CreditsResponse:
        """Get current credit balance.

        Returns:
            :class:`CreditsResponse` with balance and usage stats.
        """
        data = self._get("/api/v1/credits")
        return CreditsResponse(**data)

    def health(self) -> HealthResponse:
        """Check API health (no auth required).

        Returns:
            :class:`HealthResponse` with status and version.
        """
        resp = self._client.get("/api/v1/health")
        self._raise_for_status(resp)
        return HealthResponse(**resp.json())

    def health_deep(self) -> DeepHealthResponse:
        """Deep health check with service connectivity (no auth required).

        Returns:
            :class:`DeepHealthResponse` with per-service status.
        """
        resp = self._client.get("/health/deep")
        self._raise_for_status(resp)
        return DeepHealthResponse(**resp.json())

    # ── Convenience Methods ──────────────────────────────────────

    def text_to_image(self, prompt: str, **kwargs: Any) -> GenerateResponse:
        """Shortcut for generating a text-to-image job."""
        return self.generate(type="text-to-image", prompt=prompt, **kwargs)

    def text_to_video(self, prompt: str, **kwargs: Any) -> GenerateResponse:
        """Shortcut for generating a text-to-video job."""
        return self.generate(type="text-to-video", prompt=prompt, **kwargs)

    def image_to_video(self, prompt: str, *, image_url: str, **kwargs: Any) -> GenerateResponse:
        """Shortcut for generating an image-to-video job."""
        return self.generate(type="image-to-video", prompt=prompt, image_url=image_url, **kwargs)

    # ── Internal ─────────────────────────────────────────────────

    def _get(self, path: str) -> Dict[str, Any]:
        resp = self._client.get(path)
        self._raise_for_status(resp)
        return resp.json()

    def _post(self, path: str, json: Any = None) -> Dict[str, Any]:
        resp = self._client.post(path, json=json)
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
