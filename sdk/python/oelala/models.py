"""Data models for the Oelala SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class GenerateRequest:
    """Request to generate an image or video.

    Args:
        type: Generation type - "text-to-image", "text-to-video", or "image-to-video".
        prompt: Text prompt describing the desired output.
        negative_prompt: What to avoid in the output.
        width: Output width in pixels (256-2048, default 1024).
        height: Output height in pixels (256-2048, default 1024).
        steps: Number of diffusion steps (1-100, default 20).
        cfg: Classifier-free guidance scale (1.0-20.0, default 7.5).
        seed: Random seed (-1 for random).
        duration_seconds: Video duration in seconds (1-30, video types only).
        image_url: Source image URL (required for image-to-video).
    """

    type: Literal["text-to-image", "text-to-video", "image-to-video"]
    prompt: str
    negative_prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    seed: Optional[int] = None
    duration_seconds: Optional[int] = None
    image_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API request payload, omitting None values."""
        d: Dict[str, Any] = {"type": self.type, "prompt": self.prompt}
        if self.negative_prompt is not None:
            d["negative_prompt"] = self.negative_prompt
        if self.width is not None:
            d["width"] = self.width
        if self.height is not None:
            d["height"] = self.height
        if self.steps is not None:
            d["steps"] = self.steps
        if self.cfg is not None:
            d["cfg"] = self.cfg
        if self.seed is not None:
            d["seed"] = self.seed
        if self.duration_seconds is not None:
            d["duration_seconds"] = self.duration_seconds
        if self.image_url is not None:
            d["image_url"] = self.image_url
        return d


@dataclass
class GenerateResponse:
    """Response from the generate endpoint."""

    job_id: str
    status: str
    credits_used: int
    estimated_time_seconds: Optional[int] = None


@dataclass
class JobStatus:
    """Status of a generation job."""

    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: Optional[int] = None
    created_at: str = ""
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def is_done(self) -> bool:
        """Whether the job has reached a terminal state."""
        return self.status in ("completed", "failed")

    @property
    def succeeded(self) -> bool:
        """Whether the job completed successfully."""
        return self.status == "completed"


@dataclass
class CreditsResponse:
    """User credit balance."""

    balance: int
    lifetime_purchased: int
    lifetime_used: int


@dataclass
class WebhookEvent:
    """Parsed webhook event payload."""

    event: str
    event_id: str
    timestamp: str
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def job_id(self) -> Optional[str]:
        """Job ID from event data."""
        return self.data.get("job_id")

    @property
    def is_completed(self) -> bool:
        """Whether this is a job.completed event."""
        return self.event == "job.completed"

    @property
    def is_failed(self) -> bool:
        """Whether this is a job.failed event."""
        return self.event == "job.failed"

    @property
    def output_url(self) -> Optional[str]:
        """Output URL (only for job.completed events)."""
        return self.data.get("output_url")

    @property
    def error(self) -> Optional[str]:
        """Error message (only for job.failed events)."""
        return self.data.get("error")


@dataclass
class HealthResponse:
    """API health check response."""

    status: str
    version: str
    timestamp: str


@dataclass
class DeepHealthResponse:
    """Deep health check response with service statuses."""

    status: str
    services: Dict[str, Any] = field(default_factory=dict)
    disk: Optional[Dict[str, Any]] = None
    timestamp: str = ""
