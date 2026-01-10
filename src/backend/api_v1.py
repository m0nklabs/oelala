"""
Oelala REST API v1
Public programmatic API for external integrations.
"""

import os
import logging
from typing import Optional, List, Literal
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path

from api_key_auth import get_api_key_user
from auth import User
from credits import calculate_credits, get_credit_manager
from credits_api import check_credits, deduct_credits

logger = logging.getLogger(__name__)

# Create v1 API router
router = APIRouter(prefix="/api/v1", tags=["API v1"])

# Debug flag
DEBUG = os.getenv("OELALA_DEBUG", "0") == "1"


def debug_log(msg: str):
    if DEBUG:
        logger.info(f"🌐 API-v1: {msg}")


# =============================================================================
# Pydantic Models
# =============================================================================


class GenerateRequest(BaseModel):
    """Request body for generation endpoint."""

    type: Literal["text-to-image", "text-to-video", "image-to-video"] = Field(
        ..., description="Type of generation"
    )
    prompt: str = Field(..., min_length=1, description="Text prompt for generation")
    negative_prompt: Optional[str] = Field(
        None, description="Negative prompt (what to avoid)"
    )
    width: Optional[int] = Field(1024, ge=256, le=2048, description="Output width")
    height: Optional[int] = Field(1024, ge=256, le=2048, description="Output height")
    steps: Optional[int] = Field(20, ge=1, le=100, description="Inference steps")
    cfg: Optional[float] = Field(7.5, ge=1.0, le=20.0, description="CFG scale")
    seed: Optional[int] = Field(-1, description="Random seed (-1 for random)")
    duration_seconds: Optional[int] = Field(
        None, ge=1, le=30, description="Video duration in seconds (for video generation)"
    )
    # For image-to-video
    image_url: Optional[str] = Field(None, description="URL of source image (for I2V)")


class GenerateResponse(BaseModel):
    """Response from generation endpoint."""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status (queued, running, completed, failed)")
    credits_used: int = Field(..., description="Credits deducted for this generation")
    estimated_time_seconds: Optional[int] = Field(
        None, description="Estimated completion time in seconds"
    )


class JobStatus(BaseModel):
    """Job status response."""

    job_id: str = Field(..., description="Job identifier")
    status: Literal["queued", "running", "completed", "failed"] = Field(
        ..., description="Current job status"
    )
    progress: Optional[int] = Field(None, ge=0, le=100, description="Progress percentage")
    created_at: str = Field(..., description="Job creation timestamp (ISO 8601)")
    completed_at: Optional[str] = Field(None, description="Job completion timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")
    result_url: Optional[str] = Field(
        None, description="URL to download result (when completed)"
    )
    metadata: Optional[dict] = Field(None, description="Additional job metadata")


class CreditsResponse(BaseModel):
    """Credits balance response."""

    balance: int = Field(..., description="Current available credits")
    lifetime_purchased: int = Field(..., description="Total credits ever purchased")
    lifetime_used: int = Field(..., description="Total credits ever used")


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    user: User = Depends(get_api_key_user),
):
    """
    Generate image or video using AI.

    **Authentication:** Requires API key via `X-API-Key` header.

    **Credit Costs:**
    - Text-to-Image (SDXL): ~10 credits
    - Text-to-Video (Wan2.2): ~50-200 credits depending on duration
    - Image-to-Video (Wan2.2): ~50-200 credits depending on duration

    **Rate Limits:** TBD (per API key)

    **Example:**
    ```bash
    curl -X POST https://oelala.xyz/api/v1/generate \\
      -H "X-API-Key: oelala_your_key_here" \\
      -H "Content-Type: application/json" \\
      -d '{
        "type": "text-to-image",
        "prompt": "a beautiful sunset over mountains",
        "width": 1024,
        "height": 1024
      }'
    ```
    """
    debug_log(
        f"Generate request: type={request.type}, prompt={request.prompt[:50]}..., user={user.id}"
    )

    # Import here to avoid circular dependency
    from comfyui_client import get_comfyui_client

    client = get_comfyui_client()
    if not client or not client.is_available():
        raise HTTPException(
            status_code=503, detail="Generation service temporarily unavailable"
        )

    # Calculate credits
    if request.type == "text-to-image":
        credits_required = calculate_credits(
            "sdxl",
            width=request.width or 1024,
            height=request.height or 1024,
            steps=request.steps or 20,
        )
        generation_type = "T2I"
    elif request.type in ["text-to-video", "image-to-video"]:
        duration = request.duration_seconds or 3
        credits_required = calculate_credits(
            "wan22_i2v",
            width=request.width or 512,
            height=request.height or 512,
            duration_seconds=duration,
            steps=request.steps or 20,
        )
        generation_type = "T2V" if request.type == "text-to-video" else "I2V"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported type: {request.type}")

    # Check and deduct credits
    await check_credits(user, credits_required)

    # For now, return a placeholder response
    # In a real implementation, this would queue the job to ComfyUI
    import uuid

    job_id = str(uuid.uuid4())

    # Deduct credits
    await deduct_credits(user, credits_required, job_id, f"{generation_type} Generation")

    logger.info(
        f"🎨 API v1 generation queued: job={job_id}, type={request.type}, "
        f"credits={credits_required}, user={user.id}"
    )

    # TODO: Actual job queuing logic
    # For text-to-image: use /generate-image workflow
    # For text-to-video: use /generate-text workflow
    # For image-to-video: use /generate workflow

    return GenerateResponse(
        job_id=job_id,
        status="queued",
        credits_used=credits_required,
        estimated_time_seconds=30 if request.type == "text-to-image" else 120,
    )


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(
    job_id: str,
    user: User = Depends(get_api_key_user),
):
    """
    Get status of a generation job.

    **Authentication:** Requires API key via `X-API-Key` header.

    **Job Statuses:**
    - `queued`: Job is waiting to start
    - `running`: Job is currently processing
    - `completed`: Job finished successfully, result ready for download
    - `failed`: Job failed with error

    **Polling:** Poll this endpoint every 5-10 seconds until status is `completed` or `failed`.

    **Example:**
    ```bash
    curl https://oelala.xyz/api/v1/jobs/abc-123 \\
      -H "X-API-Key: oelala_your_key_here"
    ```
    """
    debug_log(f"Job status request: job_id={job_id}, user={user.id}")

    # TODO: Implement actual job status lookup
    # Check ComfyUI queue and history
    # Match job_id to prompt_id
    # Return actual status

    # Placeholder response
    return JobStatus(
        job_id=job_id,
        status="running",
        progress=50,
        created_at=datetime.utcnow().isoformat() + "Z",
        completed_at=None,
        error=None,
        result_url=None,
        metadata={"type": "text-to-image", "prompt": "example"},
    )


@router.get("/jobs/{job_id}/download")
async def download_job_result(
    job_id: str,
    user: User = Depends(get_api_key_user),
):
    """
    Download the result of a completed job.

    **Authentication:** Requires API key via `X-API-Key` header.

    **Prerequisites:** Job must be in `completed` status (check with `/jobs/{job_id}` first).

    **Response:** Binary file (image/png, image/jpeg, or video/mp4)

    **Example:**
    ```bash
    curl https://oelala.xyz/api/v1/jobs/abc-123/download \\
      -H "X-API-Key: oelala_your_key_here" \\
      -o result.mp4
    ```
    """
    debug_log(f"Download request: job_id={job_id}, user={user.id}")

    # TODO: Implement actual file download
    # 1. Verify job belongs to this user
    # 2. Check job is completed
    # 3. Locate output file in ComfyUI/output or generated/
    # 4. Return FileResponse

    # Placeholder - return 404 for now
    raise HTTPException(
        status_code=404,
        detail="Job result not found. Check job status first with GET /jobs/{job_id}",
    )


@router.get("/credits", response_model=CreditsResponse)
async def get_credits(
    user: User = Depends(get_api_key_user),
):
    """
    Get current credit balance.

    **Authentication:** Requires API key via `X-API-Key` header.

    **Example:**
    ```bash
    curl https://oelala.xyz/api/v1/credits \\
      -H "X-API-Key: oelala_your_key_here"
    ```
    """
    debug_log(f"Credits request: user={user.id}")

    manager = get_credit_manager()

    if not manager.service_key:
        logger.warning("SUPABASE_SERVICE_KEY not configured - returning default balance")
        return CreditsResponse(
            balance=25,
            lifetime_purchased=0,
            lifetime_used=0,
        )

    try:
        balance = await manager.get_balance(user.id)
        return CreditsResponse(
            balance=balance.balance,
            lifetime_purchased=balance.lifetime_purchased,
            lifetime_used=balance.lifetime_used,
        )
    except Exception as e:
        logger.error(f"Error getting balance: {e}")
        raise HTTPException(status_code=500, detail="Failed to get credit balance")


# Health check for API v1
@router.get("/health")
async def health_check():
    """
    Health check endpoint (no authentication required).

    Returns API version and status.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
