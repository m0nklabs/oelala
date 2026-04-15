"""
Core data types for the Unified Generation system.

All enums, Pydantic models, and data structures used across
adapters, the registry, and the router.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MediaType(str, Enum):
    """Type of media being generated or consumed."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"


class Operation(str, Enum):
    """What the adapter does to the input."""

    GENERATE = "generate"  # T2I, T2V — from scratch
    TRANSFORM = "transform"  # I2I denoise, V2V style transfer
    EDIT = "edit"  # Qwen instruction-based editing
    UPSCALE = "upscale"
    INTERPOLATE = "interpolate"  # Frame interpolation
    SWAP = "swap"  # Face swap
    CAPTION = "caption"  # Image/Video → text
    LIPSYNC = "lipsync"
    INPAINT = "inpaint"


class ComputeTarget(str, Enum):
    """Where the adapter runs."""

    LOCAL = "local"
    CLOUD = "cloud"
    AUTO = "auto"


class LoraFormat(str, Enum):
    """LoRA weight format expected by the adapter."""

    NONE = "none"
    SINGLE_STAGE = "single"  # {name, strength}
    DUAL_STAGE = "dual"  # {high, low, strength} (Wan2.2 only)


class AdapterConstraints(BaseModel):
    """Declares what an adapter supports — used by router AND frontend."""

    max_width: int = 2048
    max_height: int = 2048
    min_width: int = 256
    min_height: int = 256
    max_frames: Optional[int] = None
    resolution_step: int = 16
    resolution_presets: list[str] = Field(default_factory=list)
    aspect_ratios: list[str] = Field(default_factory=list)
    min_steps: int = 1
    max_steps: int = 100
    default_steps: int = 20
    default_cfg: float = 7.0
    supported_samplers: list[str] = Field(default_factory=list)
    supported_schedulers: list[str] = Field(default_factory=list)
    max_loras: int = 5
    supports_lightning: bool = False
    supports_negative_prompt: bool = True
    max_input_images: int = 1
    allowed_fps: list[int] = Field(default_factory=list)
    max_duration_seconds: Optional[float] = None


class LoraStackItem(BaseModel):
    """A single LoRA in the request stack."""

    name: str = ""
    strength: float = 1.0
    # Dual-stage fields (Wan2.2 only, adapter handles internally)
    high: Optional[str] = None
    low: Optional[str] = None


class GenerationRequest(BaseModel):
    """Unified request body for all generation operations."""

    operation: Operation
    target_type: MediaType
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = -1
    steps: Optional[int] = None
    cfg: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    frames: Optional[int] = None
    fps: Optional[int] = None
    resolution: Optional[str] = None
    aspect_ratio: Optional[str] = None
    sampler: Optional[str] = None
    scheduler: Optional[str] = None
    loras: list[LoraStackItem] = Field(default_factory=list)
    lightning: bool = False
    denoise: Optional[float] = None
    # Input media
    input_images: list[str] = Field(default_factory=list)  # base64 encoded
    input_video: Optional[str] = None  # base64 encoded
    input_audio: Optional[str] = None  # base64 or URL
    # Adapter hint
    adapter_hint: Optional[str] = None  # Force specific adapter
    # Operation-specific
    instruction: Optional[str] = None  # Operation.EDIT
    checkpoint: Optional[str] = None  # Local T2I
    audio_prompt: Optional[str] = None  # Audio/music generation
    # I2I face features
    face_id: bool = False
    face_detailer: bool = False
    face_restore: bool = False
    face_id_weight: float = 0.8


class GenerationResult(BaseModel):
    """Returned by the router after dispatching to an adapter."""

    prompt_id: str
    status: str  # "queued_local", "queued_cloud"
    compute_target: ComputeTarget
    credits_used: int
    runpod_job_id: Optional[str] = None
    adapter_name: str
    meta: dict = Field(default_factory=dict)
