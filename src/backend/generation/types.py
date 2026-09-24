"""
Core data types for the Unified Generation system.

All enums, Pydantic models, and data structures used across
adapters, the registry, and the router.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


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
    EDIT = "edit"  # I2I instruction-based editing
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
    max_frames: int | None = None
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
    max_duration_seconds: float | None = None


class LoraStackItem(BaseModel):
    """A single LoRA in the request stack."""

    name: str = ""
    strength: float = 1.0
    # Dual-stage fields (Wan2.2 only, adapter handles internally)
    high: str | None = None
    low: str | None = None


class GenerationRequest(BaseModel):
    """Unified request body for all generation operations."""

    model_config = ConfigDict(extra="allow")

    operation: Operation
    target_type: MediaType
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = -1
    steps: int | None = None
    cfg: float | None = None
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    fps: int | None = None
    resolution: str | None = None
    aspect_ratio: str | None = None
    megapixels: float | None = None  # MiniMax-H3: target output size in MP (0.2–2.0)
    sampler: str | None = None
    scheduler: str | None = None
    loras: list[LoraStackItem] = Field(default_factory=list)
    lightning: bool = False
    denoise: float | None = None
    strength: float | None = None  # I2V conditioning strength
    # Wan2.2 cloud-specific
    high_noise_steps: int | None = None  # Steps for high noise pass
    shift: float | None = None  # ModelSamplingSD3 shift
    # Input media
    input_images: list[str] = Field(default_factory=list)  # base64 encoded
    input_video: str | None = None  # base64 encoded
    input_audio: str | None = None  # base64 or URL
    # Adapter hint
    adapter_hint: str | None = None  # Force specific adapter
    # Operation-specific
    instruction: str | None = None  # Operation.EDIT
    checkpoint: str | None = None  # Local T2I
    edit_model: str | None = None  # Qwen edit model variant
    audio_prompt: str | None = None  # Audio/music generation
    # I2I face features
    face_id: bool = False
    face_detailer: bool = False
    face_restore: bool = False
    face_id_weight: float = 0.8
    # Inpainting
    input_mask: str | None = None  # base64-encoded mask (white=regen)
    feathering: int = 16
    # Face swap
    face_indices: str = "0"  # "0", "0,1", or "-1" for all
    # Audio / TTS / Music
    audio_mode: str | None = None  # "tts", "music", "sfx"
    voice: str | None = None  # TTS voice name
    audio_style: str | None = None  # Music style
    duration: float | None = None  # Audio duration seconds
    speed: float | None = None  # TTS speed
    pitch: float | None = None  # TTS pitch
    # Voice clone
    voice_sample_path: str | None = None
    # Lip sync
    lips_expression: float = 1.5
    inference_steps: int | None = None
    # V2V
    v2v_mode: str | None = None  # "style_transfer", "anime", "enhance"
    preserve_motion: bool = True
    # Upscale
    upscale_model: str | None = None
    upscale_scale: float = 4.0
    face_enhance: bool = False
    upscale_preset: str | None = None  # "fast", "balanced", "quality"
    # Interpolation
    interpolation_mode: str | None = None  # "fps", "slowmo"
    target_fps: int | None = None
    multiplier: float | None = None
    # Captioning
    caption_mode: str | None = None  # "brief", "detailed", "tags", etc.
    detail_level: int = 3
    include_negative: bool = False
    include_motion: bool = False
    frame_interval: float | None = None  # Video caption frame interval
    max_frames: int | None = None  # Video caption max frames
    # User context (set by router.dispatch, used by adapters for media storage)
    user_id: str | None = None  # Supabase user ID for result storage


class GenerationResult(BaseModel):
    """Returned by the router after dispatching to an adapter."""

    prompt_id: str
    status: str  # "queued_local", "queued_cloud"
    compute_target: ComputeTarget
    credits_used: int
    runpod_job_id: str | None = None
    adapter_name: str
    meta: dict = Field(default_factory=dict)
