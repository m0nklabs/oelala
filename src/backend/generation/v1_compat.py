"""
V1 → V2 compatibility layer.

Converts V1 form-based endpoint parameters to V2 GenerationRequest
and V2 GenerationResult back to V1 response dicts.

Provides ``dispatch_v1()`` as a single-call helper that:
    1. Converts V1 form params → GenerationRequest
    2. Dispatches through V2 router
    3. Registers the job for auto-upload (V1 behaviour)
    4. Returns V1-compatible response dict

Usage in endpoint wrappers::

    @app.post("/generate-sdxl")
    async def generate_sdxl_image(
        prompt: str = Form(...), ..., user = Depends(get_current_user),
    ):
        return await dispatch_v1(
            form=dict(prompt=prompt, ...),
            files={},
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            adapter_hint="sdxl-local-t2i",
            user=user,
        )
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from fastapi import UploadFile

from .types import (
    GenerationRequest,
    GenerationResult,
    LoraStackItem,
    MediaType,
    Operation,
)

logger = logging.getLogger(__name__)

# ── Module-level state (set by init_v1_compat) ──────────────────────
_router: Any = None
_check_credits_fn: Any = None
_deduct_credits_fn: Any = None
_get_comfyui_client_fn: Callable | None = None


def init_v1_compat(
    router: Any,
    check_credits: Any,
    deduct_credits: Any,
    get_comfyui_client: Callable | None = None,
) -> None:
    """
    Initialize the V1 compat layer with dependencies from app.py.

    Called once at startup after the V2 router is ready.
    """
    global _router, _check_credits_fn, _deduct_credits_fn, _get_comfyui_client_fn
    _router = router
    _check_credits_fn = check_credits
    _deduct_credits_fn = deduct_credits
    _get_comfyui_client_fn = get_comfyui_client
    logger.info("🔌 V1 compat layer initialized")


# ── V1 field name → V2 field name mapping ────────────────────────────
_FIELD_ALIASES = {
    "num_frames": "frames",
    "guidance_scale": "cfg",
    "guidance": "cfg",
    "sampler_name": "sampler",
    "text": "prompt",
    "lora_configs": "_lora_json",
    "compute_target": "_compute_target",
    "model_type": "_model_type",
    "post_processing": "_post_processing",
    "mode": "audio_mode",
    "style": "audio_style",
    "output_filename": "_ignored",
}

_PASSTHROUGH_FIELDS = {
    "prompt",
    "negative_prompt",
    "seed",
    "steps",
    "cfg",
    "width",
    "height",
    "frames",
    "fps",
    "resolution",
    "aspect_ratio",
    "sampler",
    "scheduler",
    "checkpoint",
    "lightning",
    "denoise",
    "strength",
    "voice",
    "audio_mode",
    "audio_style",
    "duration",
    "speed",
    "pitch",
    "face_id",
    "face_detailer",
    "face_restore",
    "face_id_weight",
    "face_indices",
    "v2v_mode",
    "preserve_motion",
    "upscale_model",
    "upscale_scale",
    "upscale_preset",
    "face_enhance",
    "instruction",
    "edit_model",
    "shift",
    "high_noise_steps",
    "audio_prompt",
    "caption_mode",
    "detail_level",
    "interpolation_mode",
    "target_fps",
    "multiplier",
    "generation_mode",
    "extend_mode",
    "clip_count",
    "unet_high_noise",
    "unet_low_noise",
}


def _parse_loras(raw: str | list | None) -> list[LoraStackItem]:
    """Parse V1 LoRA config (JSON string or list of dicts) → list[LoraStackItem]."""
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(raw, list):
        return []
    items = []
    for entry in raw:
        if isinstance(entry, dict) and entry.get("name"):
            items.append(
                LoraStackItem(
                    name=entry["name"],
                    strength=float(entry.get("strength", 1.0)),
                    high=entry.get("high"),
                    low=entry.get("low"),
                )
            )
    return items


async def _read_upload_file(upload: UploadFile) -> str:
    """Read an UploadFile and return base64-encoded string."""
    import base64

    data = await upload.read()
    return base64.b64encode(data).decode("ascii")


async def form_to_generation_request(
    form: dict[str, Any],
    files: dict[str, UploadFile | list[UploadFile]] | None = None,
    operation: Operation = Operation.GENERATE,
    target_type: MediaType = MediaType.IMAGE,
    adapter_hint: str | None = None,
) -> GenerationRequest:
    """
    Convert V1 form parameters to a V2 GenerationRequest.

    Args:
        form: Dict of form field values (from Form(...) params).
        files: Dict of uploaded files (UploadFile instances).
        operation: The V2 operation type.
        target_type: The V2 target media type.
        adapter_hint: Force a specific adapter (optional).

    Returns:
        A fully populated GenerationRequest.
    """
    files = files or {}

    # Start with required fields
    req_data: dict[str, Any] = {
        "operation": operation,
        "target_type": target_type,
    }
    if adapter_hint:
        req_data["adapter_hint"] = adapter_hint

    # Process form fields
    for key, value in form.items():
        if value is None:
            continue

        # Check for alias mapping
        mapped = _FIELD_ALIASES.get(key, key)

        if mapped == "_lora_json":
            req_data["loras"] = _parse_loras(value)
        elif mapped == "_ignored":
            continue
        elif mapped.startswith("_"):
            # Special fields handled by caller or ignored
            continue
        elif mapped in _PASSTHROUGH_FIELDS:
            req_data[mapped] = value
        elif key in _PASSTHROUGH_FIELDS:
            req_data[key] = value
        # else: unknown field, skip silently

    # Process uploaded files → base64
    input_images: list[str] = []
    for name, upload in files.items():
        if upload is None:
            continue
        if isinstance(upload, list):
            for f in upload:
                b64 = await _read_upload_file(f)
                input_images.append(b64)
        else:
            b64 = await _read_upload_file(upload)
            if name in ("image", "source_image", "input_image"):
                input_images.append(b64)
            elif name in ("video", "input_video"):
                req_data["input_video"] = b64
            elif name in ("audio", "input_audio"):
                req_data["input_audio"] = b64
            elif name in ("mask", "input_mask"):
                req_data["input_mask"] = b64
            else:
                input_images.append(b64)

    if input_images:
        req_data["input_images"] = input_images

    return GenerationRequest(**req_data)


def generation_result_to_v1_response(
    result: GenerationResult,
    v1_format: str = "standard",
) -> dict[str, Any]:
    """
    Convert a V2 GenerationResult back to a V1-compatible response dict.

    V1 format variants:
    - standard: {"status": "queued", "prompt_id", "job_id", "credits_used", "meta"}
    - cloud: {"status": "queued_cloud", "job_id", "runpod_job_id", "credits_used", "meta"}

    Args:
        result: The V2 GenerationResult.
        v1_format: Response format ("standard" or "cloud").

    Returns:
        Dict matching the V1 response shape.
    """
    # Map V2 status to V1 status
    status_map = {
        "queued_local": "queued",
        "queued_cloud": "queued_cloud",
        "completed": "completed",
    }
    v1_status = status_map.get(result.status, result.status)

    response: dict[str, Any] = {
        "status": v1_status,
        "prompt_id": result.prompt_id,
        "job_id": result.prompt_id,  # V1 returns job_id as separate field
        "credits_used": result.credits_used,
    }

    # Cloud responses include the RunPod job ID
    if v1_format == "cloud" and result.runpod_job_id:
        response["runpod_job_id"] = result.runpod_job_id

    # Include adapter meta as V1 meta
    if result.meta:
        response["meta"] = result.meta

    return response


# ── High-level dispatch helper ──────────────────────────────────────


async def dispatch_v1(
    form: dict[str, Any],
    files: dict[str, UploadFile | list[UploadFile]] | None,
    operation: Operation,
    target_type: MediaType,
    adapter_hint: str,
    user: Any,
    *,
    v1_format: str = "standard",
    register_job_settings: dict[str, Any] | None = None,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """
    All-in-one V1→V2 dispatch: convert, route, register job, return V1 response.

    This is the single function each V1 endpoint thin wrapper should call.

    Args:
        form: V1 form field values.
        files: Uploaded files (UploadFile instances).
        operation: V2 operation.
        target_type: V2 target media type.
        adapter_hint: Which adapter to use.
        user: Authenticated user object.
        v1_format: Response format ("standard" or "cloud").
        register_job_settings: If provided, register the job with ComfyUI
            for auto-upload. Dict with keys like job_type, model_name, etc.
        progress_callback: Optional WebSocket progress callback.

    Returns:
        V1-compatible response dict.
    """
    if _router is None:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503, detail="V2 generation engine not initialized"
        )

    # 1. Convert V1 form → V2 GenerationRequest
    gen_req = await form_to_generation_request(
        form=form,
        files=files,
        operation=operation,
        target_type=target_type,
        adapter_hint=adapter_hint,
    )

    # 2. Dispatch through V2 router (handles credits, validation, execution)
    result = await _router.dispatch(
        gen_req,
        user,
        check_credits_fn=_check_credits_fn,
        deduct_credits_fn=_deduct_credits_fn,
        progress_callback=progress_callback,
    )

    # 3. Register job for auto-upload (V1 behaviour)
    if register_job_settings and result.prompt_id and _get_comfyui_client_fn:
        try:
            client = _get_comfyui_client_fn()
            client.register_job(
                prompt_id=result.prompt_id,
                user_id=user.id,
                prompt=gen_req.prompt or "",
                settings=register_job_settings,
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to register job: {e}")

    # 4. Convert V2 result → V1 response
    return generation_result_to_v1_response(result, v1_format)
