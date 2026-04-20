"""
GenerationRouter — dispatches GenerationRequests to the correct adapter.

Handles:
1. Adapter resolution (by hint or auto-match)
2. Resolution string → pixel mapping
3. Frame count normalization (4k+1 for Wan2.2)
4. ComfyUI image pre-upload for local adapters
5. Control validation against adapter.constraints()
6. LoRA filtering by model compatibility
7. Credit check + deduction
8. Adapter execution
9. Job tracking
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable, Optional

from .adapter import GenerationAdapter, ProgressCallback
from .registry import AdapterRegistry
from .types import (
    ComputeTarget,
    GenerationRequest,
    GenerationResult,
    LoraFormat,
    MediaType,
)
from . import lora_utils

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone resolution helpers (no ComfyUI client dependency)
# ---------------------------------------------------------------------------

# Base heights for named resolution presets
_RESOLUTION_HEIGHTS = {"480p": 480, "576p": 576, "720p": 720, "1080p": 1080}

# Aspect ratio numeric pairs
_ASPECT_RATIOS = {
    "16:9": (16, 9),
    "9:16": (9, 16),
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "21:9": (21, 9),
    "auto": (1, 1),
}


def resolve_resolution(
    resolution: Optional[str],
    aspect_ratio: Optional[str],
    step: int = 8,
) -> tuple[int, int] | None:
    """
    Convert a named resolution + aspect ratio to (width, height) in pixels.

    Returns *None* if *resolution* is not provided (caller should keep
    whatever width/height is already on the request).
    """
    if not resolution:
        return None

    height = _RESOLUTION_HEIGHTS.get(resolution, 480)
    ar_w, ar_h = _ASPECT_RATIOS.get(aspect_ratio or "1:1", (1, 1))

    if ar_w >= ar_h:
        width = int(height * ar_w / ar_h)
    else:
        width = height
        height = int(width * ar_h / ar_w)

    # Snap to VAE-friendly multiples
    width = (width // step) * step
    height = (height // step) * step
    return width, height


def normalize_frame_count(frames: int) -> int:
    """
    Snap *frames* to the nearest Wan2.2 valid count (4k+1).

    Wan2.2 requires frame counts of 5, 9, 13, … , 321.
    """
    k = round((frames - 1) / 4)
    k = max(1, k)  # minimum 5 frames
    return 4 * k + 1


def _is_base64_image(data: str) -> bool:
    """Heuristic: is *data* a base64-encoded image (not a ComfyUI filename)?"""
    if len(data) > 260:
        return True
    if data.startswith(("data:image/", "/9j/", "iVBOR")):
        return True
    return False


class GenerationRouter:
    """
    Central dispatch point for all generation requests.

    The router is stateless — all state lives in the registry,
    the credit system, and the job queue.
    """

    def __init__(
        self,
        registry: AdapterRegistry,
        *,
        comfyui_upload_fn: Optional[Callable] = None,
    ) -> None:
        self.registry = registry
        # Callable: async (b64_data: str, filename: str) -> str (ComfyUI filename)
        self._comfyui_upload_fn = comfyui_upload_fn

    def resolve_adapter(self, req: GenerationRequest) -> GenerationAdapter:
        """
        Find the best adapter for a request.

        Priority:
        1. adapter_hint (exact name match)
        2. Auto-match by (operation, input_types, target_type)
        """
        # 1. Explicit adapter hint
        if req.adapter_hint:
            adapter = self.registry.get(req.adapter_hint)
            if adapter is None:
                raise ValueError(
                    f"Requested adapter '{req.adapter_hint}' not found. "
                    f"Available: {[a.name for a in self.registry.list_all()]}"
                )
            return adapter

        # 2. Auto-match
        # Determine input type from request content
        if req.input_images:
            input_type = MediaType.IMAGE
        elif req.input_video:
            input_type = MediaType.VIDEO
        elif req.input_audio:
            input_type = MediaType.AUDIO
        else:
            input_type = MediaType.TEXT

        candidates = self.registry.find(
            operation=req.operation,
            input_type=input_type,
            target_type=req.target_type,
        )

        if not candidates:
            raise ValueError(
                f"No adapter found for operation={req.operation.value}, "
                f"input={input_type.value if input_type else 'none'}, "
                f"target={req.target_type.value}"
            )

        # If multiple candidates, prefer local over cloud (unless cloud requested)
        # Sort by name for deterministic behaviour regardless of registration order
        if len(candidates) > 1:
            local = sorted(
                [a for a in candidates if a.compute == ComputeTarget.LOCAL],
                key=lambda a: a.name,
            )
            cloud = sorted(
                [a for a in candidates if a.compute == ComputeTarget.CLOUD],
                key=lambda a: a.name,
            )
            return local[0] if local else cloud[0]

        return candidates[0]

    def resolve_resolution_fields(self, req: GenerationRequest) -> GenerationRequest:
        """
        If *resolution* and/or *aspect_ratio* are set but width/height are
        not, compute pixel dimensions from the named preset.
        """
        if req.width is not None and req.height is not None:
            return req  # already explicit

        result = resolve_resolution(req.resolution, req.aspect_ratio)
        if result is None:
            return req  # no resolution string to resolve

        w, h = result
        updates: dict[str, Any] = {}
        if req.width is None:
            updates["width"] = w
        if req.height is None:
            updates["height"] = h
        return req.model_copy(update=updates) if updates else req

    def normalize_frames(
        self, req: GenerationRequest, adapter: GenerationAdapter
    ) -> GenerationRequest:
        """
        Snap frame count to 4k+1 for Wan2.2 adapters.
        """
        if req.frames is None:
            return req
        if "wan2" not in adapter.model_family.lower():
            return req

        normalised = normalize_frame_count(req.frames)
        if normalised != req.frames:
            logger.debug(
                f"🎞️ Frame count normalised: {req.frames} → {normalised} (4k+1)"
            )
            return req.model_copy(update={"frames": normalised})
        return req

    async def upload_local_images(
        self, req: GenerationRequest, adapter: GenerationAdapter
    ) -> GenerationRequest:
        """
        For local adapters, if input_images contain base64 data,
        upload them to ComfyUI and replace with the returned filename.
        """
        if adapter.compute != ComputeTarget.LOCAL:
            return req
        if not req.input_images:
            return req
        if self._comfyui_upload_fn is None:
            return req

        new_images: list[str] = []
        for idx, img in enumerate(req.input_images):
            if _is_base64_image(img):
                # Strip data-URI prefix if present
                raw = img
                if raw.startswith("data:"):
                    raw = raw.split(",", 1)[-1]
                try:
                    filename = f"v2_input_{random.randint(10000, 99999)}_{idx}.png"
                    result_name = await self._comfyui_upload_fn(raw, filename)
                    new_images.append(result_name)
                    logger.debug(f"📤 Uploaded image {idx} → {result_name}")
                except Exception:
                    logger.exception(f"❌ Failed to upload image {idx} to ComfyUI")
                    new_images.append(img)  # fall through with original
            else:
                new_images.append(img)  # already a ComfyUI filename

        return req.model_copy(update={"input_images": new_images})

    def validate_controls(
        self, req: GenerationRequest, adapter: GenerationAdapter
    ) -> GenerationRequest:
        """
        Validate and clamp request controls against adapter constraints.

        Returns a (possibly modified) request with defaults applied.
        """
        c = adapter.constraints()
        updates: dict[str, Any] = {}

        # Apply default steps/cfg if not specified
        if req.steps is None:
            updates["steps"] = c.default_steps
        if req.cfg is None:
            updates["cfg"] = c.default_cfg

        # Clamp resolution to adapter limits and step
        if req.width is not None:
            w = max(c.min_width, min(c.max_width, req.width))
            updates["width"] = (w // c.resolution_step) * c.resolution_step
        if req.height is not None:
            h = max(c.min_height, min(c.max_height, req.height))
            updates["height"] = (h // c.resolution_step) * c.resolution_step

        # Clamp steps (use the default we just applied if steps was None)
        steps = updates.get("steps", req.steps)
        if steps is not None:
            updates["steps"] = max(c.min_steps, min(c.max_steps, steps))

        # Generate random seed if -1
        if req.seed == -1:
            updates["seed"] = random.randint(0, 2**32 - 1)

        return req.model_copy(update=updates) if updates else req

    def filter_loras(
        self, req: GenerationRequest, adapter: GenerationAdapter
    ) -> GenerationRequest:
        """
        Filter LoRAs for model compatibility and enforce max count.

        Delegates to lora_utils for the actual filtering logic.
        """
        if not req.loras:
            return req

        # Convert LoraStackItem models to dicts for existing helper functions
        lora_dicts = [lr.model_dump(exclude_none=True) for lr in req.loras]

        # For single-stage adapters, sanitize dual-stage configs
        if adapter.lora_format == LoraFormat.SINGLE_STAGE:
            lora_dicts = lora_utils.sanitize_lora_configs_for_single_stage(lora_dicts)

        # Filter by model compatibility
        lora_dicts = lora_utils.filter_loras_by_model_compat(
            lora_dicts, adapter.model_family
        )

        # Enforce max LoRAs
        c = adapter.constraints()
        if len(lora_dicts) > c.max_loras:
            logger.warning(
                f"⚠️ Trimming LoRA stack from {len(lora_dicts)} to {c.max_loras}"
            )
            lora_dicts = lora_dicts[: c.max_loras]

        # Convert back to LoraStackItem
        from .types import LoraStackItem

        filtered_loras = [LoraStackItem(**d) for d in lora_dicts]
        return req.model_copy(update={"loras": filtered_loras})

    async def dispatch(
        self,
        req: GenerationRequest,
        user: Any,
        *,
        check_credits_fn: Any = None,
        deduct_credits_fn: Any = None,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        """
        Full dispatch pipeline:
        1. Resolve adapter
        2. Validate controls
        3. Filter LoRAs
        4. Calculate + check credits
        5. Execute adapter
        6. Deduct credits
        7. Return result

        Args:
            req: The generation request
            user: Authenticated user object (from auth.User)
            check_credits_fn: async fn(user, amount) — raises on insufficient
            deduct_credits_fn: async fn(user, amount, job_id, description) -> bool
            progress_callback: Optional WebSocket progress callback
        """
        # 1. Resolve adapter
        adapter = self.resolve_adapter(req)
        logger.info(f"🎯 Resolved adapter: {adapter.name}")

        # 2. Resolve resolution string → width/height
        req = self.resolve_resolution_fields(req)

        # 3. Validate controls (clamp, defaults)
        req = self.validate_controls(req, adapter)

        # 4. Normalize frames (4k+1 for Wan2.2)
        req = self.normalize_frames(req, adapter)

        # 5. Filter LoRAs
        req = self.filter_loras(req, adapter)

        # 6. Upload images to ComfyUI for local adapters
        req = await self.upload_local_images(req, adapter)

        # 7. Calculate + check credits
        credits_required = adapter.cost(req)
        if check_credits_fn:
            await check_credits_fn(user, credits_required)

        # 8. Execute
        result = await adapter.execute(req, progress_callback=progress_callback)
        result = result.model_copy(
            update={"credits_used": credits_required, "adapter_name": adapter.name}
        )

        # 9. Deduct credits
        if deduct_credits_fn:
            await deduct_credits_fn(
                user, credits_required, result.prompt_id, adapter.name
            )

        logger.info(
            f"✅ Dispatched via {adapter.name}: prompt_id={result.prompt_id}, "
            f"credits={credits_required}"
        )

        return result
