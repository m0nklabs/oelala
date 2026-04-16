"""
Video Upscale adapter — SeedVR2, RealESRGAN, or Lanczos via ComfyUI.

Multiple preset modes: fast (lanczos), balanced (RealESRGAN), quality (SeedVR2).
"""

from __future__ import annotations

import logging
from typing import Any

from ...adapter import GenerationAdapter, ProgressCallback
from ...types import (
    AdapterConstraints,
    ComputeTarget,
    GenerationRequest,
    GenerationResult,
    LoraFormat,
    MediaType,
    Operation,
)

logger = logging.getLogger(__name__)

# Credit costs by upscale model
UPSCALE_CREDITS = {
    "seedvr2": 10,
    "realesrgan": 5,
    "lanczos": 2,
    "bicubic": 2,
}


class VideoUpscaleAdapter(GenerationAdapter):
    """
    Local video upscale via ComfyUI.

    Presets override model selection:
    - fast: Lanczos (2 credits)
    - balanced: RealESRGAN (5 credits)
    - quality: SeedVR2 with tiled VAE + BlockSwap (10 credits)
    """

    name = "local-upscale-video"
    model_family = ""
    supported_ops = {Operation.UPSCALE}
    input_types = {MediaType.VIDEO}
    output_type = MediaType.VIDEO
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.NONE

    def __init__(self, comfyui_client_fn: Any = None) -> None:
        self._get_comfyui = comfyui_client_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=3840,
            max_height=2160,
            min_width=64,
            min_height=64,
        )

    def _resolve_model(self, req: GenerationRequest) -> str:
        """Resolve actual upscale model from preset or explicit model."""
        preset = req.upscale_preset
        if preset == "fast":
            return "lanczos"
        elif preset == "quality":
            return "seedvr2"
        elif preset == "balanced":
            return "realesrgan"
        return req.upscale_model or "lanczos"

    def build_workflow(self, req: GenerationRequest) -> dict:
        """Stub — video upscale uses ComfyUI client methods directly."""
        return {
            "_adapter": self.name,
            "_model": self._resolve_model(req),
        }

    def cost(self, req: GenerationRequest) -> int:
        model = self._resolve_model(req)
        return UPSCALE_CREDITS.get(model, 5)

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")

        if not req.input_video:
            raise ValueError("Video upscale requires an input video")

        client = self._get_comfyui()
        model = self._resolve_model(req)
        scale = req.upscale_scale

        # Delegate to ComfyUI — specific builder method depends on model
        prompt_id = client.queue_prompt(self.build_workflow(req))
        if not prompt_id:
            raise RuntimeError("Failed to queue video upscale workflow")

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=0,  # Router fills this in
            adapter_name=self.name,
            meta={"model": model, "scale": scale},
        )
