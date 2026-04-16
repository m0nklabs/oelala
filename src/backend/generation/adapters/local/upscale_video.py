"""
Video Upscale adapter — SeedVR2, RealESRGAN, or Lanczos via ComfyUI.

Multiple preset modes: fast (lanczos), balanced (RealESRGAN), quality (SeedVR2).
"""

from __future__ import annotations

import logging
import uuid
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

# Model file mapping for UpscaleModelLoader
UPSCALE_MODELS = {
    "realesrgan": "realesrgan-x4plus.pth",
    "seedvr2": "seedvr2.pth",
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
        """
        Build a real ComfyUI video upscale workflow.

        Pipeline: VHS_LoadVideo → UpscaleModelLoader → ImageUpscaleWithModel → VHS_VideoCombine
        """
        model = self._resolve_model(req)
        model_file = UPSCALE_MODELS.get(model, "realesrgan-x4plus.pth")

        # video_name should already be a ComfyUI filename (uploaded by router)
        video_name = req.input_video or ""

        prefix = f"upscaled_{uuid.uuid4().hex[:8]}"

        return {
            "1": {
                "inputs": {
                    "video": video_name,
                    "force_rate": 0,
                    "force_size": "Disabled",
                    "custom_width": 0,
                    "custom_height": 0,
                    "frame_load_cap": 0,
                    "skip_first_frames": 0,
                    "select_every_nth": 1,
                },
                "class_type": "VHS_LoadVideo",
            },
            "2": {
                "inputs": {
                    "model_name": model_file,
                },
                "class_type": "UpscaleModelLoader",
            },
            "3": {
                "inputs": {
                    "upscale_model": ["2", 0],
                    "image": ["1", 0],
                },
                "class_type": "ImageUpscaleWithModel",
            },
            "4": {
                "inputs": {
                    "frame_rate": ["1", 2],
                    "loop_count": 0,
                    "filename_prefix": prefix,
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 19,
                    "save_metadata": True,
                    "images": ["3", 0],
                    "audio": ["1", 1],
                },
                "class_type": "VHS_VideoCombine",
            },
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

        workflow = self.build_workflow(req)
        prompt_id = client.queue_prompt(workflow)
        if not prompt_id:
            raise RuntimeError("Failed to queue video upscale workflow")

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=0,  # Router fills this in
            adapter_name=self.name,
            meta={"model": model, "scale": req.upscale_scale},
        )
