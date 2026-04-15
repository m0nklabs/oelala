"""
Image Upscale adapter — RealESRGAN/Lanczos via ComfyUI.

Supports multiple upscale models with optional face enhancement.
"""

from __future__ import annotations

import logging
from typing import Any

from generation.adapter import GenerationAdapter, ProgressCallback
from generation.types import (
    AdapterConstraints,
    ComputeTarget,
    GenerationRequest,
    GenerationResult,
    LoraFormat,
    MediaType,
    Operation,
)

logger = logging.getLogger(__name__)


class ImageUpscaleAdapter(GenerationAdapter):
    """
    Local image upscale via ComfyUI — RealESRGAN, SwinIR, or Lanczos.

    Optional face enhancement using GFPGAN/CodeFormer.
    """

    name = "local-upscale-image"
    model_family = ""
    supported_ops = {Operation.UPSCALE}
    input_types = {MediaType.IMAGE}
    output_type = MediaType.IMAGE
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.NONE

    def __init__(self, comfyui_client_fn: Any = None) -> None:
        self._get_comfyui = comfyui_client_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=8192,
            max_height=8192,
            min_width=64,
            min_height=64,
            max_input_images=1,
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        """Build ComfyUI upscale workflow."""
        model = req.upscale_model or "RealESRGAN_x4plus.pth"

        workflow = {
            "1": {
                "inputs": {"image": req.input_images[0] if req.input_images else "input.png"},
                "class_type": "LoadImage",
            },
            "2": {
                "inputs": {"model_name": model},
                "class_type": "UpscaleModelLoader",
            },
            "3": {
                "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]},
                "class_type": "ImageUpscaleWithModel",
            },
            "4": {
                "inputs": {"filename_prefix": "oelala_upscale", "images": ["3", 0]},
                "class_type": "SaveImage",
            },
        }

        return workflow

    def cost(self, req: GenerationRequest) -> int:
        return 2

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")

        if not req.input_images:
            raise ValueError("Image upscale requires an input image")

        workflow = self.build_workflow(req)
        client = self._get_comfyui()
        prompt_id = client.queue_prompt(workflow)

        if not prompt_id:
            raise RuntimeError("Failed to queue upscale workflow")

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=self.cost(req),
            adapter_name=self.name,
            meta={"model": req.upscale_model or "RealESRGAN_x4plus.pth"},
        )
