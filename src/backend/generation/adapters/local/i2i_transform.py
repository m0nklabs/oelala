"""
I2I Transform adapter — image-to-image with optional face features.

Supports: IP-Adapter FaceID, FaceDetailer, GFPGAN face restore.
Delegates workflow to ComfyUIClient._build_i2i_workflow().
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


class I2ITransformAdapter(GenerationAdapter):
    """
    Local I2I transform via ComfyUI — SDXL-based with face features.

    Supports optional IP-Adapter FaceID Plus V2, FaceDetailer, and
    GFPGAN face restoration as toggleable features.
    """

    name = "local-i2i-transform"
    model_family = "sdxl"
    supported_ops = {Operation.TRANSFORM}
    input_types = {MediaType.IMAGE}
    output_type = MediaType.IMAGE
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.NONE

    def __init__(self, comfyui_client_fn: Any = None) -> None:
        self._get_comfyui = comfyui_client_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=2048,
            max_height=2048,
            min_width=512,
            min_height=512,
            resolution_step=64,
            min_steps=10,
            max_steps=100,
            default_steps=25,
            default_cfg=7.5,
            supported_samplers=["dpmpp_2m", "euler", "euler_ancestral", "dpmpp_sde"],
            supported_schedulers=["karras", "normal", "simple"],
            supports_negative_prompt=True,
            max_input_images=1,
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        """Stub — I2I workflow built by ComfyUIClient._build_i2i_workflow()."""
        return {
            "_adapter": self.name,
            "_note": "I2I uses ComfyUIClient._build_i2i_workflow()",
        }

    def cost(self, req: GenerationRequest) -> int:
        base = 2  # SDXL base
        if req.face_id:
            base += 3
        if req.face_detailer:
            base += 2
        if req.face_restore:
            base += 1
        return base

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")

        if not req.input_images:
            raise ValueError("I2I transform requires an input image")

        client = self._get_comfyui()

        # Delegate to ComfyUI's internal I2I builder
        workflow = client._build_i2i_workflow(
            image_name=req.input_images[0],
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            denoise=req.denoise or 0.7,
            checkpoint=req.checkpoint or "CyberRealistic_Pony_v14.1_FP16.safetensors",
            steps=req.steps or 25,
            cfg=req.cfg or 7.5,
            seed=req.seed,
            sampler_name=req.sampler or "dpmpp_2m",
            scheduler=req.scheduler or "karras",
            face_id=req.face_id,
            face_detailer=req.face_detailer,
            face_restore=req.face_restore,
            face_id_weight=req.face_id_weight,
        )

        prompt_id = client.queue_prompt(workflow)
        if not prompt_id:
            raise RuntimeError("Failed to queue I2I workflow")

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=self.cost(req),
            adapter_name=self.name,
            meta={
                "face_id": req.face_id,
                "face_detailer": req.face_detailer,
                "face_restore": req.face_restore,
            },
        )
