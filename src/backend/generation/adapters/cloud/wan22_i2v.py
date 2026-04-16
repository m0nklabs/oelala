"""
Wan2.2 Cloud I2V adapter — image-to-video via RunPod.

Delegates workflow building to ComfyUIClient.build_cloud_wan22_i2v_workflow().
Supports dual-stage LoRAs (high/low noise models).
"""

from __future__ import annotations

import logging
import uuid
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


class Wan22CloudI2VAdapter(GenerationAdapter):
    """
    Wan 2.2 Image-to-Video on RunPod (48GB+ GPU).

    Uses fp8_scaled dual-pass sampling with dedicated high/low noise models.
    Optimal for high-quality I2V on cloud GPUs (A6000/A40/L40S).
    """

    name = "wan22-cloud-i2v"
    model_family = "wan2.2"
    supported_ops = {Operation.GENERATE}
    input_types = {MediaType.IMAGE}
    output_type = MediaType.VIDEO
    compute = ComputeTarget.CLOUD
    lora_format = LoraFormat.DUAL_STAGE

    def __init__(
        self,
        submit_to_runpod_fn: Any = None,
        comfyui_client_fn: Any = None,
    ) -> None:
        """
        Args:
            submit_to_runpod_fn: Async function for RunPod submission.
            comfyui_client_fn: Callable that returns a ComfyUIClient instance.
        """
        self._submit_to_runpod = submit_to_runpod_fn
        self._get_comfyui = comfyui_client_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=1920,
            max_height=1080,
            min_width=480,
            min_height=480,
            max_frames=321,
            resolution_step=16,
            resolution_presets=["480p", "576p", "720p", "1080p"],
            aspect_ratios=["9:16", "16:9", "1:1", "4:3", "3:4", "3:2", "2:3"],
            min_steps=10,
            max_steps=30,
            default_steps=15,
            default_cfg=3.0,
            supported_samplers=["dpmpp_2m", "euler", "uni_pc"],
            supported_schedulers=["beta", "karras", "normal"],
            max_loras=5,
            supports_lightning=False,
            supports_negative_prompt=True,
            max_input_images=1,
            allowed_fps=[8, 12, 16, 24],
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")
        comfyui = self._get_comfyui()

        lora_dicts = (
            [lr.model_dump(exclude_none=True) for lr in req.loras]
            if req.loras
            else None
        )

        return comfyui.build_cloud_wan22_i2v_workflow(
            image_name=req.input_images[0] if req.input_images else "input.png",
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_frames=req.frames or 81,
            fps=req.fps or 16,
            steps=req.steps or 15,
            cfg=req.cfg or 3.0,
            seed=req.seed,
            high_noise_steps=req.high_noise_steps or 8,
            shift=req.shift or 8.0,
            sampler_name=req.sampler or "dpmpp_2m",
            scheduler=req.scheduler or "beta",
            lora_configs=lora_dicts,
            aspect_ratio=req.aspect_ratio or "9:16",
        )

    def cost(self, req: GenerationRequest) -> int:
        """
        Cloud Wan22 I2V cost = 2× base Wan22 cost.
        Base: 5 credits short (≤81f), 8 medium (≤161f), 15 long.
        """
        frames = req.frames or 81
        if frames <= 81:
            base = 5
        elif frames <= 161:
            base = 8
        else:
            base = 15

        # Cloud premium multiplier
        return base * 2

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        prompt_id = str(uuid.uuid4())

        submit_fn = self._submit_to_runpod
        if submit_fn is None:
            try:
                from app import _submit_to_runpod

                submit_fn = _submit_to_runpod
            except ImportError:
                raise RuntimeError(
                    "_submit_to_runpod not available — pass it via constructor"
                )

        if not req.input_images:
            raise ValueError("Wan22 I2V requires an input image")

        # Build workflow via ComfyUI client
        workflow = self.build_workflow(req)
        if not workflow:
            raise RuntimeError("Failed to build Cloud Wan22 I2V workflow")

        # Build LoRA download URLs for cloud worker
        lora_dicts = (
            [lr.model_dump(exclude_none=True) for lr in req.loras] if req.loras else []
        )
        from generation import lora_utils

        cloud_lora_downloads = (
            lora_utils.build_lora_download_list(lora_dicts) if lora_dicts else []
        )

        # Images dict for RunPod (filename -> base64 image content)
        input_images_b64 = (
            {"input.png": req.input_images[0]}
            if len(req.input_images) > 0
            else None
        )

        job_info = {
            "user_id": "adapter",
            "prompt": req.prompt[:100],
            "job_type": "cloud_wan22_i2v",
            "num_frames": req.frames or 81,
            "fps": req.fps or 16,
            "steps": req.steps or 15,
            "seed": req.seed,
            "compute_target": "cloud",
        }

        credits_used = self.cost(req)

        result = await submit_fn(
            workflow=workflow,
            user_id="adapter",
            prompt_id=prompt_id,
            job_info=job_info,
            images=input_images_b64,
            lora_downloads=cloud_lora_downloads if cloud_lora_downloads else None,
            prompt_full=req.prompt,
        )

        return GenerationResult(
            prompt_id=result.get("prompt_id", prompt_id),
            status="queued_cloud",
            compute_target=ComputeTarget.CLOUD,
            credits_used=credits_used,
            runpod_job_id=result.get("runpod_job_id"),
            adapter_name=self.name,
            meta=result,
        )
