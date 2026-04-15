"""
Wan2.2 Cloud T2V adapter — text-to-video via RunPod.

Delegates workflow building to ComfyUIClient.build_cloud_wan22_t2v_workflow().
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


class Wan22CloudT2VAdapter(GenerationAdapter):
    """
    Wan 2.2 Text-to-Video on RunPod (48GB+ GPU).

    Uses fp8_scaled dual-pass sampling with dedicated high/low noise T2V models.
    Pixel-frame budget capped at 100M to prevent OOM on cloud GPUs.
    """

    name = "wan22-cloud-t2v"
    model_family = "wan2.2"
    supported_ops = {Operation.GENERATE}
    input_types = {MediaType.TEXT}
    output_type = MediaType.VIDEO
    compute = ComputeTarget.CLOUD
    lora_format = LoraFormat.DUAL_STAGE

    def __init__(
        self,
        submit_to_runpod_fn: Any = None,
        comfyui_client_fn: Any = None,
    ) -> None:
        self._submit_to_runpod = submit_to_runpod_fn
        self._get_comfyui = comfyui_client_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=1920,
            max_height=1080,
            min_width=480,
            min_height=480,
            max_frames=161,
            resolution_step=16,
            resolution_presets=["480p", "576p", "720p"],
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

        return comfyui.build_cloud_wan22_t2v_workflow(
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
        Cloud Wan22 T2V cost = 2× base cost.
        T2V base: 8 credits short (≤81f), 12 medium (≤161f).
        """
        frames = req.frames or 81
        if frames <= 81:
            base = 8
        else:
            base = 12

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

        # Validate pixel-frame budget for T2V
        width = req.width or 720
        height = req.height or 1280
        frames = req.frames or 81
        budget = width * height * frames
        max_budget = 100_000_000
        if budget > max_budget:
            raise ValueError(
                f"Cloud Wan22 T2V exceeds safety budget ({budget:,} > {max_budget:,}). "
                "Lower duration or resolution."
            )

        workflow = self.build_workflow(req)
        if not workflow:
            raise RuntimeError("Failed to build Cloud Wan22 T2V workflow")

        lora_dicts = (
            [lr.model_dump(exclude_none=True) for lr in req.loras] if req.loras else []
        )
        from generation import lora_utils

        cloud_lora_downloads = (
            lora_utils.build_lora_download_list(lora_dicts) if lora_dicts else []
        )

        job_info = {
            "user_id": "adapter",
            "prompt": req.prompt[:100],
            "job_type": "cloud_wan22_t2v",
            "num_frames": frames,
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
