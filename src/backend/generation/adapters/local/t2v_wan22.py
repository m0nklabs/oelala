"""
Wan2.2 Local T2V adapter — text-to-video via local ComfyUI.

Delegates to ComfyUIClient.build_t2v_q6_workflow().
Uses dual-pass Q6_K GGUF models with DisTorch2 multi-GPU.
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


class Wan22LocalT2VQ6Adapter(GenerationAdapter):
    """
    Wan2.2 T2V Q6_K on local ComfyUI with DisTorch2 multi-GPU.

    Pure text-to-video, no image conditioning. Uses dedicated T2V
    high/low noise GGUF models for dual-pass sampling.
    """

    name = "wan22-local-t2v-q6"
    model_family = "wan2.2"
    supported_ops = {Operation.GENERATE}
    input_types = {MediaType.TEXT}
    output_type = MediaType.VIDEO
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.DUAL_STAGE

    def __init__(self, comfyui_client_fn: Any = None) -> None:
        self._get_comfyui = comfyui_client_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=1280,
            max_height=720,
            min_width=480,
            min_height=480,
            max_frames=321,
            resolution_step=16,
            resolution_presets=["480p", "576p", "720p"],
            aspect_ratios=["9:16", "16:9", "1:1", "4:3", "3:4", "3:2", "2:3"],
            min_steps=4,
            max_steps=30,
            default_steps=6,
            default_cfg=1.0,
            supported_samplers=["uni_pc", "euler", "dpmpp_2m"],
            supported_schedulers=["normal", "simple", "karras"],
            max_loras=5,
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

        return comfyui.build_t2v_q6_workflow(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_frames=req.frames or 81,
            fps=req.fps or 16,
            steps=req.steps or 6,
            cfg=req.cfg or 1.0,
            seed=req.seed,
            high_noise_steps=req.high_noise_steps or 3,
            sampler_name=req.sampler or "uni_pc",
            scheduler=req.scheduler or "normal",
            aspect_ratio=req.aspect_ratio or "9:16",
            lora_configs=lora_dicts,
        )

    def cost(self, req: GenerationRequest) -> int:
        """Local T2V cost — slightly higher than I2V."""
        frames = req.frames or 81
        if frames <= 81:
            return 8
        elif frames <= 161:
            return 12
        else:
            return 15

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")

        workflow = self.build_workflow(req)
        if not workflow:
            raise RuntimeError("Failed to build Wan22 T2V Q6 workflow")

        client = self._get_comfyui()
        prompt_id = client.queue_prompt(workflow)

        if not prompt_id:
            raise RuntimeError("Failed to queue Wan22 T2V workflow to ComfyUI")

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=self.cost(req),
            adapter_name=self.name,
            meta={
                "frames": req.frames or 81,
                "fps": req.fps or 16,
            },
        )
