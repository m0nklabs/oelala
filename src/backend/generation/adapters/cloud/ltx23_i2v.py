"""
LTX-2.3 Cloud I2V adapter — image-to-video via RunPod.

Delegates workflow building to ComfyUIClient.build_cloud_ltx23_i2v_workflow().
Uses single-stage LoRAs. Requires 80GB+ GPU.
Supports optional audio-video generation via audio_prompt.
"""

from __future__ import annotations

import logging
import os
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


class LTX23CloudI2VAdapter(GenerationAdapter):
    """
    LTX-2.3 22B Image-to-Video on RunPod (80GB+ GPU).

    Single-stage distilled pipeline using 8-step sigma schedule.
    Gemma 3 12B text encoder. Optional audio-video generation.
    """

    name = "ltx23-cloud-i2v"
    model_family = "ltx"
    supported_ops = {Operation.GENERATE}
    input_types = {MediaType.IMAGE}
    output_type = MediaType.VIDEO
    compute = ComputeTarget.CLOUD
    lora_format = LoraFormat.SINGLE_STAGE

    def __init__(
        self,
        submit_to_runpod_fn: Any = None,
        comfyui_client_fn: Any = None,
    ) -> None:
        self._submit_to_runpod = submit_to_runpod_fn
        self._get_comfyui = comfyui_client_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=1280,
            max_height=768,
            min_width=256,
            min_height=256,
            max_frames=257,
            resolution_step=32,  # LTX requires divisible by 32
            resolution_presets=["480p", "576p", "720p"],
            aspect_ratios=[
                "9:16",
                "16:9",
                "1:1",
                "4:3",
                "3:4",
                "3:2",
                "2:3",
                "21:9",
                "9:21",
            ],
            min_steps=8,
            max_steps=50,
            default_steps=20,
            default_cfg=3.0,
            max_loras=5,
            supports_lightning=False,
            supports_negative_prompt=True,
            max_input_images=1,
            allowed_fps=[25],
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

        return comfyui.build_cloud_ltx23_i2v_workflow(
            image_name=req.input_images[0] if req.input_images else "input.png",
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width or 768,
            height=req.height or 512,
            num_frames=req.frames or 97,
            fps=req.fps or 25,
            seed=req.seed,
            strength=req.strength or 1.0,
            lora_configs=lora_dicts,
            audio_prompt=req.audio_prompt if req.audio_prompt else None,
        )

    def cost(self, req: GenerationRequest) -> int:
        """
        LTX-2.3 I2V — same credit formula as Wan22 I2V for now.
        """
        frames = req.frames or 97
        if frames <= 97:
            return 5
        elif frames <= 161:
            return 8
        else:
            return 15

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        prompt_id = str(uuid.uuid4())
        endpoint_id = os.getenv("RUNPOD_LTX23_ENDPOINT_ID")

        if not endpoint_id:
            raise RuntimeError(
                "LTX-2.3 I2V requires RUNPOD_LTX23_ENDPOINT_ID to be configured"
            )

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
            raise ValueError("LTX-2.3 I2V requires an input image")

        workflow = self.build_workflow(req)
        if not workflow:
            raise RuntimeError("Failed to build LTX-2.3 I2V cloud workflow")

        lora_dicts = (
            [lr.model_dump(exclude_none=True) for lr in req.loras] if req.loras else []
        )

        from ... import lora_utils
        from ...lora_utils import sanitize_lora_configs_for_single_stage

        # LTX uses single-stage — sanitize any dual-stage configs
        lora_dicts = sanitize_lora_configs_for_single_stage(lora_dicts)
        cloud_lora_downloads = (
            lora_utils.build_lora_download_list(lora_dicts) if lora_dicts else []
        )

        input_images_b64 = (
            {"input.png": req.input_images[0]} if len(req.input_images) > 0 else None
        )

        job_info = {
            "user_id": "adapter",
            "prompt": req.prompt[:100],
            "job_type": "ltx23_i2v",
            "model": "ltx23",
            "num_frames": req.frames or 97,
            "fps": req.fps or 25,
            "seed": req.seed,
            "compute_target": "cloud",
        }


        result = await submit_fn(
            workflow=workflow,
            user_id="adapter",
            prompt_id=prompt_id,
            job_info=job_info,
            images=input_images_b64,
            lora_downloads=cloud_lora_downloads if cloud_lora_downloads else None,
            prompt_full=req.prompt,
            endpoint_id=endpoint_id,
        )

        return GenerationResult(
            prompt_id=result.get("prompt_id", prompt_id),
            status="queued_cloud",
            compute_target=ComputeTarget.CLOUD,
            credits_used=0,  # Router fills this in
            runpod_job_id=result.get("runpod_job_id"),
            adapter_name=self.name,
            meta=result,
        )
