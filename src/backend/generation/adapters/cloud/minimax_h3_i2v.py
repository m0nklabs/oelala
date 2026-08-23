"""
MiniMax-H3 Cloud I2V adapter — image-to-video (+audio) via RunPod.

Delegates workflow building to ComfyUIClient.build_cloud_minimax_h3_i2v_workflow().
The input image is anchored as the first keyframe of the video. MiniMax-H3
(FL2VA) always generates a synchronized soundtrack. Requires 80GB+ GPU.
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


class MiniMaxH3CloudI2VAdapter(GenerationAdapter):
    """
    MiniMax-H3 22B Image-to-Video on RunPod (80GB+ GPU).

    Same FL2VA checkpoint as T2V; the input image becomes the first
    keyframe. 24 fps, 17k+5 frame grid. Audio is generated unconditionally.
    """

    name = "minimax-h3-cloud-i2v"
    model_family = "minimax_h3"
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
            max_width=1920,  # 2K via the official MP selector (2.0 MP @16:9 → 1920×1088)
            max_height=1920,
            min_width=256,
            min_height=256,
            max_frames=362,  # trained range is ~124-362 at 24 fps
            resolution_step=32,
            resolution_presets=[],  # H3 sizes are set via megapixels, not named presets
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
            default_cfg=1.0,
            max_loras=5,
            supports_lightning=False,
            supports_negative_prompt=False,  # MiniMax-H3 has no negative prompt
            max_input_images=1,
            allowed_fps=[24],
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")
        comfyui = self._get_comfyui()

        return comfyui.build_cloud_minimax_h3_i2v_workflow(
            image_name="input.png" if req.input_images else "",
            prompt=req.prompt,
            width=req.width or 1344,
            height=req.height or 768,
            num_frames=req.frames or 124,
            fps=req.fps or 24,
            seed=req.seed,
            steps=req.steps or 20,
            aspect_ratio=req.aspect_ratio or "16:9",
            megapixels=req.megapixels,
        )

    def cost(self, req: GenerationRequest) -> int:
        """MiniMax-H3 I2V cost — same credit formula as LTX-2.3 I2V."""
        frames = req.frames or 124
        if frames <= 124:
            return 5
        elif frames <= 210:
            return 8
        else:
            return 15

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        prompt_id = str(uuid.uuid4())
        endpoint_id = os.getenv("RUNPOD_MINIMAX_H3_ENDPOINT_ID")

        if not endpoint_id:
            raise RuntimeError(
                "MiniMax-H3 I2V requires RUNPOD_MINIMAX_H3_ENDPOINT_ID to be configured"
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
            raise ValueError("MiniMax-H3 I2V requires an input image")

        workflow = self.build_workflow(req)
        if not workflow:
            raise RuntimeError("Failed to build MiniMax-H3 I2V cloud workflow")

        input_images_b64 = (
            {"input.png": req.input_images[0]} if len(req.input_images) > 0 else None
        )

        job_info = {
            "user_id": req.user_id or "adapter",
            "prompt": req.prompt[:100],
            "job_type": "minimax_h3_i2v",
            "model": "minimax_h3",
            "num_frames": req.frames or 124,
            "steps": req.steps or 20,
            "cfg": req.cfg or 1.0,
            "fps": req.fps or 24,
            "seed": req.seed,
            "compute_target": "cloud",
        }

        result = await submit_fn(
            workflow=workflow,
            user_id=req.user_id or "adapter",
            prompt_id=prompt_id,
            job_info=job_info,
            images=input_images_b64,
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
