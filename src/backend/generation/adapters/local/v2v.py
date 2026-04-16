"""
Video-to-Video adapter — style transfer via first-frame I2V.

Extracts first frame → runs I2V with style prompt → V2V output.
Uses ComfyUIClient.generate_distorch2_video() for actual processing.
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


class V2VStyleTransferAdapter(GenerationAdapter):
    """
    Local Video-to-Video style transfer via ComfyUI.

    Extracts the first frame from input video, applies I2V generation
    with the style prompt, producing a style-transferred video.
    """

    name = "local-v2v"
    model_family = "wan2.2"
    supported_ops = {Operation.TRANSFORM}
    input_types = {MediaType.VIDEO}
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
            max_frames=161,
            resolution_step=16,
            resolution_presets=["480p", "576p", "720p"],
            aspect_ratios=["9:16", "16:9", "1:1", "4:3", "3:4"],
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
        return {
            "_adapter": self.name,
            "_note": "V2V delegates to generate_distorch2_video()",
        }

    def cost(self, req: GenerationRequest) -> int:
        frames = req.frames or 41
        if frames <= 81:
            return 5
        elif frames <= 161:
            return 8
        return 15

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")

        if not req.input_video:
            raise ValueError("V2V requires an input video")

        client = self._get_comfyui()

        result = client.generate_distorch2_video(
            video_data=req.input_video,
            style_prompt=req.prompt,
            mode=req.v2v_mode or "style_transfer",
            strength=req.strength or 0.5,
            num_frames=req.frames or 41,
            resolution=req.resolution or "480p",
            fps=req.fps or 16,
            preserve_motion=req.preserve_motion,
            seed=req.seed,
        )

        if not result:
            raise RuntimeError("Failed to generate V2V video")

        return GenerationResult(
            prompt_id=result if isinstance(result, str) else str(result),
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=0,  # Router fills this in
            adapter_name=self.name,
            meta={"v2v_mode": req.v2v_mode or "style_transfer"},
        )
