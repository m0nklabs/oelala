"""
Frame Interpolation adapter — RIFE/FILM via ComfyUI.

Supports two modes: fps (increase to target FPS) and slowmo (multiply frames).
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


class InterpolateAdapter(GenerationAdapter):
    """
    Local frame interpolation via ComfyUI — RIFE or FILM models.

    Two modes:
    - fps: Increase frame rate to target_fps (e.g. 24→60)
    - slowmo: Multiply frame count by multiplier (e.g. 2× slowmo)
    """

    name = "local-interpolate"
    model_family = ""
    supported_ops = {Operation.INTERPOLATE}
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
            allowed_fps=[24, 30, 48, 60, 120],
        )

    def build_workflow(
        self,
        req: GenerationRequest,
        video_name: str = "",
    ) -> dict:
        """Build RIFE interpolation ComfyUI workflow.

        Args:
            req: Generation request.
            video_name: Filename in ComfyUI's input folder (uploaded beforehand).
        """
        mode = req.interpolation_mode or "fps"
        multiplier = req.multiplier or 2.0

        workflow = {
            "1": {
                "inputs": {
                    "video": video_name,
                    "force_rate": 0,
                    "force_size": "Disabled",
                },
                "class_type": "VHS_LoadVideo",
            },
            "2": {
                "inputs": {
                    "ckpt_name": "rife49.pth",
                    "clear_cache_after_n_frames": 10,
                    "multiplier": int(multiplier),
                    "fast_mode": True,
                    "ensemble": True,
                    "scale_factor": 1.0,
                    "frames": ["1", 0],
                },
                "class_type": "RIFE VFI",
            },
            "3": {
                "inputs": {
                    "frame_rate": req.target_fps or 60 if mode == "fps" else 30,
                    "loop_count": 0,
                    "filename_prefix": "oelala_interpolated",
                    "format": "video/h264-mp4",
                    "images": ["2", 0],
                },
                "class_type": "VHS_VideoCombine",
            },
        }

        return workflow

    def cost(self, req: GenerationRequest) -> int:
        return 3

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")

        if not req.input_video:
            raise ValueError("Interpolation requires an input video")

        client = self._get_comfyui()

        # Upload video to ComfyUI input folder and get filename back.
        # Falls back to raw value if upload method is not available
        # (e.g. when the input is already a filename on disk).
        video_name = req.input_video
        if hasattr(client, "upload_video_from_b64"):
            video_name = client.upload_video_from_b64(req.input_video)
        else:
            logger.warning(
                "⚠️ ComfyUI client has no upload_video_from_b64 — "
                "assuming input_video is already a filename"
            )

        prompt_id = client.queue_prompt(self.build_workflow(req, video_name=video_name))

        if not prompt_id:
            raise RuntimeError("Failed to queue interpolation workflow")

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=self.cost(req),
            adapter_name=self.name,
            meta={
                "mode": req.interpolation_mode or "fps",
                "target_fps": req.target_fps or 60,
            },
        )
