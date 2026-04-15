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

    def build_workflow(self, req: GenerationRequest) -> dict:
        """Build RIFE interpolation ComfyUI workflow."""
        return {
            "_adapter": self.name,
            "_mode": req.interpolation_mode or "fps",
            "_target_fps": req.target_fps or 60,
        }

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
        prompt_id = client.queue_prompt(self.build_workflow(req))

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
