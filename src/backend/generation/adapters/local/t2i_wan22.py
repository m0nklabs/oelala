"""
Wan2.2 Local T2I adapter — text-to-image via local ComfyUI.

Uses DisTorch2 multi-GPU setup with high/low noise models.
Optimal at 512×512 to 768×768. Very fast with 8 steps.
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


class Wan22LocalT2IAdapter(GenerationAdapter):
    """
    Wan2.2 text-to-image on local ComfyUI with DisTorch2 multi-GPU.

    Uses dual-pass high/low noise sampling with GGUF quantized models.
    Very fast (8 steps) for quick image generation.
    """

    name = "wan22-local-t2i"
    model_family = "wan2.2"
    supported_ops = {Operation.GENERATE}
    input_types = {MediaType.TEXT}
    output_type = MediaType.IMAGE
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.DUAL_STAGE

    def __init__(self, comfyui_client_fn: Any = None) -> None:
        self._get_comfyui = comfyui_client_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=1024,
            max_height=1024,
            min_width=256,
            min_height=256,
            resolution_step=16,
            aspect_ratios=["1:1", "16:9", "9:16", "4:3", "3:4"],
            min_steps=4,
            max_steps=30,
            default_steps=8,
            default_cfg=7.0,
            max_loras=5,
            supports_negative_prompt=True,
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        """Build Wan2.2 T2I workflow — delegates to comfyui_client."""
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")

        # The full workflow is complex (DisTorch2 multi-GPU).
        # Delegate to ComfyUIClient.generate_wan22_t2i() which
        # handles all the node wiring internally.
        # For build_workflow() we return a minimal stub — the actual
        # execution goes through generate_wan22_t2i().
        return {
            "_adapter": self.name,
            "_note": "Wan22 T2I uses ComfyUIClient.generate_wan22_t2i() for full workflow",
        }

    def cost(self, req: GenerationRequest) -> int:
        return 2  # Wan22 T2I is 2 credits

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")
        client = self._get_comfyui()

        # Wan22 T2I uses the ComfyUI client's built-in method
        # which handles DisTorch2 multi-GPU workflow construction.
        result_path = client.generate_wan22_t2i(
            prompt=req.prompt,
            output_dir="/tmp/oelala_generated",
            width=req.width or 512,
            height=req.height or 512,
            steps=req.steps or 8,
            seed=req.seed,
        )

        if not result_path:
            raise RuntimeError("Failed to generate Wan22 T2I image")

        return GenerationResult(
            prompt_id=result_path,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=self.cost(req),
            adapter_name=self.name,
            meta={"width": req.width or 512, "height": req.height or 512},
        )
