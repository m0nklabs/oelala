"""
ERNIE-Image Local T2I adapter — text-to-image via local ComfyUI.

Uses Flux2 sampling pipeline (SamplerCustomAdvanced + Flux2Scheduler).
NOTE: DisTorch2 multi-GPU produces NaN with ERNIE — plain loaders only.
ERNIE-Image: Flux2 latent format, Ministral-3-3B text encoder,
FlowMatch sampling. Optimal at 1024x1024.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from ...adapter import GenerationAdapter, ProgressCallback
from ...types import (
    AdapterConstraints,
    ComputeTarget,
    GenerationRequest,
    GenerationResult,
    MediaType,
    Operation,
)

logger = logging.getLogger(__name__)


class ErnieLocalT2IAdapter(GenerationAdapter):
    """
    ERNIE-Image text-to-image on local ComfyUI.
    Uses Flux2 sampling pipeline with ComfyUI's dynamic VRAM offloading.

    Uses Ministral-3-3B text encoder, Flux2 VAE, FlowMatch sampling.
    Best at 1024x1024, supports various aspect ratios.
    NOTE: DisTorch2 multi-GPU produces NaN — plain loaders only.
    """

    name = "ernie-local-t2i"
    model_family = "ernie"
    supported_ops = {Operation.GENERATE}
    input_types = {MediaType.TEXT}
    output_type = MediaType.IMAGE
    compute = ComputeTarget.LOCAL

    def __init__(self, comfyui_client_fn: Any = None) -> None:
        self._get_comfyui = comfyui_client_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=1536,
            max_height=1536,
            min_width=512,
            min_height=512,
            resolution_step=16,
            aspect_ratios=["1:1", "16:9", "9:16", "4:3", "3:4"],
            min_steps=10,
            max_steps=100,
            default_steps=20,
            default_cfg=4.0,
            supports_negative_prompt=True,
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        """Build ERNIE-Image T2I workflow — delegates to comfyui_client."""
        return {
            "_adapter": self.name,
            "_note": "ERNIE T2I uses ComfyUIClient.generate_ernie_t2i() for full workflow",
        }

    def cost(self, req: GenerationRequest) -> int:
        return 3  # ERNIE is heavier than Wan22 T2I

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")
        client = self._get_comfyui()

        result_path = client.generate_ernie_t2i(
            prompt=req.prompt,
            output_dir=str(Path(__file__).resolve().parents[5] / "media" / "generated"),
            width=req.width or 1024,
            height=req.height or 1024,
            steps=req.steps or 50,
            guidance=req.cfg or 4.0,
            seed=req.seed,
        )

        if not result_path:
            raise RuntimeError("ERNIE-Image T2I generation failed — no output")

        return GenerationResult(
            prompt_id=str(uuid.uuid4()),
            status="completed",
            compute_target=ComputeTarget.LOCAL,
            credits_used=0,
            adapter_name=self.name,
            meta={
                "width": req.width or 1024,
                "height": req.height or 1024,
                "result_path": result_path,
            },
        )
