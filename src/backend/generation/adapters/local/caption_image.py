"""
Image Caption adapter — image-to-text via Guardian vision LLM.

Multiple modes: brief, detailed, tags, structured, prompt_i2v, etc.
Uses Guardian LLM vision API (not ComfyUI).
"""

from __future__ import annotations

import logging
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


class ImageCaptionAdapter(GenerationAdapter):
    """
    Image captioning via Guardian vision LLM.

    Does NOT use ComfyUI — calls Guardian vision API directly.
    Supports multiple captioning modes with configurable detail levels.
    """

    name = "local-caption-image"
    model_family = ""
    supported_ops = {Operation.CAPTION}
    input_types = {MediaType.IMAGE}
    output_type = MediaType.TEXT
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.NONE

    def __init__(self, guardian_client_fn: Any = None) -> None:
        self._get_guardian = guardian_client_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=4096,
            max_height=4096,
            min_width=64,
            min_height=64,
            max_input_images=1,
            supports_negative_prompt=False,
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        """No ComfyUI workflow — uses Guardian vision LLM."""
        return {"_adapter": self.name, "_engine": "guardian_vision"}

    def cost(self, req: GenerationRequest) -> int:
        return 1

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        if self._get_guardian is None:
            raise RuntimeError("Guardian client not available")

        if not req.input_images:
            raise ValueError("Image captioning requires an input image")

        guardian = self._get_guardian()
        caption = await guardian.caption_image(
            image_data=req.input_images[0],
            mode=req.caption_mode or "brief",
            detail_level=req.detail_level,
        )

        if not caption:
            raise RuntimeError("Image captioning failed")

        return GenerationResult(
            prompt_id=str(uuid.uuid4()),
            status="completed",
            compute_target=ComputeTarget.LOCAL,
            credits_used=0,  # Router fills this in
            adapter_name=self.name,
            meta={
                "caption": caption,
                "mode": req.caption_mode or "brief",
            },
        )
