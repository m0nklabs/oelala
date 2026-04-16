"""
Video Caption adapter — video-to-text via frame extraction + VLM.

Extracts key frames via OpenCV, sends to vision-language model for captioning.
Multiple modes: brief, detailed, timeline, prompt.
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


class VideoCaptionAdapter(GenerationAdapter):
    """
    Video captioning via frame extraction + vision-language model.

    Extracts frames at configurable intervals, sends to VLM (SmolVLM/CogVLM/LLaVA).
    Does NOT use ComfyUI — uses Guardian/OpenAI vision API.
    """

    name = "local-caption-video"
    model_family = ""
    supported_ops = {Operation.CAPTION}
    input_types = {MediaType.VIDEO}
    output_type = MediaType.TEXT
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.NONE

    def __init__(self, guardian_client_fn: Any = None) -> None:
        self._get_guardian = guardian_client_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=3840,
            max_height=2160,
            min_width=64,
            min_height=64,
            supports_negative_prompt=False,
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        """No ComfyUI workflow — uses frame extraction + VLM."""
        return {"_adapter": self.name, "_engine": "guardian_vision"}

    def cost(self, req: GenerationRequest) -> int:
        return 2

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        if self._get_guardian is None:
            raise RuntimeError("Guardian client not available")

        if not req.input_video:
            raise ValueError("Video captioning requires an input video")

        guardian = self._get_guardian()
        caption = await guardian.caption_video(
            video_data=req.input_video,
            mode=req.caption_mode or "brief",
            frame_interval=req.frame_interval or 1.0,
            max_frames=req.max_frames or 8,
        )

        if not caption:
            raise RuntimeError("Video captioning failed")

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
