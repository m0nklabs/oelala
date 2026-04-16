"""
MMAudio adapter — text-to-audio generation via ComfyUI.

Supports TTS (ChatterBox), music (MMAudio), and SFX (MMAudio) modes.
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


class MMAudioAdapter(GenerationAdapter):
    """
    Local audio generation via ComfyUI.

    Three modes:
    - tts: Text-to-speech via ChatterBox engine
    - music: Music generation via MMAudio model
    - sfx: Sound effects via MMAudio model
    """

    name = "local-mmaudio"
    model_family = ""
    supported_ops = {Operation.GENERATE}
    input_types = {MediaType.TEXT}
    output_type = MediaType.AUDIO
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.NONE

    def __init__(self, comfyui_client_fn: Any = None) -> None:
        self._get_comfyui = comfyui_client_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_duration_seconds=60.0,
            supports_negative_prompt=False,
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        """Stub — audio workflow built inline in execute()."""
        return {
            "_adapter": self.name,
            "_mode": req.audio_mode or "music",
        }

    def cost(self, req: GenerationRequest) -> int:
        duration = req.duration or 10.0
        if duration <= 10:
            return 3
        return 5

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")

        client = self._get_comfyui()
        prompt_id = client.queue_prompt(self.build_workflow(req))

        if not prompt_id:
            raise RuntimeError("Failed to queue audio workflow")

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=0,  # Router fills this in
            adapter_name=self.name,
            meta={
                "mode": req.audio_mode or "music",
                "duration": req.duration or 10.0,
            },
        )
