"""
Voice Clone adapter — F5-TTS via ComfyUI.

Uses F5TTSAudio node with various model variants (F5v1, F5-DE, E2, etc.).
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

# F5-TTS model type mapping
F5_MODEL_MAP = {
    "F5v1": "F5v1",
    "F5": "F5",
    "F5-DE": "F5-DE",
    "F5-FR": "F5-FR",
    "F5-ZH": "F5-ZH",
    "E2": "E2",
}


class VoiceCloneAdapter(GenerationAdapter):
    """
    Local voice cloning via ComfyUI F5-TTS node.

    Takes a voice sample + text input, generates cloned speech audio.
    """

    name = "local-voice-clone"
    model_family = ""
    supported_ops = {Operation.GENERATE}
    input_types = {MediaType.AUDIO}
    output_type = MediaType.AUDIO
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.NONE

    def __init__(self, comfyui_client_fn: Any = None) -> None:
        self._get_comfyui = comfyui_client_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_duration_seconds=120.0,
            supports_negative_prompt=False,
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        """Build F5-TTS workflow."""
        model_type = F5_MODEL_MAP.get(req.checkpoint or "F5v1", "F5v1")

        workflow = {
            "1": {
                "inputs": {
                    "model_type": model_type,
                    "gen_text": req.prompt,
                    "ref_audio": req.voice_sample_path or req.input_audio or "",
                    "speed": req.speed or 1.0,
                    "seed": req.seed,
                },
                "class_type": "F5TTSAudio",
            },
            "2": {
                "inputs": {"filename_prefix": "oelala_voice", "audio": ["1", 0]},
                "class_type": "SaveAudio",
            },
        }

        return workflow

    def cost(self, req: GenerationRequest) -> int:
        return 20  # Voice cloning is expensive

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")

        client = self._get_comfyui()
        workflow = self.build_workflow(req)
        prompt_id = client.queue_prompt(workflow)

        if not prompt_id:
            raise RuntimeError("Failed to queue voice clone workflow")

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=self.cost(req),
            adapter_name=self.name,
            meta={"model": req.checkpoint or "F5v1"},
        )
