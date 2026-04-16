"""
Lip Sync adapter — LatentSync via ComfyUI.

Synchronizes lip movements in video to match audio input.
Uses VHS_LoadVideo + LoadAudio + LatentSyncNode pipeline.
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


class LipSyncAdapter(GenerationAdapter):
    """
    Local lip sync via ComfyUI LatentSync.

    Takes video + audio input, synchronizes lip movements to match speech.
    Forces output to 25fps as required by LatentSync model.
    """

    name = "local-lipsync"
    model_family = ""
    supported_ops = {Operation.LIPSYNC}
    input_types = {MediaType.VIDEO}
    output_type = MediaType.VIDEO
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.NONE

    def __init__(self, comfyui_client_fn: Any = None) -> None:
        self._get_comfyui = comfyui_client_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=1920,
            max_height=1080,
            min_width=256,
            min_height=256,
            allowed_fps=[25],  # LatentSync requires 25fps
            min_steps=5,
            max_steps=50,
            default_steps=20,
            supports_negative_prompt=False,
        )

    def build_workflow(
        self,
        req: GenerationRequest,
        video_name: str = "",
        audio_name: str = "",
    ) -> dict:
        """Build LatentSync workflow.

        Args:
            req: Generation request.
            video_name: Filename in ComfyUI's input folder (uploaded beforehand).
            audio_name: Filename in ComfyUI's input folder (uploaded beforehand).
        """
        workflow = {
            "1": {
                "inputs": {
                    "video": video_name,
                    "force_rate": 25,
                    "force_size": "Disabled",
                },
                "class_type": "VHS_LoadVideo",
            },
            "2": {
                "inputs": {"audio": audio_name},
                "class_type": "LoadAudio",
            },
            "3": {
                "inputs": {
                    "video": ["1", 0],
                    "audio": ["2", 0],
                    "lips_expression": req.lips_expression,
                    "inference_steps": req.inference_steps or 20,
                    "seed": req.seed,
                },
                "class_type": "LatentSyncNode",
            },
            "4": {
                "inputs": {
                    "frame_rate": 25,
                    "filename_prefix": "oelala_lipsync",
                    "images": ["3", 0],
                    "audio": ["2", 0],
                },
                "class_type": "VHS_VideoCombine",
            },
        }

        return workflow

    def cost(self, req: GenerationRequest) -> int:
        return 5

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")

        if not req.input_video:
            raise ValueError("Lip sync requires an input video")
        if not req.input_audio:
            raise ValueError("Lip sync requires input audio")

        client = self._get_comfyui()

        # Upload video/audio to ComfyUI input folder and get filenames back.
        # Falls back to raw value if upload methods are not available
        # (e.g. when the inputs are already filenames on disk).
        video_name = req.input_video
        if hasattr(client, "upload_video_from_b64"):
            video_name = client.upload_video_from_b64(req.input_video)
        else:
            logger.warning(
                "⚠️ ComfyUI client has no upload_video_from_b64 — "
                "assuming input_video is already a filename"
            )

        audio_name = req.input_audio
        if hasattr(client, "upload_audio_from_b64"):
            audio_name = client.upload_audio_from_b64(req.input_audio)
        else:
            logger.warning(
                "⚠️ ComfyUI client has no upload_audio_from_b64 — "
                "assuming input_audio is already a filename"
            )

        workflow = self.build_workflow(req, video_name=video_name, audio_name=audio_name)
        prompt_id = client.queue_prompt(workflow)

        if not prompt_id:
            raise RuntimeError("Failed to queue lip sync workflow")

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=self.cost(req),
            adapter_name=self.name,
            meta={"lips_expression": req.lips_expression},
        )
