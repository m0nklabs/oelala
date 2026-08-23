"""
MiniMax-H3 Local T2V adapter — text-to-video (+audio) via the user's Windows PC ComfyUI.

Runs the MiniMax-H3 FL2VA workflow on the Windows PC's ComfyUI server
(get_windows_comfyui_client()), which is a DIFFERENT server from the default
ComfyUI on ai-kvm2 (get_comfyui_client()). It uses the int8_convrot model set
that was downloaded onto that machine (see download_minimax_h3.* and
README_MiniMax_H3_workflow.md). The model always generates a synchronized
soundtrack — no separate audio prompt needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
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

# LoRAs for the local MiniMax-H3 (Windows-PC ComfyUI) live here on this
# server and are uploaded to the Windows-PC on demand before dispatch.
_MINIMAX_H3_LORA_DIR = Path("/mnt/ssd/loras/minimax-h3")


def _upload_minimax_loras(client: Any, req: GenerationRequest) -> None:
    """Upload any requested MiniMax-H3 LoRAs to the Windows-PC ComfyUI server.

    ``LoraStackItem.name`` is the LoRA filename as the server knows it, so each
    requested LoRA is uploaded from the local ``minimax-h3`` folder (by that
    basename) before the workflow runs. Missing files are warned about and
    skipped; the run proceeds without them rather than failing the whole job.
    """
    if not req.loras:
        return
    for lora in req.loras:
        name = (lora.name or "").strip()
        if not name or name == "None":
            continue
        src = _MINIMAX_H3_LORA_DIR / name
        if not src.is_file():
            logger.warning(f"🎨 MiniMax-H3 LoRA not found, skipping: {src}")
            continue
        uploaded = client.upload_lora(str(src))
        if not uploaded:
            logger.warning(f"🎨 MiniMax-H3 LoRA upload failed, skipping: {name}")


class MiniMaxH3LocalT2VAdapter(GenerationAdapter):
    """
    MiniMax-H3 22B Text-to-Video on the Windows PC ComfyUI.

    Joint video+audio DiT. 24 fps, 17k+5 frame grid (~5s at 124 frames).
    int8+convrot diffusion checkpoint + int8_convrot Qwen3-VL-32B text
    encoder (the Comfy-Org int8 pruned pack for 16 GB GPUs).
    """

    name = "minimax-h3-local-t2v"
    model_family = "minimax_h3"
    supported_ops = {Operation.GENERATE}
    input_types = {MediaType.TEXT}
    output_type = MediaType.VIDEO
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.SINGLE_STAGE

    def __init__(self, comfyui_client_fn: Any = None) -> None:
        self._get_comfyui = comfyui_client_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=1920,
            max_height=1920,
            min_width=256,
            min_height=256,
            max_frames=362,
            resolution_step=32,
            resolution_presets=[],
            aspect_ratios=[
                "9:16",
                "16:9",
                "1:1",
                "4:3",
                "3:4",
                "3:2",
                "2:3",
                "21:9",
                "9:21",
            ],
            min_steps=8,
            max_steps=50,
            default_steps=20,
            default_cfg=1.0,
            max_loras=5,
            supports_lightning=False,
            supports_negative_prompt=False,
            allowed_fps=[24],
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")
        comfyui = self._get_comfyui()
        lora_configs = (
            [lr.model_dump(exclude_none=True) for lr in req.loras] if req.loras else None
        )
        return comfyui.build_local_minimax_h3_t2v_workflow(
            prompt=req.prompt,
            num_frames=req.frames or 124,
            fps=req.fps or 24,
            seed=req.seed,
            steps=req.steps or 20,
            aspect_ratio=req.aspect_ratio or "16:9",
            megapixels=req.megapixels,
            lora_configs=lora_configs,
        )

    def cost(self, req: GenerationRequest) -> int:
        frames = req.frames or 124
        if frames <= 124:
            return 8
        elif frames <= 210:
            return 12
        else:
            return 15

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")
        client = self._get_comfyui()
        if client is None:
            raise RuntimeError("No enabled local ComfyUI backend for minimax_h3")

        if not client.is_available():
            raise RuntimeError(
                f"ComfyUI backend at {client.host}:{client.port} is not reachable — "
                "is the configured compute backend running on that machine?"
            )

        workflow = self.build_workflow(req)
        if not workflow:
            raise RuntimeError("Failed to build MiniMax-H3 local T2V workflow")

        _upload_minimax_loras(client, req)

        prompt_id = client.queue_prompt(workflow)
        if not prompt_id:
            raise RuntimeError("Failed to queue MiniMax-H3 local T2V to ComfyUI")

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=0,
            adapter_name=self.name,
            meta={
                "frames": req.frames or 124,
                "fps": req.fps or 24,
                "server": f"{client.host}:{client.port}",
            },
        )
