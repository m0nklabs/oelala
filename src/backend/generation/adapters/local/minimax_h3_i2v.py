"""
MiniMax-H3 Local I2V adapter — image-to-video (+audio) via the user's Windows PC ComfyUI.

Runs the MiniMax-H3 FL2VA workflow on the Windows PC's ComfyUI server
(get_windows_comfyui_client()), separate from the default ai-kvm2 ComfyUI.
The input image is anchored as the first keyframe. Because it targets a
different server, this adapter uploads the input image itself
(handles_own_image_upload=True) and the router skips its default pre-upload.
"""

from __future__ import annotations

import base64
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
        if not src.resolve().is_relative_to(_MINIMAX_H3_LORA_DIR.resolve()):
            logger.warning(f"🎨 Refusing out-of-directory LoRA path: {name}")
            continue
        if not src.is_file():
            logger.warning(f"🎨 MiniMax-H3 LoRA not found, skipping: {src}")
            continue
        uploaded = client.upload_lora(str(src))
        if not uploaded:
            logger.warning(f"🎨 MiniMax-H3 LoRA upload failed, skipping: {name}")


class MiniMaxH3LocalI2VAdapter(GenerationAdapter):
    """
    MiniMax-H3 22B Image-to-Video on the Windows PC ComfyUI.

    Same FL2VA checkpoint as T2V; the input image becomes the first keyframe.
    24 fps, 17k+5 frame grid. Audio is generated unconditionally.
    """

    name = "minimax-h3-local-i2v"
    model_family = "minimax_h3"
    supported_ops = {Operation.GENERATE}
    input_types = {MediaType.IMAGE}
    output_type = MediaType.VIDEO
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.SINGLE_STAGE
    handles_own_image_upload = True

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
            max_input_images=1,
            allowed_fps=[24],
        )

    @staticmethod
    def _to_png_bytes(img: str) -> bytes:
        """Decode a base64 (or data-URI) image string into PNG bytes."""
        raw = img
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[-1]
        raw = "".join(raw.split())
        missing = (-len(raw)) % 4
        if missing:
            raw = raw + ("=" * missing)
        return base64.b64decode(raw)

    def build_workflow(self, req: GenerationRequest, image_name: str = "") -> dict:
        """Build the local MiniMax-H3 I2V ComfyUI workflow.

        ``image_name`` is the filename the input image was uploaded as on the
        target server (it is read back from the upload in ``execute``). It
        defaults to a placeholder so workflow-only callers can still build one.
        """
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")
        comfyui = self._get_comfyui()
        lora_configs = (
            [lr.model_dump(exclude_none=True) for lr in req.loras]
            if req.loras
            else None
        )
        return comfyui.build_local_minimax_h3_i2v_workflow(
            image_name=image_name or "input.png",
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
            return 5
        elif frames <= 210:
            return 8
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

        if not req.input_images:
            raise ValueError("MiniMax-H3 local I2V requires an input image")

        if not client.is_available():
            raise RuntimeError(
                f"ComfyUI backend at {client.host}:{client.port} is not reachable — "
                "is the configured compute backend running on that machine?"
            )

        # Upload the input image to the resolved server (router pre-upload skipped).
        try:
            img_bytes = self._to_png_bytes(req.input_images[0])
            uploaded_name = client.upload_image_from_bytes(
                img_bytes, filename="v2_minimax_i2v_input.png"
            )
        except Exception as e:
            logger.exception("❌ Failed to decode/upload MiniMax-H3 I2V image")
            raise RuntimeError(f"Failed to upload MiniMax-H3 I2V image: {e}")

        if not uploaded_name:
            raise RuntimeError("ComfyUI image upload failed on Windows server")

        _upload_minimax_loras(client, req)

        workflow = self.build_workflow(req, image_name=uploaded_name)
        if not workflow:
            raise RuntimeError("Failed to build MiniMax-H3 local I2V workflow")

        prompt_id = client.queue_prompt(workflow)
        if not prompt_id:
            raise RuntimeError("Failed to queue MiniMax-H3 local I2V to ComfyUI")

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=0,
            adapter_name=self.name,
            meta={
                "frames": req.frames or 124,
                "fps": req.fps or 24,
                "image": uploaded_name,
                "server": f"{client.host}:{client.port}",
            },
        )
