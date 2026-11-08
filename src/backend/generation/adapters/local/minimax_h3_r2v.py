"""
MiniMax-H3 Local R2V adapter — reference-to-video (+audio) via a remote ComfyUI node.

Runs the MiniMax-H3 REF2VA workflow on the enabled 'comfyui' compute backend
that declares the minimax_h3 model family (e.g. the Windows PC), separate from
the default ai-kvm2 ComfyUI. One source video (frames + its soundtrack) and up
to four reference images condition generation; the prompt describes how to
combine them — the canonical use is character swap ("Subject 1 from the video
is replaced by the person in Picture 1") while camera, framing and timing are
preserved. Because it targets a different server, this adapter uploads the
video and images itself (handles_own_image_upload=True) and the router skips
its default pre-upload.
"""

from __future__ import annotations

import asyncio
import base64
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


class MiniMaxH3LocalR2VAdapter(GenerationAdapter):
    """
    MiniMax-H3 REF2VA (~21B pruned int8) Reference-to-Video on the Windows PC ComfyUI.

    Separate checkpoint from FL2VA (minimax_h3_ref2va_pruned_int8_convrot):
    the source video conditions motion/camera/timing, reference images carry
    the identities to insert. Audio follows the reference video's soundtrack.
    24 fps, 17k+5 frame grid.
    """

    name = "minimax-h3-local-r2v"
    model_family = "minimax_h3"
    supported_ops = {Operation.GENERATE}
    input_types = {MediaType.VIDEO}
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
            max_input_images=4,
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

    @staticmethod
    def _to_video_bytes(video: str) -> bytes:
        """Decode a base64 (or data-URI) video string into raw bytes."""
        raw = video
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[-1]
        raw = "".join(raw.split())
        missing = (-len(raw)) % 4
        if missing:
            raw = raw + ("=" * missing)
        return base64.b64decode(raw)

    def build_workflow(
        self,
        req: GenerationRequest,
        video_name: str = "",
        ref_image_names: list[str] | None = None,
    ) -> dict:
        """Build the local MiniMax-H3 R2V ComfyUI workflow.

        ``video_name`` is the filename the source video was uploaded as on the
        target server; ``ref_image_names`` are the uploaded reference images.
        Placeholders keep workflow-only callers working.
        """
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")
        comfyui = self._get_comfyui()
        lora_configs = (
            [lr.model_dump(exclude_none=True) for lr in req.loras]
            if req.loras
            else None
        )
        return comfyui.build_local_minimax_h3_r2v_workflow(
            prompt=req.prompt,
            video_name=video_name or "r2v_source.mp4",
            ref_image_names=ref_image_names or [],
            num_frames=req.frames or 124,
            fps=req.fps or 24,
            seed=req.seed,
            steps=req.steps or 20,
            aspect_ratio=req.aspect_ratio or "16:9",
            megapixels=req.megapixels,
            lora_configs=lora_configs,
            quality_mode=req.quality_mode or "full",
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

        if not req.input_video:
            raise ValueError("MiniMax-H3 local R2V requires an input video")

        # Windows ComfyUI is lazy-started behind the caretaker wake proxy: the
        # first request may wait ~2 minutes for the cold boot, so give the
        # preflight a generous timeout (connection-refused still fails fast).
        # All blocking client calls run in a thread: they must not freeze the
        # backend's async event loop while the wake proxy cold-starts ComfyUI.
        if not await asyncio.to_thread(client.is_available, timeout=300):
            raise RuntimeError(
                f"ComfyUI backend at {client.host}:{client.port} is not reachable — "
                "is the configured compute backend running on that machine?"
            )

        # Upload the source video and the reference images to the resolved
        # server (router pre-upload skipped via handles_own_image_upload).
        #
        # The filenames carry a per-request id: ComfyUI's /upload/image is
        # called with overwrite=true, so fixed names let a second request
        # replace the media of a workflow that is still queued — that job then
        # renders somebody else's video. One id per request keeps them apart.
        upload_id = uuid.uuid4().hex[:12]
        try:
            video_bytes = self._to_video_bytes(req.input_video)
            uploaded_video = await asyncio.to_thread(
                client.upload_video_from_bytes,
                video_bytes,
                filename=f"v2_minimax_r2v_source_{upload_id}.mp4",
            )
        except Exception as e:
            logger.exception("❌ Failed to decode/upload MiniMax-H3 R2V source video")
            raise RuntimeError(f"Failed to upload MiniMax-H3 R2V source video: {e}")

        if not uploaded_video:
            raise RuntimeError("ComfyUI video upload failed on Windows server")

        ref_image_names: list[str] = []
        for idx, img_b64 in enumerate((req.input_images or [])[:4]):
            try:
                img_bytes = self._to_png_bytes(img_b64)
                uploaded = await asyncio.to_thread(
                    client.upload_image_from_bytes,
                    img_bytes,
                    filename=f"v2_minimax_r2v_ref_{upload_id}_{idx}.png",
                )
            except Exception as e:
                logger.exception(f"❌ Failed to upload R2V reference image {idx}")
                raise RuntimeError(f"Failed to upload R2V reference image {idx}: {e}")
            if not uploaded:
                raise RuntimeError(f"ComfyUI image upload failed for ref {idx}")
            ref_image_names.append(uploaded)

        workflow = self.build_workflow(
            req, video_name=uploaded_video, ref_image_names=ref_image_names
        )
        if not workflow:
            raise RuntimeError("Failed to build MiniMax-H3 local R2V workflow")

        prompt_id = await asyncio.to_thread(client.queue_prompt, workflow)
        if not prompt_id:
            raise RuntimeError("Failed to queue MiniMax-H3 local R2V to ComfyUI")

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=0,
            adapter_name=self.name,
            meta={
                "frames": req.frames or 124,
                "fps": req.fps or 24,
                "video": uploaded_video,
                "ref_images": ref_image_names,
                "server": f"{client.host}:{client.port}",
            },
        )
