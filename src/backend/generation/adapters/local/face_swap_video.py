"""
Face Swap Video adapter — frame-by-frame InsightFace (NOT ComfyUI).

Uses face_service.swap_faces_in_video() which processes each frame
via InsightFace and remuxes audio with ffmpeg.
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


class FaceSwapVideoAdapter(GenerationAdapter):
    """
    Local video face swap via InsightFace — frame-by-frame processing.

    Does NOT use ComfyUI — runs InsightFace + ffmpeg directly.
    """

    name = "local-faceswap-video"
    model_family = ""
    supported_ops = {Operation.SWAP}
    input_types = {MediaType.VIDEO}
    output_type = MediaType.VIDEO
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.NONE

    def __init__(self, face_service_fn: Any = None) -> None:
        self._get_face_service = face_service_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=3840,
            max_height=2160,
            min_width=64,
            min_height=64,
            max_input_images=1,  # source face image
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        return {"_adapter": self.name, "_engine": "insightface"}

    def cost(self, req: GenerationRequest) -> int:
        return 5

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        if self._get_face_service is None:
            raise RuntimeError("Face service not available")

        if not req.input_video:
            raise ValueError("Video face swap requires an input video")
        if not req.input_images:
            raise ValueError("Video face swap requires a source face image")

        face_service = self._get_face_service()

        result = await face_service.swap_faces_in_video(
            video_data=req.input_video,
            source_image=req.input_images[0],
            face_indices=req.face_indices,
        )

        if not result:
            raise RuntimeError("Video face swap failed")

        return GenerationResult(
            prompt_id=str(uuid.uuid4()),
            status="completed",
            compute_target=ComputeTarget.LOCAL,
            credits_used=0,  # Router fills this in
            adapter_name=self.name,
            meta={"face_indices": req.face_indices},
        )
