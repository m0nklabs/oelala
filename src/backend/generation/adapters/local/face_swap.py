"""
Face Swap Image adapter — InsightFace inswapper_128 (NOT ComfyUI).

Uses face_service.swap_faces_to_bytes() which runs insightface
directly in a thread pool, not via ComfyUI workflows.
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


class FaceSwapImageAdapter(GenerationAdapter):
    """
    Local face swap via InsightFace inswapper_128.

    Does NOT use ComfyUI — runs directly in thread pool.
    Supports multiple face indices ("0", "0,1", "-1" for all faces).
    """

    name = "local-faceswap"
    model_family = ""
    supported_ops = {Operation.SWAP}
    input_types = {MediaType.IMAGE}
    output_type = MediaType.IMAGE
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.NONE

    def __init__(self, face_service_fn: Any = None) -> None:
        self._get_face_service = face_service_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=4096,
            max_height=4096,
            min_width=64,
            min_height=64,
            max_input_images=2,  # target + source
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        """No ComfyUI workflow — uses InsightFace directly."""
        return {"_adapter": self.name, "_engine": "insightface"}

    def cost(self, req: GenerationRequest) -> int:
        return 1

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        if self._get_face_service is None:
            raise RuntimeError("Face service not available")

        if len(req.input_images) < 2:
            raise ValueError("Face swap requires 2 input images (target + source)")

        face_service = self._get_face_service()

        result = await face_service.swap_faces_to_bytes(
            target_image=req.input_images[0],
            source_image=req.input_images[1],
            face_indices=req.face_indices,
        )

        if not result:
            raise RuntimeError("Face swap failed")

        return GenerationResult(
            prompt_id=str(uuid.uuid4()),
            status="completed",
            compute_target=ComputeTarget.LOCAL,
            credits_used=0,  # Router fills this in
            adapter_name=self.name,
            meta={"face_indices": req.face_indices},
        )
