"""
Video-to-Video adapter — style transfer via first-frame I2V.

Extracts first frame → runs I2V with style prompt → V2V output.
Uses ComfyUIClient.generate_distorch2_video() for actual processing.
"""

from __future__ import annotations

import logging
import tempfile
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

# Prompt enhancement templates per V2V mode
_MODE_PROMPTS = {
    "anime": "{base}, anime style, smooth animation, consistent character design, fluid motion",
    "enhance": "{base}, high quality, sharp details, natural movement, enhanced clarity",
    "style_transfer": "{base}, artistic style transfer, preserve motion, consistent style throughout",
}

# Canonical output directory
_OUTPUT_DIR = Path(__file__).resolve().parents[5] / "media" / "generated"


class V2VStyleTransferAdapter(GenerationAdapter):
    """
    Local Video-to-Video style transfer via ComfyUI.

    Extracts the first frame from input video, applies I2V generation
    with the style prompt, producing a style-transferred video.
    """

    name = "local-v2v"
    model_family = "wan2.2"
    supported_ops = {Operation.TRANSFORM}
    input_types = {MediaType.VIDEO}
    output_type = MediaType.VIDEO
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.DUAL_STAGE

    def __init__(self, comfyui_client_fn: Any = None) -> None:
        self._get_comfyui = comfyui_client_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=1280,
            max_height=720,
            min_width=480,
            min_height=480,
            max_frames=161,
            resolution_step=16,
            resolution_presets=["480p", "576p", "720p"],
            aspect_ratios=["9:16", "16:9", "1:1", "4:3", "3:4"],
            min_steps=4,
            max_steps=30,
            default_steps=6,
            default_cfg=1.0,
            supported_samplers=["uni_pc", "euler", "dpmpp_2m"],
            supported_schedulers=["normal", "simple", "karras"],
            max_loras=5,
            supports_negative_prompt=True,
            allowed_fps=[8, 12, 16, 24],
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        return {
            "_adapter": self.name,
            "_note": "V2V delegates to generate_distorch2_video()",
        }

    def cost(self, req: GenerationRequest) -> int:
        frames = req.frames or 41
        if frames <= 81:
            return 5
        elif frames <= 161:
            return 8
        return 15

    def _build_prompt(self, req: GenerationRequest) -> str:
        """Build the style-enhanced prompt based on V2V mode."""
        base = (req.prompt or "").strip()
        mode = req.v2v_mode or "style_transfer"
        preserve = req.preserve_motion if req.preserve_motion is not None else True

        if preserve:
            template = _MODE_PROMPTS.get(mode, _MODE_PROMPTS["style_transfer"])
            return template.format(base=base)
        return base

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        import cv2

        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")

        if not req.input_video:
            raise ValueError("V2V requires an input video")

        client = self._get_comfyui()

        # Resolve dimensions — router may have already set them
        width = req.width or 848
        height = req.height or 480

        # Extract first frame from video
        video_source = req.input_video
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_source}")

        try:
            ret, first_frame = cap.read()
        finally:
            cap.release()

        if not ret or first_frame is None:
            raise ValueError("Could not extract first frame from video")

        # Resize to target resolution
        if first_frame.shape[1] != width or first_frame.shape[0] != height:
            first_frame = cv2.resize(
                first_frame, (width, height), interpolation=cv2.INTER_LANCZOS4
            )

        # Save to temp file for ComfyUI (will be cleaned up by OS)
        with tempfile.NamedTemporaryFile(
            suffix=".png", prefix="v2v_frame_", delete=False
        ) as f:
            frame_path = f.name
            cv2.imwrite(frame_path, first_frame)

        logger.info(f"📸 V2V first frame extracted: {frame_path} ({width}x{height})")

        full_prompt = self._build_prompt(req)
        logger.info(f"🎨 V2V prompt: {full_prompt[:100]}...")

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_prefix = f"oelala_v2v_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Call generate_distorch2_video with the CORRECT signature
        output_path = client.generate_distorch2_video(
            image_path=frame_path,
            prompt=full_prompt,
            output_dir=str(_OUTPUT_DIR),
            output_prefix=output_prefix,
            resolution=req.resolution or "480p",
            aspect_ratio=req.aspect_ratio or "16:9",
            num_frames=req.frames or 41,
            fps=req.fps or 16,
            steps=req.steps or 6,
            cfg=req.cfg or 1.0,
            seed=req.seed if req.seed and req.seed >= 0 else -1,
        )

        # Clean up temp frame file
        try:
            Path(frame_path).unlink(missing_ok=True)
        except OSError:
            pass

        if not output_path:
            raise RuntimeError("V2V generation failed — no output")

        return GenerationResult(
            prompt_id=output_prefix,
            status="completed",
            compute_target=ComputeTarget.LOCAL,
            credits_used=0,  # Router fills this in
            adapter_name=self.name,
            meta={
                "v2v_mode": req.v2v_mode or "style_transfer",
                "result_path": output_path,
                "width": width,
                "height": height,
            },
        )
