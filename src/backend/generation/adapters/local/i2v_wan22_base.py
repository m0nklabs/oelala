"""
Wan2.2 Local I2V base adapter — shared logic for all Wan22 I2V variants.

All local Wan22 I2V adapters share:
- Same input/output types (image→video)
- Same LoRA format (dual-stage)
- Same model family (wan2.2)
- Similar constraints

Each variant overrides _get_quant_config() to return its specific
model names, default settings, and delegates to its builder method.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
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


class QuantConfig:
    """Configuration for a specific Wan22 quantization variant."""

    def __init__(
        self,
        name: str,
        builder_method: str,
        default_steps: int = 6,
        default_cfg: float = 1.0,
        default_sampler: str = "uni_pc",
        default_scheduler: str = "normal",
        max_frames: int = 321,
        resolution_presets: list[str] | None = None,
        supports_extra_params: bool = False,
    ) -> None:
        self.name = name
        self.builder_method = builder_method
        self.default_steps = default_steps
        self.default_cfg = default_cfg
        self.default_sampler = default_sampler
        self.default_scheduler = default_scheduler
        self.max_frames = max_frames
        self.resolution_presets = resolution_presets or ["480p", "576p", "720p"]
        self.supports_extra_params = supports_extra_params


class Wan22LocalI2VBase(GenerationAdapter):
    """
    Base class for all local Wan22 I2V adapters.

    Subclasses only need to set class attributes and override _get_quant_config().
    """

    model_family = "wan2.2"
    supported_ops = {Operation.GENERATE}
    input_types = {MediaType.IMAGE}
    output_type = MediaType.VIDEO
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.DUAL_STAGE

    def __init__(self, comfyui_client_fn: Any = None) -> None:
        self._get_comfyui = comfyui_client_fn

    @abstractmethod
    def _get_quant_config(self) -> QuantConfig:
        """Return quantization-specific config for this variant."""
        ...

    def constraints(self) -> AdapterConstraints:
        qc = self._get_quant_config()
        return AdapterConstraints(
            max_width=1920,
            max_height=1080,
            min_width=480,
            min_height=480,
            max_frames=qc.max_frames,
            resolution_step=16,
            resolution_presets=qc.resolution_presets,
            aspect_ratios=["9:16", "16:9", "1:1", "4:3", "3:4", "3:2", "2:3"],
            min_steps=4,
            max_steps=30,
            default_steps=qc.default_steps,
            default_cfg=qc.default_cfg,
            supported_samplers=["uni_pc", "euler", "dpmpp_2m"],
            supported_schedulers=["normal", "simple", "karras"],
            max_loras=5,
            supports_negative_prompt=True,
            max_input_images=1,
            allowed_fps=[8, 12, 16, 24],
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")
        comfyui = self._get_comfyui()

        qc = self._get_quant_config()
        builder = getattr(comfyui, qc.builder_method)

        lora_dicts = (
            [lr.model_dump(exclude_none=True) for lr in req.loras]
            if req.loras
            else None
        )

        kwargs = {
            "image_name": req.input_images[0] if req.input_images else "input.png",
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "num_frames": req.frames or 81,
            "fps": req.fps or 16,
            "steps": req.steps or qc.default_steps,
            "cfg": req.cfg or qc.default_cfg,
            "seed": req.seed,
            "high_noise_steps": req.high_noise_steps or (qc.default_steps // 2),
            "aspect_ratio": req.aspect_ratio or "9:16",
            "lora_configs": lora_dicts,
        }

        if qc.supports_extra_params and req.model_extra:
            kwargs.update(req.model_extra)

        # Some builders have extra params like sampler/scheduler
        if qc.default_sampler:
            kwargs["sampler_name"] = req.sampler or qc.default_sampler
        if qc.default_scheduler:
            kwargs["scheduler"] = req.scheduler or qc.default_scheduler

        return builder(**kwargs)

    def cost(self, req: GenerationRequest) -> int:
        """Local I2V cost based on frame count."""
        frames = req.frames or 81
        if frames <= 81:
            return 5
        elif frames <= 161:
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

        if not req.input_images:
            raise ValueError(f"{self.name} requires an input image")

        workflow = self.build_workflow(req)
        if not workflow:
            raise RuntimeError(f"Failed to build {self.name} workflow")

        client = self._get_comfyui()
        prompt_id = client.queue_prompt(workflow)

        if not prompt_id:
            raise RuntimeError(f"Failed to queue {self.name} workflow to ComfyUI")

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=0,  # Router fills this in
            adapter_name=self.name,
            meta={
                "frames": req.frames or 81,
                "fps": req.fps or 16,
            },
        )
