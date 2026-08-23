"""
Krea 2 Turbo Local T2I adapter — text-to-image via local ComfyUI.

Krea 2 is an image generation model from Krea AI (trained from scratch),
focused on creative/stylistic exploration. Turbo is the 8-step distilled
checkpoint for fast, high-quality generation.

Pipeline (flat workflow, verified pattern):
  UNETLoader(krea2_turbo_int8_convrot) + CLIPLoader(type=krea2,
  qwen3vl_4b) + VAELoader(qwen_image_vae) -> KSampler (8 steps, CFG 1.0,
  euler/simple) -> VAEDecode -> SaveImage.

Notes:
- Uses Qwen3-VL-4B as text encoder (NOT CLIP/T5). CLIPLoader type must be "krea2".
- Requires the qwen_image_vae VAE (shared with Qwen Image).
- Distilled model: CFG must stay ~1.0 (higher degrades output).
- ComfyUI >= v0.27 for the INT8 ConvRot fix (v0.33.x recommended).
"""

from __future__ import annotations

import logging
import random
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

# Krea 2 native resolutions (1K–2K, 16:9/9:16 supported)
KREA2_RESOLUTIONS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
    "2:3": (832, 1216),
    "3:2": (1216, 832),
}


class Krea2LocalT2IAdapter(GenerationAdapter):
    """
    Krea 2 Turbo text-to-image on local ComfyUI (INT8 ConvRot variant).
    """

    name = "krea2-local-t2i"
    model_family = "krea2"
    supported_ops = {Operation.GENERATE}
    input_types = {MediaType.TEXT}
    output_type = MediaType.IMAGE
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.NONE

    def __init__(self, comfyui_client_fn: Any = None) -> None:
        self._get_comfyui = comfyui_client_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=2048,
            max_height=2048,
            min_width=512,
            min_height=512,
            resolution_step=64,
            aspect_ratios=list(KREA2_RESOLUTIONS.keys()),
            min_steps=4,
            max_steps=20,
            default_steps=8,
            default_cfg=1.0,
            supported_samplers=["euler"],
            supported_schedulers=["simple"],
            max_loras=1,
            supports_negative_prompt=False,
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        seed = req.seed if req.seed >= 0 else random.randint(0, 2**63 - 1)
        width = (
            req.width
            or KREA2_RESOLUTIONS.get(req.aspect_ratio or "1:1", (1024, 1024))[0]
        )
        height = (
            req.height
            or KREA2_RESOLUTIONS.get(req.aspect_ratio or "1:1", (1024, 1024))[1]
        )
        steps = req.steps or 8
        cfg = req.cfg or 1.0

        workflow = {
            "1": {
                "inputs": {
                    "unet_name": "krea2_turbo_int8_convrot.safetensors",
                    "weight_dtype": "default",
                },
                "class_type": "UNETLoader",
            },
            "2": {
                "inputs": {
                    "clip_name": "qwen3vl_4b_bf16.safetensors",
                    "type": "krea2",
                },
                "class_type": "CLIPLoader",
            },
            "3": {
                "inputs": {"text": req.prompt, "clip": ["2", 0]},
                "class_type": "CLIPTextEncode",
            },
            "4": {
                "inputs": {"width": width, "height": height, "batch_size": 1},
                "class_type": "EmptyLatentImage",
            },
            "5": {
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["3", 0],
                    "negative": ["3", 0],  # Krea 2: distilled, no negative prompt
                    "latent_image": ["4", 0],
                },
                "class_type": "KSampler",
            },
            "6": {
                "inputs": {"vae_name": "qwen_image_vae.safetensors"},
                "class_type": "VAELoader",
            },
            "7": {
                "inputs": {"samples": ["5", 0], "vae": ["6", 0]},
                "class_type": "VAEDecode",
            },
            "8": {
                "inputs": {"filename_prefix": "oelala_krea2", "images": ["7", 0]},
                "class_type": "SaveImage",
            },
        }

        return workflow

    def cost(self, req: GenerationRequest) -> int:
        width = req.width or 1024
        height = req.height or 1024
        if width * height > 1024 * 1024:
            return 2  # HD
        return 1

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
            raise RuntimeError("Failed to queue Krea 2 workflow to ComfyUI")

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=0,  # Router fills this in
            adapter_name=self.name,
            meta={
                "width": width if (width := req.width) else 1024,
                "height": req.height or 1024,
                "steps": req.steps or 8,
            },
        )
