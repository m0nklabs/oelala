"""
Wan2.2 Local T2I adapter — text-to-image via local ComfyUI.

Uses DisTorch2 multi-GPU setup with high/low noise models.
Optimal at 512×512 to 768×768. Very fast with 8 steps.
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


class Wan22LocalT2IAdapter(GenerationAdapter):
    """
    Wan2.2 text-to-image on local ComfyUI with DisTorch2 multi-GPU.

    Uses dual-pass high/low noise sampling with GGUF quantized models.
    Very fast (8 steps) for quick image generation.
    """

    name = "wan22-local-t2i"
    model_family = "wan2.2"
    supported_ops = {Operation.GENERATE}
    input_types = {MediaType.TEXT}
    output_type = MediaType.IMAGE
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.DUAL_STAGE

    def __init__(self, comfyui_client_fn: Any = None) -> None:
        self._get_comfyui = comfyui_client_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=1024,
            max_height=1024,
            min_width=256,
            min_height=256,
            resolution_step=16,
            aspect_ratios=["1:1", "16:9", "9:16", "4:3", "3:4"],
            min_steps=4,
            max_steps=30,
            default_steps=8,
            default_cfg=7.0,
            max_loras=5,
            supports_negative_prompt=True,
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        """Build Wan2.2 T2I workflow for queued local execution."""
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")

        width = req.width or 512
        height = req.height or 512
        steps = req.steps or 8
        seed = req.seed if req.seed is not None and req.seed >= 0 else random.randint(0, 2**63 - 1)
        seed2 = random.randint(0, 2**63 - 1)
        half_steps = steps // 2

        return {
            "3": {
                "inputs": {"text": req.prompt, "clip": ["29", 1]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt"},
            },
            "4": {
                "inputs": {"text": req.negative_prompt or "", "clip": ["29", 1]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Negative Prompt"},
            },
            "5": {
                "inputs": {
                    "width": width,
                    "height": height,
                    "length": 1,
                    "batch_size": 1,
                },
                "class_type": "EmptyHunyuanLatentVideo",
                "_meta": {"title": "Empty HunyuanVideo 1.0 Latent"},
            },
            "9": {
                "inputs": {"samples": ["36", 0], "vae": ["55", 0]},
                "class_type": "VAEDecode",
                "_meta": {"title": "VAE Decode"},
            },
            "10": {
                "inputs": {"filename_prefix": "Wan22_T2I", "images": ["9", 0]},
                "class_type": "SaveImage",
                "_meta": {"title": "Save Image"},
            },
            "29": {
                "inputs": {
                    "lora_name": "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors",
                    "strength_model": 0,
                    "strength_clip": 0,
                    "model": ["50", 0],
                    "clip": ["51", 0],
                },
                "class_type": "LoraLoader",
                "_meta": {"title": "Load LoRA"},
            },
            "35": {
                "inputs": {
                    "add_noise": "enable",
                    "noise_seed": seed,
                    "steps": steps,
                    "cfg": 1,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "start_at_step": 0,
                    "end_at_step": half_steps,
                    "return_with_leftover_noise": "disable",
                    "model": ["29", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0],
                },
                "class_type": "KSamplerAdvanced",
                "_meta": {"title": "KSampler (Advanced)"},
            },
            "36": {
                "inputs": {
                    "add_noise": "enable",
                    "noise_seed": seed2,
                    "steps": steps,
                    "cfg": 1,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "start_at_step": half_steps,
                    "end_at_step": steps,
                    "return_with_leftover_noise": "disable",
                    "model": ["44", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["35", 0],
                },
                "class_type": "KSamplerAdvanced",
                "_meta": {"title": "KSampler (Advanced)"},
            },
            "44": {
                "inputs": {
                    "lora_name": "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors",
                    "strength_model": 0,
                    "model": ["52", 0],
                },
                "class_type": "LoraLoaderModelOnly",
                "_meta": {"title": "LoraLoaderModelOnly"},
            },
            "50": {
                "inputs": {
                    "unet_name": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
                    "weight_dtype": "default",
                    "compute_device": "cuda:1",
                    "virtual_vram_gb": 5,
                    "donor_device": "cuda:0",
                    "expert_mode_allocations": "",
                    "eject_models": True,
                },
                "class_type": "UNETLoaderDisTorch2MultiGPU",
                "_meta": {"title": "UNETLoaderDisTorch2MultiGPU"},
            },
            "51": {
                "inputs": {
                    "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                    "type": "wan",
                    "device": "cuda:1",
                },
                "class_type": "CLIPLoaderMultiGPU",
                "_meta": {"title": "CLIPLoaderMultiGPU"},
            },
            "52": {
                "inputs": {
                    "unet_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
                    "weight_dtype": "default",
                    "compute_device": "cuda:1",
                    "virtual_vram_gb": 5,
                    "donor_device": "cuda:0",
                    "expert_mode_allocations": "",
                    "eject_models": True,
                },
                "class_type": "UNETLoaderDisTorch2MultiGPU",
                "_meta": {"title": "UNETLoaderDisTorch2MultiGPU"},
            },
            "55": {
                "inputs": {
                    "vae_name": "wan_2.1_vae.safetensors",
                    "compute_device": "cuda:1",
                    "virtual_vram_gb": 0,
                    "donor_device": "cuda:0",
                    "expert_mode_allocations": "",
                    "eject_models": True,
                },
                "class_type": "VAELoaderDisTorch2MultiGPU",
                "_meta": {"title": "VAELoaderDisTorch2MultiGPU"},
            },
        }

    def cost(self, req: GenerationRequest) -> int:
        return 2  # Wan22 T2I is 2 credits

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
            raise RuntimeError("Failed to queue Wan2.2 T2I workflow to ComfyUI")

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=0,  # Router fills this in
            adapter_name=self.name,
            meta={
                "width": req.width or 512,
                "height": req.height or 512,
                "steps": req.steps or 8,
            },
        )
