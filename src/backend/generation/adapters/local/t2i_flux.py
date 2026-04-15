"""
Flux Dev Local T2I adapter — text-to-image via local ComfyUI.

Flux doesn't use negative prompts or CFG in the traditional sense.
Uses guidance scale instead.
"""

from __future__ import annotations

import logging
import random
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

FLUX_RESOLUTIONS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
    "2:3": (832, 1216),
    "3:2": (1216, 832),
}


class FluxLocalT2IAdapter(GenerationAdapter):
    """
    Flux Dev text-to-image on local ComfyUI.

    Uses Power LoRA Loader (rgthree) with up to 4 LoRA slots.
    No negative prompt. Guidance scale replaces CFG.
    """

    name = "flux-local-t2i"
    model_family = "flux"
    supported_ops = {Operation.GENERATE}
    input_types = {MediaType.TEXT}
    output_type = MediaType.IMAGE
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.SINGLE_STAGE

    def __init__(self, comfyui_client_fn: Any = None) -> None:
        self._get_comfyui = comfyui_client_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=2048,
            max_height=2048,
            min_width=512,
            min_height=512,
            resolution_step=64,
            aspect_ratios=list(FLUX_RESOLUTIONS.keys()),
            min_steps=10,
            max_steps=50,
            default_steps=20,
            default_cfg=3.5,
            max_loras=4,
            supports_negative_prompt=False,
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        seed = req.seed if req.seed >= 0 else random.randint(0, 2**63 - 1)
        checkpoint = req.checkpoint or "flux1-dev-fp8.safetensors"
        width = req.width or FLUX_RESOLUTIONS.get(req.aspect_ratio or "1:1", (1024, 1024))[0]
        height = req.height or FLUX_RESOLUTIONS.get(req.aspect_ratio or "1:1", (1024, 1024))[1]
        guidance = req.cfg or 3.5

        workflow = {
            "1": {
                "inputs": {"ckpt_name": checkpoint},
                "class_type": "CheckpointLoaderSimple",
            },
            "2": {
                "inputs": {
                    "PowerLoraLoaderHeaderWidget": {"type": "PowerLoraLoaderHeaderWidget"},
                    "lora_1": {"on": False, "lora": "None", "strength": 1},
                    "lora_2": {"on": False, "lora": "None", "strength": 1},
                    "lora_3": {"on": False, "lora": "None", "strength": 1},
                    "lora_4": {"on": False, "lora": "None", "strength": 1},
                    "\u2795 Add Lora": "",
                    "model": ["1", 0],
                    "clip": ["1", 1],
                },
                "class_type": "Power Lora Loader (rgthree)",
            },
            "3": {
                "inputs": {"text": req.prompt, "clip": ["2", 1]},
                "class_type": "CLIPTextEncode",
            },
            "4": {
                "inputs": {"width": width, "height": height, "batch_size": 1},
                "class_type": "EmptyLatentImage",
            },
            "5": {
                "inputs": {
                    "noise": ["8", 0],
                    "guider": ["9", 0],
                    "sampler": ["10", 0],
                    "sigmas": ["11", 0],
                    "latent_image": ["4", 0],
                },
                "class_type": "SamplerCustomAdvanced",
            },
            "6": {
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
                "class_type": "VAEDecode",
            },
            "7": {
                "inputs": {"filename_prefix": "oelala_flux", "images": ["6", 0]},
                "class_type": "SaveImage",
            },
            "8": {
                "inputs": {"noise_seed": seed},
                "class_type": "RandomNoise",
            },
            "9": {
                "inputs": {"model": ["2", 0], "conditioning": ["3", 0]},
                "class_type": "BasicGuider",
            },
            "10": {
                "inputs": {"sampler_name": "euler"},
                "class_type": "KSamplerSelect",
            },
            "11": {
                "inputs": {"model": ["2", 0], "steps": req.steps or 20},
                "class_type": "BasicScheduler",
            },
            "12": {
                "inputs": {"guidance": guidance, "conditioning": ["3", 0]},
                "class_type": "FluxGuidance",
            },
        }

        # Apply LoRAs
        if req.loras:
            for i, lora in enumerate(req.loras[:4], 1):
                if lora.name and lora.name != "None":
                    workflow["2"]["inputs"][f"lora_{i}"] = {
                        "on": True,
                        "lora": lora.name,
                        "strength": lora.strength,
                    }

        return workflow

    def cost(self, req: GenerationRequest) -> int:
        width = req.width or 1024
        height = req.height or 1024
        if width * height > 1024 * 1024:
            return 3  # HD
        return 2

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
            raise RuntimeError("Failed to queue Flux workflow to ComfyUI")

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=self.cost(req),
            adapter_name=self.name,
            meta={"checkpoint": req.checkpoint or "flux1-dev-fp8.safetensors"},
        )
