"""
SD 1.5 Local T2I adapter — text-to-image via local ComfyUI.

Uses Power LoRA Loader (rgthree) with up to 6 LoRA slots.
Optimal at 512×512 to 768×768.
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

SD15_RESOLUTIONS = {
    "1:1": (512, 512),
    "16:9": (768, 432),
    "9:16": (432, 768),
    "4:3": (576, 432),
    "3:4": (432, 576),
    "2:3": (512, 768),
    "3:2": (768, 512),
}


class SD15LocalT2IAdapter(GenerationAdapter):
    """
    SD 1.5 text-to-image on local ComfyUI.

    Uses Power LoRA Loader (rgthree) with up to 6 LoRA slots.
    Lightweight model, fast generation.
    """

    name = "sd15-local-t2i"
    model_family = "sd1.5"
    supported_ops = {Operation.GENERATE}
    input_types = {MediaType.TEXT}
    output_type = MediaType.IMAGE
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.SINGLE_STAGE

    def __init__(self, comfyui_client_fn: Any = None) -> None:
        self._get_comfyui = comfyui_client_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=1024,
            max_height=1024,
            min_width=256,
            min_height=256,
            resolution_step=64,
            aspect_ratios=list(SD15_RESOLUTIONS.keys()),
            min_steps=10,
            max_steps=100,
            default_steps=25,
            default_cfg=7.0,
            supported_samplers=["dpmpp_sde", "dpmpp_2m", "euler", "euler_ancestral"],
            supported_schedulers=["karras", "normal"],
            max_loras=6,
            supports_negative_prompt=True,
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)
        checkpoint = req.checkpoint or "Realistic_Vision_V5.1.safetensors"
        width = (
            req.width or SD15_RESOLUTIONS.get(req.aspect_ratio or "1:1", (512, 512))[0]
        )
        height = (
            req.height or SD15_RESOLUTIONS.get(req.aspect_ratio or "1:1", (512, 512))[1]
        )

        workflow = {
            "1": {
                "inputs": {"ckpt_name": checkpoint},
                "class_type": "CheckpointLoaderSimple",
            },
            "2": {
                "inputs": {
                    "PowerLoraLoaderHeaderWidget": {
                        "type": "PowerLoraLoaderHeaderWidget"
                    },
                    "lora_1": {"on": False, "lora": "None", "strength": 1},
                    "lora_2": {"on": False, "lora": "None", "strength": 1},
                    "lora_3": {"on": False, "lora": "None", "strength": 1},
                    "lora_4": {"on": False, "lora": "None", "strength": 1},
                    "lora_5": {"on": False, "lora": "None", "strength": 1},
                    "lora_6": {"on": False, "lora": "None", "strength": 1},
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
                "inputs": {"text": req.negative_prompt, "clip": ["2", 1]},
                "class_type": "CLIPTextEncode",
            },
            "5": {
                "inputs": {"width": width, "height": height, "batch_size": 1},
                "class_type": "EmptyLatentImage",
            },
            "6": {
                "inputs": {
                    "seed": seed,
                    "steps": req.steps or 25,
                    "cfg": req.cfg or 7.0,
                    "sampler_name": req.sampler or "dpmpp_sde",
                    "scheduler": req.scheduler or "karras",
                    "denoise": 1,
                    "model": ["2", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0],
                },
                "class_type": "KSampler",
            },
            "7": {
                "inputs": {"samples": ["6", 0], "vae": ["1", 2]},
                "class_type": "VAEDecode",
            },
            "8": {
                "inputs": {"filename_prefix": "oelala_sd15", "images": ["7", 0]},
                "class_type": "SaveImage",
            },
        }

        # Apply LoRAs
        if req.loras:
            for i, lora in enumerate(req.loras[:6], 1):
                if lora.name and lora.name != "None":
                    workflow["2"]["inputs"][f"lora_{i}"] = {
                        "on": True,
                        "lora": lora.name,
                        "strength": lora.strength,
                    }

        return workflow

    def cost(self, req: GenerationRequest) -> int:
        return 1  # SD 1.5 is always 1 credit

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
            raise RuntimeError("Failed to queue SD 1.5 workflow to ComfyUI")

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=self.cost(req),
            adapter_name=self.name,
            meta={"checkpoint": req.checkpoint or "Realistic_Vision_V5.1.safetensors"},
        )
