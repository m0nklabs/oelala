"""
SDXL Local T2I adapter — text-to-image via local ComfyUI.

Builds the SDXL workflow inline (Power LoRA Loader ×3), queues to ComfyUI.
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

# SDXL-optimal resolutions at ~1MP
SDXL_RESOLUTIONS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
    "2:3": (832, 1216),
    "3:2": (1216, 832),
    "21:9": (1536, 640),
    "9:21": (640, 1536),
}


class SDXLLocalT2IAdapter(GenerationAdapter):
    """
    SDXL text-to-image on local ComfyUI.

    Uses Power LoRA Loader (rgthree) with up to 3 LoRA slots.
    Supports multiple checkpoints (CyberRealistic Pony, etc.).
    """

    name = "sdxl-local-t2i"
    model_family = "sdxl"
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
            aspect_ratios=list(SDXL_RESOLUTIONS.keys()),
            min_steps=10,
            max_steps=100,
            default_steps=30,
            default_cfg=7.5,
            supported_samplers=["dpmpp_2m", "euler", "euler_ancestral", "dpmpp_sde"],
            supported_schedulers=["karras", "normal", "simple"],
            max_loras=3,
            supports_negative_prompt=True,
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)
        checkpoint = req.checkpoint or "CyberRealistic_Pony_v14.1_FP16.safetensors"
        width = (
            req.width
            or SDXL_RESOLUTIONS.get(req.aspect_ratio or "1:1", (1024, 1024))[0]
        )
        height = (
            req.height
            or SDXL_RESOLUTIONS.get(req.aspect_ratio or "1:1", (1024, 1024))[1]
        )

        workflow = {
            "1": {
                "inputs": {"ckpt_name": checkpoint},
                "class_type": "CheckpointLoaderSimple",
            },
            "2": {
                "inputs": {"text": req.prompt, "clip": ["9", 1]},
                "class_type": "CLIPTextEncode",
            },
            "3": {
                "inputs": {"text": req.negative_prompt, "clip": ["9", 1]},
                "class_type": "CLIPTextEncode",
            },
            "4": {
                "inputs": {"width": width, "height": height, "batch_size": 1},
                "class_type": "EmptyLatentImage",
            },
            "5": {
                "inputs": {
                    "seed": seed,
                    "steps": req.steps or 30,
                    "cfg": req.cfg or 7.5,
                    "sampler_name": req.sampler or "dpmpp_2m",
                    "scheduler": req.scheduler or "karras",
                    "denoise": 1,
                    "model": ["9", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0],
                },
                "class_type": "KSampler",
            },
            "6": {
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
                "class_type": "VAEDecode",
            },
            "8": {
                "inputs": {"filename_prefix": "oelala_t2i", "images": ["6", 0]},
                "class_type": "SaveImage",
            },
            "9": {
                "inputs": {
                    "PowerLoraLoaderHeaderWidget": {
                        "type": "PowerLoraLoaderHeaderWidget"
                    },
                    "lora_1": {"on": False, "lora": "None", "strength": 1},
                    "lora_2": {"on": False, "lora": "None", "strength": 1},
                    "lora_3": {"on": False, "lora": "None", "strength": 1},
                    "\u2795 Add Lora": "",
                    "model": ["1", 0],
                    "clip": ["1", 1],
                },
                "class_type": "Power Lora Loader (rgthree)",
            },
        }

        # Apply LoRAs
        if req.loras:
            for i, lora in enumerate(req.loras[:3], 1):
                if lora.name and lora.name != "None":
                    workflow["9"]["inputs"][f"lora_{i}"] = {
                        "on": True,
                        "lora": lora.name,
                        "strength": lora.strength,
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
            raise RuntimeError("Failed to queue SDXL workflow to ComfyUI")

        credits_used = self.cost(req)

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=0,  # Router fills this in
            adapter_name=self.name,
            meta={
                "checkpoint": req.checkpoint
                or "CyberRealistic_Pony_v14.1_FP16.safetensors",
                "width": req.width or 1024,
                "height": req.height or 1024,
            },
        )
