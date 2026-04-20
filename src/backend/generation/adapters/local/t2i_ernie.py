"""
ERNIE-Image Local T2I adapter — text-to-image via local ComfyUI.

Uses Flux2 sampling pipeline (SamplerCustomAdvanced + Flux2Scheduler).
NOTE: DisTorch2 multi-GPU produces NaN with ERNIE — plain loaders only.
ERNIE-Image: Flux2 latent format, Ministral-3-3B text encoder,
FlowMatch sampling. Optimal at 1024x1024.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

from ...adapter import GenerationAdapter, ProgressCallback
from ...types import (
    AdapterConstraints,
    ComputeTarget,
    GenerationRequest,
    GenerationResult,
    MediaType,
    Operation,
    LoraFormat,
)

logger = logging.getLogger(__name__)


class ErnieLocalT2IAdapter(GenerationAdapter):
    """
    ERNIE-Image text-to-image on local ComfyUI.
    Uses Flux2 sampling pipeline with ComfyUI's dynamic VRAM offloading.

    Uses Ministral-3-3B text encoder, Flux2 VAE, FlowMatch sampling.
    Best at 1024x1024, supports various aspect ratios.
    NOTE: DisTorch2 multi-GPU produces NaN — plain loaders only.
    """

    name = "ernie-local-t2i"
    model_family = "ernie"
    supported_ops = {Operation.GENERATE}
    input_types = {MediaType.TEXT}
    output_type = MediaType.IMAGE
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.NONE

    def __init__(self, comfyui_client_fn: Any = None) -> None:
        self._get_comfyui = comfyui_client_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=1536,
            max_height=1536,
            min_width=512,
            min_height=512,
            resolution_step=16,
            aspect_ratios=["1:1", "16:9", "9:16", "4:3", "3:4"],
            min_steps=10,
            max_steps=100,
            default_steps=20,
            default_cfg=4.0,
            supports_negative_prompt=True,
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        """Build ERNIE-Image T2I ComfyUI workflow (Flux2 sampling pipeline)."""
        seed = req.seed if req.seed and req.seed >= 0 else random.randint(0, 2**63 - 1)
        width = req.width or 1024
        height = req.height or 1024
        steps = req.steps or 20
        guidance = req.cfg or 4.0

        return {
            "10": {
                "inputs": {
                    "unet_name": "ernie-image.safetensors",
                    "weight_dtype": "default",
                },
                "class_type": "UNETLoader",
                "_meta": {"title": "Load ERNIE-Image UNET"},
            },
            "11": {
                "inputs": {
                    "clip_name": "ministral-3-3b.safetensors",
                    "type": "flux2",
                },
                "class_type": "CLIPLoader",
                "_meta": {"title": "Load Ministral-3-3B Text Encoder"},
            },
            "12": {
                "inputs": {"vae_name": "flux2-vae.safetensors"},
                "class_type": "VAELoader",
                "_meta": {"title": "Load Flux2 VAE"},
            },
            "20": {
                "inputs": {"text": req.prompt, "clip": ["11", 0]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt"},
            },
            "21": {
                "inputs": {"guidance": guidance, "conditioning": ["20", 0]},
                "class_type": "FluxGuidance",
                "_meta": {"title": "Flux Guidance"},
            },
            "22": {
                "inputs": {"model": ["10", 0], "conditioning": ["21", 0]},
                "class_type": "BasicGuider",
                "_meta": {"title": "Basic Guider"},
            },
            "23": {
                "inputs": {"steps": steps, "width": width, "height": height},
                "class_type": "Flux2Scheduler",
                "_meta": {"title": "Flux2 Scheduler"},
            },
            "24": {
                "inputs": {"sampler_name": "euler"},
                "class_type": "KSamplerSelect",
                "_meta": {"title": "Sampler Select (euler)"},
            },
            "25": {
                "inputs": {"noise_seed": seed},
                "class_type": "RandomNoise",
                "_meta": {"title": "Random Noise"},
            },
            "30": {
                "inputs": {"width": width, "height": height, "batch_size": 1},
                "class_type": "EmptyFlux2LatentImage",
                "_meta": {"title": "Empty Flux2 Latent (128ch)"},
            },
            "40": {
                "inputs": {
                    "noise": ["25", 0],
                    "guider": ["22", 0],
                    "sampler": ["24", 0],
                    "sigmas": ["23", 0],
                    "latent_image": ["30", 0],
                },
                "class_type": "SamplerCustomAdvanced",
                "_meta": {"title": "Sampler (FlowMatch)"},
            },
            "50": {
                "inputs": {"samples": ["40", 0], "vae": ["12", 0]},
                "class_type": "VAEDecode",
                "_meta": {"title": "VAE Decode"},
            },
            "51": {
                "inputs": {"filename_prefix": "Ernie_T2I", "images": ["50", 0]},
                "class_type": "SaveImage",
                "_meta": {"title": "Save Image"},
            },
        }

    def cost(self, req: GenerationRequest) -> int:
        return 3  # ERNIE is heavier than standard T2I

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")
        client = self._get_comfyui()

        # Build workflow and embed extras if allowed
        workflow = self.build_workflow(req)
        
        # Merge extra frontend configuration dynamically 
        # (This is identical to what we added in wan22 logic)
        for _, node in workflow.items():
            if node.get("class_type") == "UNETLoader" and getattr(req, "model_extra", None):
                for k, v in req.model_extra.items():
                    if k not in node["inputs"]:
                        node["inputs"][k] = v

        prompt_id = client.queue_prompt(workflow)

        if not prompt_id:
            raise RuntimeError("Failed to queue ERNIE-Image workflow to ComfyUI")

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=0,
            adapter_name=self.name,
            meta={
                "width": req.width or 1024,
                "height": req.height or 1024,
            },
        )
