"""
Inpaint adapter — masked region regeneration via ComfyUI.

Uses KSampler + mask-based latent noise to regenerate selected areas.
"""

from __future__ import annotations

import logging
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


class InpaintAdapter(GenerationAdapter):
    """
    Local inpainting via ComfyUI.

    Takes image + mask (white=regenerate, black=keep), applies
    checkpoint-based inpainting with GrowMask feathering.
    """

    name = "local-inpaint"
    model_family = "sdxl"
    supported_ops = {Operation.INPAINT}
    input_types = {MediaType.IMAGE}
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
            min_steps=10,
            max_steps=100,
            default_steps=20,
            default_cfg=7.0,
            supported_samplers=["dpmpp_2m", "euler", "dpmpp_sde"],
            supported_schedulers=["karras", "normal"],
            supports_negative_prompt=True,
            max_input_images=1,
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        """Build inpainting ComfyUI workflow."""
        checkpoint = req.checkpoint or "dreamshaperXL_lightningDPMSDE.safetensors"
        image_name = req.input_images[0] if req.input_images else "input.png"
        mask_name = req.input_mask or "mask.png"
        feathering = req.feathering

        workflow = {
            "1": {
                "inputs": {"ckpt_name": checkpoint},
                "class_type": "CheckpointLoaderSimple",
            },
            "2": {
                "inputs": {"image": image_name},
                "class_type": "LoadImage",
            },
            "3": {
                "inputs": {"image": mask_name},
                "class_type": "LoadImage",
            },
            "4": {
                "inputs": {"channel": "red", "image": ["3", 0]},
                "class_type": "ImageToMask",
            },
            "5": {
                "inputs": {
                    "expand": feathering,
                    "tapered_corners": True,
                    "mask": ["4", 0],
                },
                "class_type": "GrowMask",
            },
            "6": {
                "inputs": {"pixels": ["2", 0], "vae": ["1", 2]},
                "class_type": "VAEEncode",
            },
            "7": {
                "inputs": {"text": req.prompt, "clip": ["1", 1]},
                "class_type": "CLIPTextEncode",
            },
            "8": {
                "inputs": {"text": req.negative_prompt, "clip": ["1", 1]},
                "class_type": "CLIPTextEncode",
            },
            "9": {
                "inputs": {"samples": ["6", 0], "mask": ["5", 0]},
                "class_type": "SetLatentNoiseMask",
            },
            "10": {
                "inputs": {
                    "seed": req.seed,
                    "steps": req.steps or 20,
                    "cfg": req.cfg or 7.0,
                    "sampler_name": req.sampler or "dpmpp_2m",
                    "scheduler": req.scheduler or "karras",
                    "denoise": req.denoise or 0.85,
                    "model": ["1", 0],
                    "positive": ["7", 0],
                    "negative": ["8", 0],
                    "latent_image": ["9", 0],
                },
                "class_type": "KSampler",
            },
            "11": {
                "inputs": {"samples": ["10", 0], "vae": ["1", 2]},
                "class_type": "VAEDecode",
            },
            "12": {
                "inputs": {"filename_prefix": "oelala_inpaint", "images": ["11", 0]},
                "class_type": "SaveImage",
            },
        }

        return workflow

    def cost(self, req: GenerationRequest) -> int:
        return 2

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")

        if not req.input_images:
            raise ValueError("Inpainting requires an input image")
        if not req.input_mask:
            raise ValueError("Inpainting requires a mask image")

        client = self._get_comfyui()
        workflow = self.build_workflow(req)
        prompt_id = client.queue_prompt(workflow)

        if not prompt_id:
            raise RuntimeError("Failed to queue inpaint workflow")

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=self.cost(req),
            adapter_name=self.name,
            meta={
                "checkpoint": req.checkpoint
                or "dreamshaperXL_lightningDPMSDE.safetensors"
            },
        )
