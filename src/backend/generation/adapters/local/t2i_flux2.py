"""
Flux 2 Dev Local T2I adapter — text-to-image via local ComfyUI (multi-GPU).

Flux 2 Dev is a 32B rectified-flow transformer (newest Flux family). Locally
we use the unsloth Dynamic 2.0 GGUF quantized UNet (Q4_K_M) distributed across
both GPUs via the ComfyUI-MultiGPU DisTorch2 loader, plus the Mistral3 small
text encoder (FP8) and the Flux 2 (Mage) one-step VAE.

Pipeline (flat workflow):
  UnetLoaderGGUFDisTorch2MultiGPU(flux2 Q4_K_M, multi-GPU) +
  CLIPLoader(type=flux2, mistral_3_small_flux2_fp8, device=cpu) +
  VAELoader(flux2-vae) -> FluxGuidance + Flux2Scheduler + EmptyFlux2LatentImage
  -> SamplerCustomAdvanced (euler/simple) -> VAEDecode -> SaveImage.

Notes:
- No negative prompt (Flux-family uses guidance, not CFG).
- Uses Flux2Scheduler + EmptyFlux2LatentImage (FLUX.2-specific nodes).
- Multi-GPU: UNet split over cuda:0 + cuda:1 with CPU offload via DisTorch2.
- Text encoder (Mistral3-small fp8, 18GB) is 2x the size of Flux1's T5 —
  loaded on CPU (plenty of RAM) to leave both GPUs for the 20GB UNet.
- ComfyUI >= 0.31 + ComfyUI-GGUF + ComfyUI-MultiGPU required.
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

FLUX2_RESOLUTIONS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
    "2:3": (832, 1216),
    "3:2": (1216, 832),
}

# DisTorch2 allocation: spread part of the 19GB UNet across both GPUs and push
# the bulk to CPU. Verified working on this box (RTX 3060 12GB + RTX 5060 Ti
# 16GB = 28GB VRAM): loading the compute card too full (>75%) OOMs.
#   cuda:1 (16GB): 8GB of UNet + activations (~89% util, ~12GB used)
#   cuda:0 (12GB): 4GB of UNet
#   cpu: rest of the UNet (DisTorch2 streams layers on demand)
FLUX2_DISTORCH_ALLOC = "cuda:1,8gb;cuda:0,4gb;cpu,*"
FLUX2_COMPUTE_DEVICE = "cuda:1"
FLUX2_VIRTUAL_VRAM_GB = 8.0


class Flux2LocalT2IAdapter(GenerationAdapter):
    """
    Flux 2 Dev text-to-image on local ComfyUI, multi-GPU (DisTorch2).

    Uses the GGUF-quantized UNet distributed over both GPUs and the
    Mistral3-small text encoder. No negative prompt; guidance replaces CFG.
    """

    name = "flux2-local-t2i"
    model_family = "flux2"
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
            resolution_step=16,
            aspect_ratios=list(FLUX2_RESOLUTIONS.keys()),
            min_steps=10,
            max_steps=40,
            default_steps=20,
            default_cfg=4.0,
            supported_samplers=["euler"],
            supported_schedulers=["simple"],
            max_loras=0,
            supports_negative_prompt=False,
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        seed = req.seed if req.seed >= 0 else random.randint(0, 2**63 - 1)
        width = (
            req.width
            or FLUX2_RESOLUTIONS.get(req.aspect_ratio or "1:1", (1024, 1024))[0]
        )
        height = (
            req.height
            or FLUX2_RESOLUTIONS.get(req.aspect_ratio or "1:1", (1024, 1024))[1]
        )
        steps = req.steps or 20
        guidance = req.cfg or 4.0

        workflow = {
            "1": {
                "inputs": {
                    "unet_name": "flux2-dev-Q4_K_M.gguf",
                },
                "class_type": "UnetLoaderGGUFDisTorch2MultiGPU",
            },
            "2": {
                "inputs": {
                    "clip_name": "mistral_3_small_flux2_fp8.safetensors",
                    "type": "flux2",
                    "device": "cpu",
                },
                "class_type": "CLIPLoader",
            },
            "3": {
                "inputs": {"text": req.prompt, "clip": ["2", 0]},
                "class_type": "CLIPTextEncode",
            },
            "4": {
                "inputs": {"width": width, "height": height, "batch_size": 1},
                "class_type": "EmptyFlux2LatentImage",
            },
            "5": {
                "inputs": {"vae_name": "flux2-vae.safetensors"},
                "class_type": "VAELoader",
            },
            "6": {
                "inputs": {"steps": steps, "width": width, "height": height},
                "class_type": "Flux2Scheduler",
            },
            "7": {
                "inputs": {"noise_seed": seed, "seed_control": "randomize"},
                "class_type": "RandomNoise",
            },
            "8": {
                "inputs": {"sampler_name": "euler"},
                "class_type": "KSamplerSelect",
            },
            "9": {
                "inputs": {"model": ["1", 0], "conditioning": ["11", 0]},
                "class_type": "BasicGuider",
            },
            "10": {
                "inputs": {
                    "noise": ["7", 0],
                    "guider": ["9", 0],
                    "sampler": ["8", 0],
                    "sigmas": ["6", 0],
                    "latent_image": ["4", 0],
                },
                "class_type": "SamplerCustomAdvanced",
            },
            "11": {
                "inputs": {"guidance": guidance, "conditioning": ["3", 0]},
                "class_type": "FluxGuidance",
            },
            "12": {
                "inputs": {"samples": ["10", 0], "vae": ["5", 0]},
                "class_type": "VAEDecode",
            },
            "13": {
                "inputs": {
                    "filename_prefix": "oelala_flux2",
                    "images": ["12", 0],
                },
                "class_type": "SaveImage",
            },
        }

        # Apply DisTorch2 allocation on the UNet loader (optional compute/donor).
        workflow["1"]["inputs"]["compute_device"] = FLUX2_COMPUTE_DEVICE
        workflow["1"]["inputs"]["virtual_vram_gb"] = FLUX2_VIRTUAL_VRAM_GB
        workflow["1"]["inputs"]["donor_device"] = "cpu"
        workflow["1"]["inputs"]["expert_mode_allocations"] = FLUX2_DISTORCH_ALLOC
        workflow["1"]["inputs"]["eject_models"] = True

        return workflow

    def cost(self, req: GenerationRequest) -> int:
        width = req.width or 1024
        height = req.height or 1024
        if width * height > 1024 * 1024:
            return 4  # HD (2x Flux1 base for a 32B model)
        return 3

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
            raise RuntimeError("Failed to queue Flux 2 workflow to ComfyUI")

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=0,  # Router fills this in
            adapter_name=self.name,
            meta={
                "checkpoint": "flux2-dev-Q4_K_M.gguf",
                "width": req.width or 1024,
                "height": req.height or 1024,
                "steps": req.steps or 20,
            },
        )
