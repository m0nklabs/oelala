"""
I2I Transform adapter — image-to-image with optional face features.

Supports: IP-Adapter FaceID, FaceDetailer, GFPGAN face restore.
Builds ComfyUI workflow inline.
"""

from __future__ import annotations

import logging
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


def _build_i2i_workflow(
    comfyui_filename: str,
    prompt: str,
    negative_prompt: str,
    checkpoint: str,
    denoise: float,
    steps: int,
    cfg: float,
    seed: int,
    sampler_name: str,
    scheduler: str,
    face_id: bool = False,
    face_detailer: bool = False,
    face_restore: bool = False,
    face_id_weight: float = 0.85,
) -> dict:
    """Build the I2I ComfyUI workflow with optional face processing nodes."""
    node_id = 1
    nodes = {}

    # Checkpoint Loader
    ckpt_id = str(node_id)
    nodes[ckpt_id] = {
        "inputs": {"ckpt_name": checkpoint},
        "class_type": "CheckpointLoaderSimple",
    }
    model_ref = [ckpt_id, 0]
    clip_ref = [ckpt_id, 1]
    vae_ref = [ckpt_id, 2]
    node_id += 1

    # Load Source Image
    img_id = str(node_id)
    nodes[img_id] = {
        "inputs": {"image": comfyui_filename, "upload": "image"},
        "class_type": "LoadImage",
    }
    image_ref = [img_id, 0]
    node_id += 1

    # VAE Encode
    vae_enc_id = str(node_id)
    nodes[vae_enc_id] = {
        "inputs": {"pixels": image_ref, "vae": vae_ref},
        "class_type": "VAEEncode",
    }
    latent_ref = [vae_enc_id, 0]
    node_id += 1

    # Positive Prompt
    pos_id = str(node_id)
    nodes[pos_id] = {
        "inputs": {"text": prompt, "clip": clip_ref},
        "class_type": "CLIPTextEncode",
    }
    positive_ref = [pos_id, 0]
    node_id += 1

    # Negative Prompt
    neg_id = str(node_id)
    nodes[neg_id] = {
        "inputs": {"text": negative_prompt, "clip": clip_ref},
        "class_type": "CLIPTextEncode",
    }
    negative_ref = [neg_id, 0]
    node_id += 1

    # Optional: IP-Adapter FaceID
    if face_id:
        ipadapter_loader_id = str(node_id)
        nodes[ipadapter_loader_id] = {
            "inputs": {
                "model": model_ref,
                "preset": "FACEID PLUS V2",
                "lora_strength": 0.6,
                "provider": "CPU",
            },
            "class_type": "IPAdapterUnifiedLoaderFaceID",
        }
        node_id += 1

        ipadapter_apply_id = str(node_id)
        nodes[ipadapter_apply_id] = {
            "inputs": {
                "weight": face_id_weight,
                "weight_faceidv2": face_id_weight,
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 1.0,
                "embeds_scaling": "V only",
                "model": [ipadapter_loader_id, 0],
                "ipadapter": [ipadapter_loader_id, 1],
                "image": image_ref,
            },
            "class_type": "IPAdapterFaceID",
        }
        model_ref = [ipadapter_apply_id, 0]
        node_id += 1

    # KSampler
    sampler_id = str(node_id)
    nodes[sampler_id] = {
        "inputs": {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": denoise,
            "model": model_ref,
            "positive": positive_ref,
            "negative": negative_ref,
            "latent_image": latent_ref,
        },
        "class_type": "KSampler",
    }
    node_id += 1

    # VAE Decode
    vae_dec_id = str(node_id)
    nodes[vae_dec_id] = {
        "inputs": {"samples": [sampler_id, 0], "vae": vae_ref},
        "class_type": "VAEDecode",
    }
    final_image_ref = [vae_dec_id, 0]
    node_id += 1

    # Optional: FaceDetailer
    if face_detailer:
        bbox_id = str(node_id)
        nodes[bbox_id] = {
            "inputs": {"model_name": "bbox/face_yolov8m.pt"},
            "class_type": "UltralyticsDetectorProvider",
        }
        node_id += 1

        sam_id = str(node_id)
        nodes[sam_id] = {
            "inputs": {"model_name": "sam_vit_b_01ec64.pth", "device_mode": "AUTO"},
            "class_type": "SAMLoader",
        }
        node_id += 1

        detailer_id = str(node_id)
        nodes[detailer_id] = {
            "inputs": {
                "guide_size": 384,
                "guide_size_for": True,
                "max_size": 1024,
                "seed": seed,
                "steps": max(15, steps // 2),
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": min(0.4, denoise * 0.6),
                "feather": 5,
                "noise_mask": True,
                "force_inpaint": True,
                "bbox_threshold": 0.5,
                "bbox_dilation": 10,
                "bbox_crop_factor": 3.0,
                "sam_detection_hint": "center-1",
                "sam_dilation": 0,
                "sam_threshold": 0.93,
                "sam_bbox_expansion": 0,
                "sam_mask_hint_threshold": 0.7,
                "sam_mask_hint_use_negative": "False",
                "drop_size": 10,
                "wildcard": "",
                "cycle": 1,
                "image": final_image_ref,
                "model": [ckpt_id, 0],
                "clip": clip_ref,
                "vae": vae_ref,
                "positive": positive_ref,
                "negative": negative_ref,
                "bbox_detector": [bbox_id, 0],
                "sam_model_opt": [sam_id, 0],
            },
            "class_type": "FaceDetailer",
        }
        final_image_ref = [detailer_id, 0]
        node_id += 1

    # Optional: Face Restore (GFPGAN via mtb)
    if face_restore:
        face_model_loader_id = str(node_id)
        nodes[face_model_loader_id] = {
            "inputs": {"model_name": "GFPGANv1.4.pth", "upscale": 1},
            "class_type": "Load Face Enhance Model (mtb)",
        }
        node_id += 1

        restore_id = str(node_id)
        nodes[restore_id] = {
            "inputs": {
                "image": final_image_ref,
                "model": [face_model_loader_id, 0],
                "aligned": False,
                "only_center_face": False,
                "weight": 0.7,
                "save_tmp_steps": False,
            },
            "class_type": "Restore Face (mtb)",
        }
        final_image_ref = [restore_id, 0]
        node_id += 1

    # Save Image
    save_id = str(node_id)
    nodes[save_id] = {
        "inputs": {"filename_prefix": "oelala_i2i", "images": final_image_ref},
        "class_type": "SaveImage",
    }

    return nodes


class I2ITransformAdapter(GenerationAdapter):
    """
    Local I2I transform via ComfyUI — SDXL-based with face features.

    Supports optional IP-Adapter FaceID Plus V2, FaceDetailer, and
    GFPGAN face restoration as toggleable features.
    """

    name = "local-i2i-transform"
    model_family = "sdxl"
    supported_ops = {Operation.TRANSFORM}
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
            default_steps=25,
            default_cfg=7.5,
            supported_samplers=["dpmpp_2m", "euler", "euler_ancestral", "dpmpp_sde"],
            supported_schedulers=["karras", "normal", "simple"],
            supports_negative_prompt=True,
            max_input_images=1,
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        """Build I2I workflow with optional face features."""
        return _build_i2i_workflow(
            comfyui_filename=req.input_images[0] if req.input_images else "",
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or "",
            checkpoint=req.checkpoint or "CyberRealistic_Pony_v14.1_FP16.safetensors",
            denoise=req.denoise or 0.7,
            steps=req.steps or 25,
            cfg=req.cfg or 7.5,
            seed=req.seed or -1,
            sampler_name=req.sampler or "dpmpp_2m",
            scheduler=req.scheduler or "karras",
            face_id=req.face_id or False,
            face_detailer=req.face_detailer or False,
            face_restore=req.face_restore or False,
            face_id_weight=req.face_id_weight or 0.85,
        )

    def cost(self, req: GenerationRequest) -> int:
        base = 2  # SDXL base
        if req.face_id:
            base += 3
        if req.face_detailer:
            base += 2
        if req.face_restore:
            base += 1
        return base

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")

        if not req.input_images:
            raise ValueError("I2I transform requires an input image")

        client = self._get_comfyui()

        workflow = _build_i2i_workflow(
            comfyui_filename=req.input_images[0],
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or "",
            denoise=req.denoise or 0.7,
            checkpoint=req.checkpoint or "CyberRealistic_Pony_v14.1_FP16.safetensors",
            steps=req.steps or 25,
            cfg=req.cfg or 7.5,
            seed=req.seed or -1,
            sampler_name=req.sampler or "dpmpp_2m",
            scheduler=req.scheduler or "karras",
            face_id=req.face_id or False,
            face_detailer=req.face_detailer or False,
            face_restore=req.face_restore or False,
            face_id_weight=req.face_id_weight or 0.85,
        )

        prompt_id = client.queue_prompt(workflow)
        if not prompt_id:
            raise RuntimeError("Failed to queue I2I workflow")

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=0,  # Router fills this in
            adapter_name=self.name,
            meta={
                "face_id": req.face_id,
                "face_detailer": req.face_detailer,
                "face_restore": req.face_restore,
            },
        )
