"""
Qwen Image Edit 2511 — cloud adapter for instruction-based image editing.

Migrated from app.py _build_qwen_edit_workflow() + /generate-qwen-edit endpoint.
Runs on RunPod (requires 48GB+ GPU for fp8mixed model).
"""

from __future__ import annotations

import logging
import os
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


def build_qwen_edit_workflow(
    image_filename: str,
    instruction: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 40,
    cfg: float = 4.0,
    seed: int = 42,
    lightning: bool = False,
    lora_configs: list[dict] | None = None,
) -> dict:
    """
    Build a Qwen-Image-Edit-2511 ComfyUI API workflow.

    Extracted from app.py _build_qwen_edit_workflow().
    """
    if lightning:
        steps = 4
        cfg = 1.0

    workflow: dict = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "qwen_image_edit_2511_fp8mixed.safetensors",
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "type": "qwen_image",
                "device": "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_vae.safetensors"},
        },
        "4": {
            "class_type": "LoadImage",
            "inputs": {"image": image_filename},
        },
        "5": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "6": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"shift": 3.1, "model": ["1", 0]},
        },
        "7": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": instruction,
                "clip": ["2", 0],
                "vae": ["3", 0],
                "image1": ["4", 0],
            },
        },
        "8": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": negative_prompt or "",
                "clip": ["2", 0],
                "vae": ["3", 0],
                "image1": ["4", 0],
            },
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["6", 0],
                "positive": ["7", 0],
                "negative": ["8", 0],
                "latent_image": ["5", 0],
            },
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["9", 0], "vae": ["3", 0]},
        },
        "11": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "oelala_qwen_edit", "images": ["10", 0]},
        },
    }

    # ── LoRA chain ──────────────────────────────────────────────
    last_model_ref: list = ["1", 0]
    lora_node_id = 20

    if lora_configs:
        for lora_cfg in lora_configs:
            lora_name = lora_cfg.get("name", "")
            if not lora_name:
                continue
            strength = lora_cfg.get("strength", 1.0)
            workflow[str(lora_node_id)] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "lora_name": lora_name,
                    "strength_model": strength,
                    "model": last_model_ref,
                },
            }
            last_model_ref = [str(lora_node_id), 0]
            lora_node_id += 1

    if lightning:
        workflow[str(lora_node_id)] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
                "strength_model": 1.0,
                "model": last_model_ref,
            },
        }
        last_model_ref = [str(lora_node_id), 0]

    workflow["6"]["inputs"]["model"] = last_model_ref

    return workflow


class QwenEditCloudAdapter(GenerationAdapter):
    """
    Qwen Image Edit 2511 via RunPod.

    Supports instruction-based image editing with optional LoRAs
    and Lightning mode for fast 4-step generation.
    """

    name = "qwen-cloud-edit"
    model_family = "qwen_image_edit"
    supported_ops = {Operation.EDIT}
    input_types = {MediaType.IMAGE}
    output_type = MediaType.IMAGE
    compute = ComputeTarget.CLOUD
    lora_format = LoraFormat.SINGLE_STAGE

    def __init__(self, submit_to_runpod_fn: Any = None) -> None:
        """
        Args:
            submit_to_runpod_fn: Async function for RunPod submission.
                Signature: async (workflow, user_id, prompt_id, job_info,
                                  images, lora_downloads, prompt_full,
                                  endpoint_id) -> dict
                If None, will attempt lazy import from app module.
        """
        self._submit_to_runpod = submit_to_runpod_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            min_width=512,
            max_width=2048,
            min_height=512,
            max_height=2048,
            resolution_step=16,
            min_steps=1,
            max_steps=100,
            default_steps=40,
            default_cfg=4.0,
            supported_samplers=["euler"],
            supported_schedulers=["simple"],
            max_loras=5,
            supports_lightning=True,
            supports_negative_prompt=True,
            max_input_images=1,
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        lora_dicts = (
            [lr.model_dump(exclude_none=True) for lr in req.loras]
            if req.loras
            else None
        )

        return build_qwen_edit_workflow(
            image_filename="input.png",
            instruction=req.instruction or req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width or 1024,
            height=req.height or 1024,
            steps=req.steps or 40,
            cfg=req.cfg or 4.0,
            seed=req.seed,
            lightning=req.lightning,
            lora_configs=lora_dicts,
        )

    def cost(self, req: GenerationRequest) -> int:
        """
        Credit calculation for Qwen Edit:
        - Base: 15 credits
        - Full quality (non-lightning): +5
        - Per LoRA: +2
        """
        credits = 15
        if not req.lightning:
            credits += 5
        credits += len(req.loras) * 2
        return credits

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        """
        Submit Qwen Edit job to RunPod.

        The actual RunPod submission uses _submit_to_runpod from app.py.
        This adapter builds the workflow and delegates execution.
        """
        import uuid

        prompt_id = str(uuid.uuid4())
        endpoint_id = os.getenv("RUNPOD_QWEN_ENDPOINT_ID")

        if not endpoint_id:
            raise RuntimeError(
                "Qwen Edit requires RUNPOD_QWEN_ENDPOINT_ID to be configured"
            )

        # The input image should be in req.input_images[0] as base64
        if not req.input_images:
            raise ValueError("Qwen Edit requires an input image")

        upload_filename = f"qwen_edit_input_{uuid.uuid4().hex[:8]}.png"
        image_b64 = req.input_images[0]

        # Build LoRA download URLs for cloud worker
        lora_dicts = (
            [lr.model_dump(exclude_none=True) for lr in req.loras] if req.loras else []
        )

        from ... import lora_utils

        cloud_lora_downloads = (
            lora_utils.build_lora_download_list(lora_dicts) if lora_dicts else []
        )

        # Build workflow
        workflow = build_qwen_edit_workflow(
            image_filename=upload_filename,
            instruction=req.instruction or req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width or 1024,
            height=req.height or 1024,
            steps=req.steps or 40,
            cfg=req.cfg or 4.0,
            seed=req.seed,
            lightning=req.lightning,
            lora_configs=lora_dicts if lora_dicts else None,
        )

        # Prepare images dict for RunPod
        input_images_b64 = {upload_filename: image_b64}

        # Job metadata
        job_info = {
            "user_id": req.user_id or "adapter",
            "prompt": req.instruction or req.prompt,
            "job_type": "qwen_edit",
            "input_image": upload_filename,
            "settings": {
                "instruction": req.instruction or req.prompt,
                "negative_prompt": req.negative_prompt,
                "width": req.width or 1024,
                "height": req.height or 1024,
                "steps": req.steps or 40,
                "cfg": req.cfg or 4.0,
                "seed": req.seed,
                "lightning": req.lightning,
                "lora_count": len(lora_dicts),
            },
        }

        # Get submit function (injected or lazy import)
        submit_fn = self._submit_to_runpod
        if submit_fn is None:
            try:
                from app import _submit_to_runpod

                submit_fn = _submit_to_runpod
            except ImportError:
                raise RuntimeError(
                    "_submit_to_runpod not available — pass it via constructor"
                )

        result = await submit_fn(
            workflow=workflow,
            user_id=job_info["user_id"],
            prompt_id=prompt_id,
            job_info=job_info,
            images=input_images_b64,
            lora_downloads=cloud_lora_downloads if cloud_lora_downloads else None,
            prompt_full=req.instruction or req.prompt,
            endpoint_id=endpoint_id,
        )

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_cloud",
            compute_target=ComputeTarget.CLOUD,
            credits_used=0,  # Router fills this in
            runpod_job_id=result.get("runpod_job_id"),
            adapter_name=self.name,
            meta={
                "instruction": req.instruction or req.prompt,
                "seed": req.seed,
                "width": req.width or 1024,
                "height": req.height or 1024,
                "steps": req.steps or 40,
                "cfg": req.cfg or 4.0,
                "lightning": req.lightning,
                "lora_count": len(lora_dicts),
            },
        )
