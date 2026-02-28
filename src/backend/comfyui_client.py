#!/usr/bin/env python3
"""
ComfyUI API Client for Oelala Backend
Enables integration with ComfyUI for Wan2.2 Q5 GGUF workflows
"""

import json
import uuid
import time
import requests
import websocket
import random
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
import logging
import io
import copy

# Import for auto-upload functionality (legacy sync client)
from storage_client import get_client as get_storage_client

# Guardian LLM proxy — VRAM management
from guardian_client import get_guardian

# Import MediaService for async uploads with Supabase sync
try:
    from media_service import MediaService

    _media_service: Optional[MediaService] = None

    def get_media_service() -> MediaService:
        """Get or create the global MediaService instance."""
        global _media_service
        if _media_service is None:
            _media_service = MediaService()
        return _media_service
except ImportError:
    get_media_service = None  # type: ignore

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Workflow Directory and Dynamic Loading
# ─────────────────────────────────────────────────────────────────────────────
WORKFLOWS_DIR = Path("/home/flip/oelala/workflows")

# Available I2V generation modes with their workflow files
I2V_GENERATION_MODES = {
    "standard": {
        "name": "Standard Q6 DisTorch2",
        "description": "High quality dual-noise 14B Q6_K model",
        "workflow_file": "ImageToVideo/wan22_i2v_distorch2_api.json",
        "default_steps": 6,
        "default_cfg": 1.0,
    },
    "nsfw_lora": {
        "name": "NSFW LoRA Preset",
        "description": "Standard workflow with NSFW LoRAs pre-configured",
        "workflow_file": "ImageToVideo/wan22_i2v_ltx2_audio_api.json",
        "default_steps": 6,
        "default_cfg": 1.0,
    },
}

# Available T2V (Text-to-Video) generation modes - different base models
T2V_GENERATION_MODES = {
    "wan22": {
        "name": "Wan2.2 14B",
        "description": "High quality Wan 2.2 T2V model with DisTorch2 multi-GPU",
        "workflow_file": None,  # Uses built-in workflow builder
        "model_type": "wan22",
        "default_steps": 6,
        "default_cfg": 1.0,
        "max_frames": 81,
        "default_frames": 41,
    },
    "ltx2": {
        "name": "LTX-2 19B",
        "description": "Lightricks LTX-2 19B distilled model, faster inference",
        "workflow_file": "ltx2_distorch2_multigpu_api.json",
        "model_type": "ltx2",
        "default_steps": 8,
        "default_cfg": 1.0,
        "max_frames": 97,
        "default_frames": 25,
    },
}


def load_workflow_from_file(workflow_path: str) -> Optional[Dict]:
    """Load a workflow JSON file and return as dict."""
    full_path = WORKFLOWS_DIR / workflow_path
    if not full_path.exists():
        logger.error(f"❌ Workflow file not found: {full_path}")
        return None
    try:
        with open(full_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Failed to load workflow {workflow_path}: {e}")
        return None


def get_available_i2v_modes() -> Dict:
    """Return available I2V generation modes."""
    return I2V_GENERATION_MODES


def get_available_t2v_modes() -> Dict:
    """Return available T2V generation modes (base models)."""
    return T2V_GENERATION_MODES


def build_ltx2_t2v_workflow(
    prompt: str,
    negative_prompt: str = "blurry, low quality, distorted, watermark",
    width: int = 768,
    height: int = 512,
    num_frames: int = 25,
    steps: int = 8,
    cfg: float = 1.0,
    seed: int = None,
    filename_prefix: str = "oelala_ltx2_t2v",
) -> Optional[Dict]:
    """
    Build LTX-2 T2V workflow by loading template and injecting parameters.

    LTX-2 19B distilled - faster inference, good quality for T2V.
    Uses DisTorch2 multi-GPU distribution.
    """
    workflow = load_workflow_from_file("ltx2_distorch2_multigpu_api.json")
    if not workflow:
        logger.error("❌ Failed to load LTX-2 T2V workflow template")
        return None

    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    logger.info(
        f"🎬 Building LTX-2 T2V workflow: {width}x{height}, {num_frames} frames, {steps} steps"
    )

    # Update prompts (nodes 4 and 5 are CLIPTextEncode)
    if "4" in workflow:
        workflow["4"]["inputs"]["text"] = prompt
    if "5" in workflow:
        workflow["5"]["inputs"]["text"] = negative_prompt

    # Update dimensions and frame count (node 7: EmptyLTXVLatentVideo)
    if "7" in workflow:
        workflow["7"]["inputs"]["width"] = width
        workflow["7"]["inputs"]["height"] = height
        workflow["7"]["inputs"]["length"] = num_frames

    # Update scheduler steps (node 8: LTXVScheduler)
    if "8" in workflow:
        workflow["8"]["inputs"]["steps"] = steps

    # Update sampler seed and cfg (node 9: SamplerCustom)
    if "9" in workflow:
        workflow["9"]["inputs"]["noise_seed"] = seed
        workflow["9"]["inputs"]["cfg"] = cfg

    # Update output filename (node 12: VHS_VideoCombine)
    if "12" in workflow:
        workflow["12"]["inputs"]["filename_prefix"] = filename_prefix

    logger.debug(f"✅ LTX-2 T2V workflow built: seed={seed}")
    return workflow


def build_ltx2_i2v_workflow(
    image_name: str,
    prompt: str,
    negative_prompt: str = "blurry, low quality, distorted, watermark, artifacts",
    width: int = 576,
    height: int = 1024,
    num_frames: int = 97,
    steps: int = 20,
    cfg: float = 3.0,
    seed: int = None,
    filename_prefix: str = "oelala_ltx2_i2v",
    fps: int = 25,
) -> Optional[Dict]:
    """
    Build LTX-2 I2V workflow by loading template and injecting parameters.

    LTX-2 19B - single model (no high/low noise), uses Gemma text encoder.
    Uses LTXVImgToVideo node for image conditioning.
    """
    workflow = load_workflow_from_file("ImageToVideo/ltx2_i2v_api.json")
    if not workflow:
        logger.error("❌ Failed to load LTX-2 I2V workflow template")
        return None

    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    logger.info(
        f"🎬 Building LTX-2 I2V workflow: {width}x{height}, {num_frames} frames, {steps} steps"
    )

    # Update input image (node 4: LoadImage)
    if "4" in workflow:
        workflow["4"]["inputs"]["image"] = image_name

    # Update prompts (nodes 5 and 6 are CLIPTextEncode)
    if "5" in workflow:
        workflow["5"]["inputs"]["text"] = prompt
    if "6" in workflow:
        workflow["6"]["inputs"]["text"] = negative_prompt

    # Update dimensions, frame count and strength (node 7: LTXVImgToVideo)
    if "7" in workflow:
        workflow["7"]["inputs"]["width"] = width
        workflow["7"]["inputs"]["height"] = height
        workflow["7"]["inputs"]["length"] = num_frames
        workflow["7"]["inputs"]["strength"] = 1.0

    # Update frame rate (node 8: LTXVConditioning)
    if "8" in workflow:
        workflow["8"]["inputs"]["frame_rate"] = float(fps)

    # Update scheduler steps (node 9: LTXVScheduler)
    if "9" in workflow:
        workflow["9"]["inputs"]["steps"] = steps

    # Update sampler seed and cfg (node 11: SamplerCustom)
    if "11" in workflow:
        workflow["11"]["inputs"]["noise_seed"] = seed
        workflow["11"]["inputs"]["cfg"] = cfg

    # Update output filename and fps (node 13: VHS_VideoCombine)
    if "13" in workflow:
        workflow["13"]["inputs"]["filename_prefix"] = filename_prefix
        workflow["13"]["inputs"]["frame_rate"] = fps

    logger.debug(f"✅ LTX-2 I2V workflow built: image={image_name}, seed={seed}")
    return workflow


# ─────────────────────────────────────────────────────────────────────────────
# WAN 2.2 Enhanced NSFW FAST MOVE V2 Q4KM Lightning (I2V)
# Settings: steps=4 (2+2), cfg=1, euler simple scheduler
# Already includes Lightning LoRAs – do NOT add extra LoRAs
# ─────────────────────────────────────────────────────────────────────────────
WAN22_ENHANCED_Q4KM_API_WORKFLOW = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "example_480.png"}},
    "2": {
        "class_type": "LoadWanVideoT5TextEncoderMultiGPU",
        "inputs": {
            "model_name": "umt5-xxl-enc-bf16.safetensors",
            "precision": "bf16",
            "device": "cuda:0",
        },
    },
    "3": {
        "class_type": "WanVideoTextEncodeMultiGPU",
        "inputs": {
            "positive_prompt": "motion, smooth camera movement",
            "negative_prompt": "",
            "force_offload": True,
            "t5": ["2", 0],
        },
    },
    "4": {
        "class_type": "WanVideoVAELoaderMultiGPU",
        "inputs": {
            "model_name": "Wan2.1_VAE.safetensors",
            "device": "cuda:0",
            "dtype": "bf16",
        },
    },
    "5": {
        "class_type": "CLIPVisionLoader",
        "inputs": {"clip_name": "clip-vit-large.safetensors"},
    },
    "6": {
        "class_type": "WanVideoClipVisionEncode",
        "inputs": {
            "strength_1": 1.0,
            "strength_2": 0.0,
            "crop": "center",
            "force_offload": True,
            "combine_embeds": "average",
            "clip_vision": ["5", 0],
            "image_1": ["1", 0],
        },
    },
    "7": {
        "class_type": "WanVideoBlockSwapMultiGPU",
        "inputs": {
            "blocks_to_swap": 40,
            "offload_img_emb": True,
            "offload_txt_emb": True,
            "swap_device": "cpu",
        },
    },
    "8": {
        "class_type": "WanVideoModelLoaderMultiGPU",
        "inputs": {
            "model": "wan22_nsfw_fastmove_v2_Q4KM_HIGH.gguf",
            "base_precision": "bf16",
            "quantization": "disabled",
            "load_device": "offload_device",
            "compute_device": "cuda:0",
            "attention_mode": "sageattn",
            "block_swap_args": ["7", 0],
        },
    },
    "9": {
        "class_type": "WanVideoImageToVideoEncodeMultiGPU",
        "inputs": {
            "width": 480,
            "height": 480,
            "num_frames": 41,
            "noise_aug_strength": 0.0,
            "start_latent_strength": 1.0,
            "end_latent_strength": 1.0,
            "force_offload": True,
            "vae": ["4", 0],
            "clip_embeds": ["6", 0],
            "start_image": ["1", 0],
            "load_device": ["8", 1],
        },
    },
    "10": {
        "class_type": "WanVideoSamplerMultiGPU",
        "inputs": {
            "steps": 4,
            "cfg": 1.0,
            "seed": 42,
            "shift": True,
            "scheduler": "euler",
            "riflex_freq_index": 0,
            "force_offload": True,
            "model": ["8", 0],
            "compute_device": ["8", 1],
            "image_embeds": ["9", 0],
            "text_embeds": ["3", 0],
        },
    },
    "11": {
        "class_type": "WanVideoDecodeMultiGPU",
        "inputs": {
            "enable_vae_tiling": True,
            "tile_x": 272,
            "tile_y": 272,
            "tile_stride_x": 192,
            "tile_stride_y": 192,
            "vae": ["4", 0],
            "samples": ["10", 0],
            "load_device": ["8", 1],
        },
    },
    "12": {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "frame_rate": 16,
            "loop_count": 0,
            "filename_prefix": "oelala_wan22enh",
            "format": "video/h264-mp4",
            "pingpong": False,
            "save_output": True,
            "crf": 19,
            "save_metadata": True,
            "trim_to_audio": False,
            "images": ["11", 0],
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# WAN 2.2 I2V DisTorch2 Dual-Pass Workflow (Q6_K 14B models)
# Uses CONVERTED T5, dual-pass sampling (high noise → low noise), expert memory allocation
# This workflow scales well with available VRAM via expert_mode_allocations
# Supports optional LoRA loading for both high and low noise models
# ─────────────────────────────────────────────────────────────────────────────
WAN22_I2V_Q6_API_WORKFLOW = {
    # Node 1: High Noise GGUF Model Loader (DisTorch2)
    "1": {
        "class_type": "UnetLoaderGGUFAdvancedDisTorch2MultiGPU",
        "inputs": {
            "unet_name": "wan2.2_i2v_high_noise_14B_Q6_K.gguf",
            "dequant_dtype": "default",
            "patch_dtype": "default",
            "patch_on_device": False,
            "compute_device": "cuda:0",
            "virtual_vram_gb": 16,
            "donor_device": "cuda:1",
            "expert_mode_allocations": "cuda:0,0.25gb;cuda:1,8gb;cpu,*",
            "eject_models": True,
        },
    },
    # Node 2: Low Noise GGUF Model Loader (DisTorch2)
    "2": {
        "class_type": "UnetLoaderGGUFAdvancedDisTorch2MultiGPU",
        "inputs": {
            "unet_name": "wan2.2_i2v_low_noise_14B_Q6_K.gguf",
            "dequant_dtype": "default",
            "patch_dtype": "default",
            "patch_on_device": False,
            "compute_device": "cuda:0",
            "virtual_vram_gb": 16,
            "donor_device": "cuda:1",
            "expert_mode_allocations": "cuda:0,0.25gb;cuda:1,8gb;cpu,*",
            "eject_models": True,
        },
    },
    # Node 3: VAE Loader (DisTorch2)
    "3": {
        "class_type": "VAELoaderDisTorch2MultiGPU",
        "inputs": {
            "vae_name": "wan_2.1_vae.safetensors",
            "compute_device": "cuda:0",
            "virtual_vram_gb": 16,
            "donor_device": "cuda:1",
            "expert_mode_allocations": "cuda:0,0.25gb;cuda:1,8gb;cpu,*",
            "eject_models": True,
        },
    },
    # Node 4: T5-XXL CLIP Loader (DisTorch2) - CONVERTED model!
    "4": {
        "class_type": "CLIPLoaderDisTorch2MultiGPU",
        "inputs": {
            "clip_name": "umt5-xxl-enc-bf16.safetensors",
            "type": "wan",
            "device": "cuda:0",
            "virtual_vram_gb": 16,
            "donor_device": "cuda:1",
            "expert_mode_allocations": "cuda:0,0.25gb;cuda:1,8gb;cpu,*",
            "eject_models": True,
        },
    },
    # Node 5: ModelSamplingSD3 for High Noise model (shift=8)
    "5": {"class_type": "ModelSamplingSD3", "inputs": {"shift": 8, "model": ["1", 0]}},
    # Node 6: ModelSamplingSD3 for Low Noise model (shift=8)
    "6": {"class_type": "ModelSamplingSD3", "inputs": {"shift": 8, "model": ["2", 0]}},
    # Node 7: SageAttention for High Noise model
    "7": {
        "class_type": "PathchSageAttentionKJ",
        "inputs": {
            "sage_attention": "sageattn_qk_int8_pv_fp16_triton",
            "allow_compile": False,
            "model": ["5", 0],
        },
    },
    # Node 8: SageAttention for Low Noise model
    "8": {
        "class_type": "PathchSageAttentionKJ",
        "inputs": {
            "sage_attention": "sageattn_qk_int8_pv_fp16_triton",
            "allow_compile": False,
            "model": ["6", 0],
        },
    },
    # Node 17: LoRA Loader for High Noise model (optional, bypassed by default)
    # When enabled, loads between SageAttn and Sampler
    "17": {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "lora_name": "",  # Empty = disabled
            "strength_model": 1.0,
            "model": ["7", 0],
        },
    },
    # Node 18: LoRA Loader for Low Noise model (optional, bypassed by default)
    "18": {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "lora_name": "",  # Empty = disabled
            "strength_model": 1.0,
            "model": ["8", 0],
        },
    },
    # Node 19 removed: AspectRatioResolution_Warper - now using direct width/height in node 12
    # Node 9: Positive Prompt (CLIPTextEncode)
    "9": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "smooth motion, cinematic", "clip": ["4", 0]},
    },
    # Node 10: Negative Prompt (CLIPTextEncode)
    "10": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "low quality, blurry, out of focus, unstable camera, artifacts, distortion, low resolution, overexposed, underexposed, color banding, missing details, unrealistic lighting, flickering shadows, frame stutter, ghosting, bad reflections, unrealistic motion, pixelated textures, wrong physics, broken animation, rendering artifacts, compression noise, jitter, visual glitches",
            "clip": ["4", 0],
        },
    },
    # Node 11: Load Image
    "11": {"class_type": "LoadImage", "inputs": {"image": "example_480.png"}},
    # Node 20: ImageResize - Resize input image to exact target resolution
    # Uses "fill / crop" to ensure exact dimensions with center cropping
    "20": {
        "class_type": "ImageResize+",
        "inputs": {
            "image": ["11", 0],
            "width": 480,
            "height": 848,
            "interpolation": "lanczos",
            "method": "fill / crop",
            "condition": "always",
            "multiple_of": 8,
        },
    },
    # Node 12: WanImageToVideo - encodes image to latent + conditioning
    # Uses direct width/height values (aspect ratio calculated by script)
    "12": {
        "class_type": "WanImageToVideo",
        "inputs": {
            "width": 480,
            "height": 848,
            "length": 41,
            "batch_size": 1,
            "positive": ["9", 0],
            "negative": ["10", 0],
            "vae": ["3", 0],
            "start_image": ["20", 0],
        },
    },
    # Node 13: KSamplerAdvanced - Pass 1 (High Noise) steps 0-3
    # Uses LoRA-wrapped model (node 17) instead of direct SageAttn output
    "13": {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "add_noise": "enable",
            "noise_seed": 42,
            "steps": 6,
            "cfg": 1.0,
            "sampler_name": "uni_pc",
            "scheduler": "normal",
            "start_at_step": 0,
            "end_at_step": 3,
            "return_with_leftover_noise": "enable",
            "model": ["17", 0],
            "positive": ["12", 0],
            "negative": ["12", 1],
            "latent_image": ["12", 2],
        },
    },
    # Node 14: KSamplerAdvanced - Pass 2 (Low Noise) steps 3+
    # Uses LoRA-wrapped model (node 18) instead of direct SageAttn output
    "14": {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "add_noise": "disable",
            "noise_seed": 0,
            "steps": 6,
            "cfg": 1.0,
            "sampler_name": "uni_pc",
            "scheduler": "normal",
            "start_at_step": 3,
            "end_at_step": 10000,
            "return_with_leftover_noise": "disable",
            "model": ["18", 0],
            "positive": ["12", 0],
            "negative": ["12", 1],
            "latent_image": ["13", 0],
        },
    },
    # Node 15: VAE Decode
    "15": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["14", 0], "vae": ["3", 0]},
    },
    # Node 16: Save Video (VHS_VideoCombine)
    "16": {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "frame_rate": 16,
            "loop_count": 0,
            "filename_prefix": "oelala_distorch2",
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 19,
            "save_metadata": True,
            "trim_to_audio": False,
            "pingpong": False,
            "save_output": True,
            "images": ["15", 0],
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# API format workflow for wan22_i2v_14b_Q5_distorch2
# Pre-built with all connections, ready for ComfyUI /prompt API
# ─────────────────────────────────────────────────────────────────────────────
WAN22_I2V_Q5_API_WORKFLOW = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "example_480.png"}},
    "2": {
        "class_type": "LoadWanVideoT5TextEncoderMultiGPU",
        "inputs": {
            "model_name": "umt5-xxl-enc-bf16.safetensors",
            "precision": "bf16",
            "device": "cuda:0",
        },
    },
    "3": {
        "class_type": "WanVideoTextEncodeMultiGPU",
        "inputs": {
            "positive_prompt": "a cat playing with yarn, smooth motion",
            "negative_prompt": "",
            "force_offload": True,
            "t5": ["2", 0],
        },
    },
    "4": {
        "class_type": "WanVideoVAELoaderMultiGPU",
        "inputs": {
            "model_name": "Wan2.1_VAE.safetensors",
            "device": "cuda:0",
            "dtype": "bf16",
        },
    },
    "5": {
        "class_type": "CLIPVisionLoader",
        "inputs": {"clip_name": "clip-vit-large.safetensors"},
    },
    "6": {
        "class_type": "WanVideoClipVisionEncode",
        "inputs": {
            "strength_1": 1.0,
            "strength_2": 0.0,
            "crop": "center",
            "force_offload": True,
            "combine_embeds": "average",
            "clip_vision": ["5", 0],
            "image_1": ["1", 0],
        },
    },
    "7": {
        "class_type": "WanVideoBlockSwapMultiGPU",
        "inputs": {
            "blocks_to_swap": 40,
            "offload_img_emb": True,
            "offload_txt_emb": True,
            "swap_device": "cpu",
        },
    },
    "8": {
        "class_type": "WanVideoModelLoaderMultiGPU",
        "inputs": {
            "model": "wan2.2_i2v_low_noise_14B_Q6_K.gguf",
            "base_precision": "bf16",
            "quantization": "disabled",
            "load_device": "offload_device",
            "compute_device": "cuda:0",
            "attention_mode": "sageattn",
            "block_swap_args": ["7", 0],
        },
    },
    "9": {
        "class_type": "WanVideoImageToVideoEncodeMultiGPU",
        "inputs": {
            "width": 480,
            "height": 480,
            "num_frames": 41,
            "noise_aug_strength": 0.0,
            "start_latent_strength": 1.0,
            "end_latent_strength": 1.0,
            "force_offload": True,
            "vae": ["4", 0],
            "clip_embeds": ["6", 0],
            "start_image": ["1", 0],
            "load_device": ["8", 1],
        },
    },
    "10": {
        "class_type": "WanVideoSamplerMultiGPU",
        "inputs": {
            "steps": 6,
            "cfg": 5.0,
            "seed": 42,
            "shift": True,
            "scheduler": "unipc",
            "riflex_freq_index": 0,
            "force_offload": True,
            "model": ["8", 0],
            "compute_device": ["8", 1],
            "image_embeds": ["9", 0],
            "text_embeds": ["3", 0],
        },
    },
    "11": {
        "class_type": "WanVideoDecodeMultiGPU",
        "inputs": {
            "enable_vae_tiling": True,
            "tile_x": 272,
            "tile_y": 272,
            "tile_stride_x": 192,
            "tile_stride_y": 192,
            "vae": ["4", 0],
            "samples": ["10", 0],
            "load_device": ["8", 1],
        },
    },
    "12": {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "frame_rate": 16,
            "loop_count": 0,
            "filename_prefix": "oelala_wan22",
            "format": "video/h264-mp4",
            "pingpong": False,
            "save_output": True,
            "crf": 19,
            "save_metadata": True,
            "trim_to_audio": False,
            "images": ["11", 0],
        },
    },
}

# Legacy node-format workflow (for reference/UI display)
WAN22_I2V_Q5_WORKFLOW = {
    "last_node_id": 14,
    "last_link_id": 18,
    "nodes": [
        {
            "id": 1,
            "type": "LoadImage",
            "pos": [50, 400],
            "size": [250, 300],
            "flags": {},
            "order": 0,
            "mode": 0,
            "outputs": [
                {"name": "IMAGE", "type": "IMAGE", "links": [1, 5], "slot_index": 0},
                {"name": "MASK", "type": "MASK", "links": None},
            ],
            "properties": {"Node name for S&R": "LoadImage"},
            "widgets_values": ["input_image.png"],
            "title": "Input Image",
        },
        {
            "id": 2,
            "type": "LoadWanVideoT5TextEncoderMultiGPU",
            "pos": [50, 50],
            "size": [400, 150],
            "flags": {},
            "order": 1,
            "mode": 0,
            "outputs": [
                {
                    "name": "TEXT_ENCODER",
                    "type": "WANTEXTENCODER",
                    "links": [2],
                    "slot_index": 0,
                }
            ],
            "properties": {"Node name for S&R": "LoadWanVideoT5TextEncoderMultiGPU"},
            "widgets_values": ["umt5-xxl-enc-bf16.safetensors", "bf16", "cuda:0"],
            "title": "T5 Text Encoder",
        },
        {
            "id": 3,
            "type": "WanVideoTextEncodeMultiGPU",
            "pos": [500, 50],
            "size": [400, 150],
            "flags": {},
            "order": 5,
            "mode": 0,
            "inputs": [{"name": "t5", "type": "WANTEXTENCODER", "link": 2}],
            "outputs": [
                {
                    "name": "TEXT_EMBEDS",
                    "type": "WANVIDEOTEXTEMBEDS",
                    "links": [10],
                    "slot_index": 0,
                }
            ],
            "properties": {"Node name for S&R": "WanVideoTextEncodeMultiGPU"},
            "widgets_values": ["motion prompt here", "", True],
            "title": "Prompt",
        },
        {
            "id": 4,
            "type": "WanVideoVAELoaderMultiGPU",
            "pos": [50, 750],
            "size": [350, 100],
            "flags": {},
            "order": 2,
            "mode": 0,
            "outputs": [
                {"name": "VAE", "type": "WANVAE", "links": [3, 12], "slot_index": 0}
            ],
            "properties": {"Node name for S&R": "WanVideoVAELoaderMultiGPU"},
            "widgets_values": ["Wan2.1_VAE.safetensors", "cuda:0", "bf16"],
            "title": "VAE Loader",
        },
        {
            "id": 5,
            "type": "CLIPVisionLoader",
            "pos": [50, 250],
            "size": [300, 60],
            "flags": {},
            "order": 3,
            "mode": 0,
            "outputs": [
                {
                    "name": "CLIP_VISION",
                    "type": "CLIP_VISION",
                    "links": [4],
                    "slot_index": 0,
                }
            ],
            "properties": {"Node name for S&R": "CLIPVisionLoader"},
            "widgets_values": ["clip-vit-large.safetensors"],
            "title": "CLIP Vision",
        },
        {
            "id": 6,
            "type": "WanVideoClipVisionEncode",
            "pos": [400, 300],
            "size": [350, 120],
            "flags": {},
            "order": 6,
            "mode": 0,
            "inputs": [
                {"name": "clip_vision", "type": "CLIP_VISION", "link": 4},
                {"name": "image_1", "type": "IMAGE", "link": 5},
            ],
            "outputs": [
                {
                    "name": "CLIP_EMBEDS",
                    "type": "WANVIDIMAGE_CLIPEMBEDS",
                    "links": [6],
                    "slot_index": 0,
                }
            ],
            "properties": {"Node name for S&R": "WanVideoClipVisionEncode"},
            "widgets_values": [1.0, 0.0, "center", "average", True],
            "title": "CLIP Vision Encode",
        },
        {
            "id": 7,
            "type": "WanVideoBlockSwapMultiGPU",
            "pos": [450, 550],
            "size": [350, 180],
            "flags": {},
            "order": 4,
            "mode": 0,
            "outputs": [
                {
                    "name": "BLOCK_SWAP_ARGS",
                    "type": "BLOCKSWAPARGS",
                    "links": [7],
                    "slot_index": 0,
                }
            ],
            "properties": {"Node name for S&R": "WanVideoBlockSwapMultiGPU"},
            "widgets_values": [40, True, True, "cpu"],
            "title": "DisTorch2 CPU Offload",
        },
        {
            "id": 8,
            "type": "WanVideoModelLoaderMultiGPU",
            "pos": [850, 400],
            "size": [450, 250],
            "flags": {},
            "order": 7,
            "mode": 0,
            "inputs": [{"name": "block_swap_args", "type": "BLOCKSWAPARGS", "link": 7}],
            "outputs": [
                {
                    "name": "MODEL",
                    "type": "WANVIDEOMODEL",
                    "links": [8],
                    "slot_index": 0,
                },
                {
                    "name": "COMPUTE_DEVICE",
                    "type": "MULTIGPUDEVICE",
                    "links": [9, 11, 13],
                    "slot_index": 1,
                },
            ],
            "properties": {"Node name for S&R": "WanVideoModelLoaderMultiGPU"},
            "widgets_values": [
                "wan2.2_i2v_low_noise_14B_Q6_K.gguf",
                "bf16",
                "disabled",
                "offload_device",
                "cuda:0",
                "sageattn",
                "default",
            ],
            "title": "Q5 Model + SageAttention",
        },
        {
            "id": 9,
            "type": "WanVideoImageToVideoEncodeMultiGPU",
            "pos": [850, 700],
            "size": [450, 280],
            "flags": {},
            "order": 8,
            "mode": 0,
            "inputs": [
                {"name": "vae", "type": "WANVAE", "link": 3},
                {"name": "clip_embeds", "type": "WANVIDIMAGE_CLIPEMBEDS", "link": 6},
                {"name": "start_image", "type": "IMAGE", "link": 1},
                {"name": "load_device", "type": "MULTIGPUDEVICE", "link": 9},
            ],
            "outputs": [
                {
                    "name": "IMAGE_EMBEDS",
                    "type": "WANVIDIMAGE_EMBEDS",
                    "links": [14],
                    "slot_index": 0,
                }
            ],
            "properties": {"Node name for S&R": "WanVideoImageToVideoEncodeMultiGPU"},
            "widgets_values": [480, 480, 41, 0.0, 1.0, 1.0, True],
            "title": "I2V Encode",
        },
        {
            "id": 10,
            "type": "WanVideoSamplerMultiGPU",
            "pos": [1350, 450],
            "size": [450, 400],
            "flags": {},
            "order": 9,
            "mode": 0,
            "inputs": [
                {"name": "model", "type": "WANVIDEOMODEL", "link": 8},
                {"name": "compute_device", "type": "MULTIGPUDEVICE", "link": 11},
                {"name": "image_embeds", "type": "WANVIDIMAGE_EMBEDS", "link": 14},
                {"name": "text_embeds", "type": "WANVIDEOTEXTEMBEDS", "link": 10},
            ],
            "outputs": [
                {"name": "SAMPLES", "type": "LATENT", "links": [15], "slot_index": 0}
            ],
            "properties": {"Node name for S&R": "WanVideoSamplerMultiGPU"},
            "widgets_values": [
                6,
                5.0,
                5.0,
                42,
                "randomize",
                True,
                "unipc",
                0,
                1.0,
                False,
                "comfy",
                0,
                -1,
                False,
            ],
            "title": "Sampler",
        },
        {
            "id": 11,
            "type": "WanVideoDecodeMultiGPU",
            "pos": [1850, 500],
            "size": [400, 220],
            "flags": {},
            "order": 10,
            "mode": 0,
            "inputs": [
                {"name": "vae", "type": "WANVAE", "link": 12},
                {"name": "samples", "type": "LATENT", "link": 15},
                {"name": "load_device", "type": "MULTIGPUDEVICE", "link": 13},
            ],
            "outputs": [
                {"name": "IMAGES", "type": "IMAGE", "links": [16], "slot_index": 0}
            ],
            "properties": {"Node name for S&R": "WanVideoDecodeMultiGPU"},
            "widgets_values": [True, 272, 272, 192, 192],
            "title": "VAE Decode",
        },
        {
            "id": 12,
            "type": "VHS_VideoCombine",
            "pos": [2300, 450],
            "size": [450, 350],
            "flags": {},
            "order": 11,
            "mode": 0,
            "inputs": [{"name": "images", "type": "IMAGE", "link": 16}],
            "properties": {"Node name for S&R": "VHS_VideoCombine"},
            "widgets_values": {
                "frame_rate": 16,
                "loop_count": 0,
                "filename_prefix": "oelala_wan22",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            },
            "title": "Save Video",
        },
    ],
    "links": [
        [1, 1, 0, 9, 2, "IMAGE"],
        [2, 2, 0, 3, 0, "WANTEXTENCODER"],
        [3, 4, 0, 9, 0, "WANVAE"],
        [4, 5, 0, 6, 0, "CLIP_VISION"],
        [5, 1, 0, 6, 1, "IMAGE"],
        [6, 6, 0, 9, 1, "WANVIDIMAGE_CLIPEMBEDS"],
        [7, 7, 0, 8, 0, "BLOCKSWAPARGS"],
        [8, 8, 0, 10, 0, "WANVIDEOMODEL"],
        [9, 8, 1, 9, 3, "MULTIGPUDEVICE"],
        [10, 3, 0, 10, 3, "WANVIDEOTEXTEMBEDS"],
        [11, 8, 1, 10, 1, "MULTIGPUDEVICE"],
        [12, 4, 0, 11, 0, "WANVAE"],
        [13, 8, 1, 11, 2, "MULTIGPUDEVICE"],
        [14, 9, 0, 10, 2, "WANVIDIMAGE_EMBEDS"],
        [15, 10, 0, 11, 1, "LATENT"],
        [16, 11, 0, 12, 0, "IMAGE"],
    ],
    "groups": [],
    "config": {},
    "extra": {"ds": {"scale": 0.65, "offset": [0, 0]}},
    "version": 0.4,
}


class ComfyUIClient:
    """Client for ComfyUI API integration"""

    def __init__(self, host: str = "localhost", port: int = 8188):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.client_id = str(uuid.uuid4())
        # Job tracking: prompt_id -> {user_id, prompt, settings, started_at}
        self.job_metadata = {}
        # Thread-safe lock for job_metadata access
        self._metadata_lock = threading.Lock()

    def is_available(self) -> bool:
        """Check if ComfyUI is running and accessible"""
        try:
            resp = requests.get(f"{self.base_url}/system_stats", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def upload_image(self, image_path: str, subfolder: str = "") -> Optional[str]:
        """Upload image to ComfyUI input folder"""
        try:
            path = Path(image_path)
            if not path.exists():
                logger.error(f"Image not found: {image_path}")
                return None

            with open(path, "rb") as f:
                files = {"image": (path.name, f, "image/png")}
                data = {"subfolder": subfolder, "overwrite": "true"}
                resp = requests.post(
                    f"{self.base_url}/upload/image", files=files, data=data
                )

            if resp.status_code == 200:
                result = resp.json()
                logger.info(f"📤 Image uploaded: {result.get('name')}")
                return result.get("name")
            else:
                logger.error(f"Upload failed: {resp.status_code} - {resp.text}")
                return None
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return None

    def upload_image_from_bytes(
        self, image_bytes: bytes, filename: str = "input_image.png"
    ) -> Optional[str]:
        """Upload image from bytes to ComfyUI"""
        try:
            files = {"image": (filename, io.BytesIO(image_bytes), "image/png")}
            data = {"subfolder": "", "overwrite": "true"}
            resp = requests.post(
                f"{self.base_url}/upload/image", files=files, data=data
            )

            if resp.status_code == 200:
                result = resp.json()
                logger.info(f"📤 Image uploaded from bytes: {result.get('name')}")
                return result.get("name")
            else:
                logger.error(f"Upload failed: {resp.status_code}")
                return None
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return None

    def upload_video(self, video_path: str, subfolder: str = "") -> Optional[str]:
        """Upload video to ComfyUI input folder.

        ComfyUI's /upload/image endpoint accepts any file type,
        despite the 'image' field name.
        """
        try:
            path = Path(video_path)
            if not path.exists():
                logger.error(f"🎬 Video not found: {video_path}")
                return None

            # Determine content type based on extension
            ext = path.suffix.lower()
            content_types = {
                ".mp4": "video/mp4",
                ".webm": "video/webm",
                ".mov": "video/quicktime",
                ".avi": "video/x-msvideo",
                ".mkv": "video/x-matroska",
                ".gif": "image/gif",
            }
            content_type = content_types.get(ext, "application/octet-stream")

            with open(path, "rb") as f:
                # ComfyUI uses 'image' field name but accepts any file
                files = {"image": (path.name, f, content_type)}
                data = {"subfolder": subfolder, "overwrite": "true"}
                resp = requests.post(
                    f"{self.base_url}/upload/image", files=files, data=data
                )

            if resp.status_code == 200:
                result = resp.json()
                logger.info(f"🎬 Video uploaded: {result.get('name')}")
                return result.get("name")
            else:
                logger.error(
                    f"🎬 Video upload failed: {resp.status_code} - {resp.text}"
                )
                return None
        except Exception as e:
            logger.error(f"🎬 Video upload error: {e}")
            return None

    def upload_video_from_bytes(
        self, video_bytes: bytes, filename: str = "input_video.mp4"
    ) -> Optional[str]:
        """Upload video from bytes to ComfyUI"""
        try:
            # Determine content type based on extension
            ext = Path(filename).suffix.lower()
            content_types = {
                ".mp4": "video/mp4",
                ".webm": "video/webm",
                ".mov": "video/quicktime",
                ".avi": "video/x-msvideo",
                ".mkv": "video/x-matroska",
                ".gif": "image/gif",
            }
            content_type = content_types.get(ext, "video/mp4")

            files = {"image": (filename, io.BytesIO(video_bytes), content_type)}
            data = {"subfolder": "", "overwrite": "true"}
            resp = requests.post(
                f"{self.base_url}/upload/image", files=files, data=data
            )

            if resp.status_code == 200:
                result = resp.json()
                logger.info(f"🎬 Video uploaded from bytes: {result.get('name')}")
                return result.get("name")
            else:
                logger.error(f"🎬 Video upload failed: {resp.status_code}")
                return None
        except Exception as e:
            logger.error(f"🎬 Video upload error: {e}")
            return None

    def get_resolution_dimensions(
        self, resolution: str, aspect_ratio: str
    ) -> Tuple[int, int]:
        """Calculate width/height from resolution and aspect ratio"""
        # Base heights for each resolution
        base_heights = {"480p": 480, "576p": 576, "720p": 720, "1080p": 1080}

        # Aspect ratio multipliers
        aspect_ratios = {
            "16:9": (16, 9),
            "9:16": (9, 16),
            "1:1": (1, 1),
            "4:3": (4, 3),
            "3:4": (3, 4),
            "21:9": (21, 9),
            "auto": (1, 1),  # Default to square
        }

        height = base_heights.get(resolution, 480)
        ar_w, ar_h = aspect_ratios.get(aspect_ratio, (1, 1))

        # Calculate width based on aspect ratio
        if ar_w >= ar_h:
            # Landscape or square
            width = int(height * ar_w / ar_h)
        else:
            # Portrait - use width as base
            width = height
            height = int(width * ar_h / ar_w)

        # Ensure dimensions are multiples of 8 for VAE
        width = (width // 8) * 8
        height = (height // 8) * 8

        return width, height

    def build_api_workflow(
        self,
        image_name: str,
        prompt: str,
        width: int = 480,
        height: int = 480,
        num_frames: int = 41,
        fps: int = 16,
        steps: int = 6,
        cfg: float = 5.0,
        seed: int = -1,
        output_prefix: str = "oelala_wan22",
        t2i_checkpoint_name: Optional[str] = None,
        t2i_prompt: Optional[str] = None,
        t2i_negative_prompt: str = "",
        t2i_steps: int = 20,
        t2i_cfg: float = 6.0,
        t2i_seed: int = -1,
        t2i_sampler_name: str = "euler",
        t2i_scheduler: str = "normal",
    ) -> Dict[str, Any]:
        """Build ComfyUI API-format workflow with custom parameters"""
        workflow = copy.deepcopy(WAN22_I2V_Q5_API_WORKFLOW)

        # Wan2.2 requires num_frames in format 4k+1 (5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, ...)
        k = round((num_frames - 1) / 4)
        k = max(1, k)  # Minimum k=1 gives 5 frames
        num_frames = 4 * k + 1

        # Node 1: LoadImage
        workflow["1"]["inputs"]["image"] = image_name

        # Node 3: Prompt
        workflow["3"]["inputs"]["positive_prompt"] = prompt

        # Node 9: I2V Encode (resolution + frames)
        workflow["9"]["inputs"]["width"] = width
        workflow["9"]["inputs"]["height"] = height
        workflow["9"]["inputs"]["num_frames"] = num_frames

        # Node 10: Sampler (steps, cfg, seed)
        workflow["10"]["inputs"]["steps"] = steps
        workflow["10"]["inputs"]["cfg"] = cfg
        workflow["10"]["inputs"]["seed"] = seed if seed >= 0 else 42

        # Node 12: Video output (fps, prefix)
        workflow["12"]["inputs"]["frame_rate"] = fps
        workflow["12"]["inputs"]["filename_prefix"] = output_prefix

        # Optional: build start image from a ComfyUI checkpoint (T2I) and feed into Wan I2V
        if t2i_checkpoint_name:
            image_prompt = (t2i_prompt or "").strip() or prompt

            seed_for_image = t2i_seed if t2i_seed >= 0 else (seed if seed >= 0 else 42)

            # Standard ComfyUI T2I graph
            workflow.update(
                {
                    "101": {
                        "class_type": "CheckpointLoaderSimple",
                        "inputs": {"ckpt_name": t2i_checkpoint_name},
                    },
                    "102": {
                        "class_type": "CLIPTextEncode",
                        "inputs": {"text": image_prompt, "clip": ["101", 1]},
                    },
                    "103": {
                        "class_type": "CLIPTextEncode",
                        "inputs": {
                            "text": t2i_negative_prompt or "",
                            "clip": ["101", 1],
                        },
                    },
                    "104": {
                        "class_type": "EmptyLatentImage",
                        "inputs": {"width": width, "height": height, "batch_size": 1},
                    },
                    "105": {
                        "class_type": "KSampler",
                        "inputs": {
                            "seed": seed_for_image,
                            "steps": max(1, int(t2i_steps)),
                            "cfg": float(t2i_cfg),
                            "sampler_name": t2i_sampler_name,
                            "scheduler": t2i_scheduler,
                            "denoise": 1.0,
                            "model": ["101", 0],
                            "positive": ["102", 0],
                            "negative": ["103", 0],
                            "latent_image": ["104", 0],
                        },
                    },
                    "106": {
                        "class_type": "VAEDecode",
                        "inputs": {"samples": ["105", 0], "vae": ["101", 2]},
                    },
                }
            )

            # Feed generated image into Wan I2V
            workflow["6"]["inputs"]["image_1"] = ["106", 0]
            workflow["9"]["inputs"]["start_image"] = ["106", 0]
            logger.info(f"🧩 Using checkpoint start image: {t2i_checkpoint_name}")

        logger.info(
            f"🔧 Built workflow: {width}x{height}, {num_frames}f, {steps} steps, cfg={cfg}"
        )
        return workflow

    def build_t2v_workflow(
        self,
        prompt: str,
        width: int = 480,
        height: int = 480,
        num_frames: int = 41,
        fps: int = 16,
        steps: int = 6,
        cfg: float = 1.0,
        seed: int = -1,
        output_prefix: str = "oelala_t2v",
        t2i_checkpoint_name: str = "dreamshaperXL_lightningDPMSDE.safetensors",
        t2i_steps: int = 20,
        t2i_cfg: float = 6.0,
    ) -> Dict[str, Any]:
        """
        Build Text-to-Video workflow using T2I + I2V pipeline.
        First generates an image from the prompt, then animates it.
        """
        # Use a placeholder image name - the T2I output will override it
        return self.build_api_workflow(
            image_name="placeholder.png",  # Will be overridden by T2I
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            steps=steps,
            cfg=cfg,
            seed=seed,
            output_prefix=output_prefix,
            t2i_checkpoint_name=t2i_checkpoint_name,
            t2i_prompt=prompt,
            t2i_negative_prompt="blurry, low quality, distorted",
            t2i_steps=t2i_steps,
            t2i_cfg=t2i_cfg,
            t2i_seed=seed,
            t2i_sampler_name="euler",
            t2i_scheduler="normal",
        )

    def build_enhanced_workflow(
        self,
        image_name: str,
        prompt: str,
        width: int = 480,
        height: int = 480,
        num_frames: int = 41,
        fps: int = 16,
        steps: int = 4,
        cfg: float = 1.0,
        seed: int = -1,
        output_prefix: str = "oelala_wan22enh",
        model_variant: str = "HIGH",
    ) -> Dict[str, Any]:
        """
        Build ComfyUI API-format workflow for WAN 2.2 Enhanced NSFW FAST MOVE V2 Q4KM.

        Lightning defaults: steps=4 (2+2 internally), cfg=1, scheduler=simple.
        """
        workflow = copy.deepcopy(WAN22_ENHANCED_Q4KM_API_WORKFLOW)

        # Wan2.2 requires num_frames in format 4k+1
        k = round((num_frames - 1) / 4)
        k = max(1, k)
        num_frames = 4 * k + 1

        # Select model variant
        model_map = {
            "HIGH": "wan22_nsfw_fastmove_v2_Q4KM_HIGH.gguf",
            "LOW": "wan22_nsfw_fastmove_v2_Q4KM_LOW.gguf",
        }
        workflow["8"]["inputs"]["model"] = model_map.get(
            model_variant.upper(), model_map["HIGH"]
        )

        # Node 1: LoadImage
        workflow["1"]["inputs"]["image"] = image_name

        # Node 3: Prompt
        workflow["3"]["inputs"]["positive_prompt"] = prompt

        # Node 9: I2V Encode (resolution + frames)
        workflow["9"]["inputs"]["width"] = width
        workflow["9"]["inputs"]["height"] = height
        workflow["9"]["inputs"]["num_frames"] = num_frames

        # Node 10: Sampler (Lightning settings)
        workflow["10"]["inputs"]["steps"] = steps
        workflow["10"]["inputs"]["cfg"] = cfg
        workflow["10"]["inputs"]["seed"] = seed if seed >= 0 else 42

        # Node 12: Video output
        workflow["12"]["inputs"]["frame_rate"] = fps
        workflow["12"]["inputs"]["filename_prefix"] = output_prefix

        logger.info(
            f"🔧 Built Enhanced workflow: {width}x{height}, {num_frames}f, {steps} steps, cfg={cfg}, variant={model_variant}"
        )
        return workflow

    def build_q6_workflow(
        self,
        image_name: str,
        prompt: str,
        negative_prompt: str = "low quality, blurry, out of focus, unstable camera, artifacts, distortion, low resolution, overexposed, underexposed, color banding, missing details, unrealistic lighting, flickering shadows, frame stutter, ghosting, bad reflections, unrealistic motion, pixelated textures, wrong physics, broken animation, rendering artifacts, compression noise, jitter, visual glitches",
        width: int = 480,
        height: int = 480,
        num_frames: int = 41,
        fps: int = 16,
        steps: int = 6,
        cfg: float = 1.0,
        seed: int = -1,
        output_prefix: str = "oelala_distorch2",
        high_noise_steps: int = 3,
        sampler_name: str = "uni_pc",
        scheduler: str = "normal",
        aspect_ratio: str = "1:1",
        long_edge: int = 480,
        unet_high_noise: str = "wan2.2_i2v_high_noise_14B_Q6_K.gguf",
        unet_low_noise: str = "wan2.2_i2v_low_noise_14B_Q6_K.gguf",
        lora_configs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Build ComfyUI API-format workflow for WAN 2.2 DisTorch2 Dual-Pass.

        This is the memory-efficient workflow with expert_mode_allocations.
        Uses dual-pass sampling: High Noise model (steps 0-3) → Low Noise model (steps 3+)

        Args:
            high_noise_steps: Steps where high noise model switches to low noise (default 3)
            sampler_name: "uni_pc", "euler", "dpmpp_2m", etc.
            scheduler: "normal", "simple", "karras", etc.
            aspect_ratio: "1:1", "9:16", "16:9", etc. for AspectRatioResolution_Warper
            long_edge: Resolution for the long edge (480, 720, 1080)
            unet_high_noise: GGUF model for high noise pass
            unet_low_noise: GGUF model for low noise pass
            lora_configs: List of LoRA configs [{high, low, strength}, ...] for stacking
        """
        workflow = copy.deepcopy(WAN22_I2V_Q6_API_WORKFLOW)

        # Wan2.2 requires num_frames in format 4k+1
        k = round((num_frames - 1) / 4)
        k = max(1, k)
        num_frames = 4 * k + 1

        # Node 1 & 2: Unet Loaders - set model names
        workflow["1"]["inputs"]["unet_name"] = unet_high_noise
        workflow["2"]["inputs"]["unet_name"] = unet_low_noise
        logger.info(f"🔧 Unet High: {unet_high_noise}, Unet Low: {unet_low_noise}")

        # Node 11: LoadImage
        workflow["11"]["inputs"]["image"] = image_name

        # Node 9: Positive Prompt
        workflow["9"]["inputs"]["text"] = prompt

        # Node 10: Negative Prompt
        workflow["10"]["inputs"]["text"] = negative_prompt

        # Node 19: AspectRatioResolution_Warper
        # Calculate width/height from aspect ratio and long_edge
        # Aspect ratio parsing
        aspect_ratios = {
            "1:1": (1, 1),
            "9:16": (9, 16),
            "16:9": (16, 9),
            "4:3": (4, 3),
            "3:4": (3, 4),
            "3:2": (3, 2),
            "2:3": (2, 3),
            "21:9": (21, 9),
            "9:21": (9, 21),
        }
        ar_w, ar_h = aspect_ratios.get(aspect_ratio, (9, 16))

        # Calculate dimensions: long_edge is for the SHORT side (the 'p' in 480p/720p refers to height in landscape)
        # For 480p 9:16 portrait: width=480, height=480*16/9=853
        # For 480p 16:9 landscape: width=480*16/9=853, height=480
        if ar_w >= ar_h:
            # Landscape or square - height is the base (short side)
            height = long_edge
            width = int(long_edge * ar_w / ar_h)
        else:
            # Portrait - width is the base (short side)
            width = long_edge
            height = int(long_edge * ar_h / ar_w)

        # Ensure dimensions are multiples of 8 for VAE
        width = (width // 8) * 8
        height = (height // 8) * 8

        # Node 20: ImageResize+ - resize input image to exact target dimensions
        workflow["20"]["inputs"]["width"] = width
        workflow["20"]["inputs"]["height"] = height
        logger.info(f"🖼️ Image resize: {width}x{height} (fill/crop to exact size)")

        # Node 12: WanImageToVideo - set direct width/height values
        workflow["12"]["inputs"]["width"] = width
        workflow["12"]["inputs"]["height"] = height
        workflow["12"]["inputs"]["length"] = num_frames
        logger.info(
            f"📐 Resolution: {width}x{height} (aspect {aspect_ratio}, long_edge {long_edge})"
        )

        # Handle LoRA loading - supports multiple stacked LoRAs with individual strengths
        # For multi-LoRA, we chain LoraLoaderModelOnly nodes
        if lora_configs and len(lora_configs) > 0:
            # Find LoRAs for high noise and low noise models
            high_loras = []
            low_loras = []
            for config in lora_configs:
                if config.get("high"):
                    high_loras.append(
                        {
                            "name": config["high"],
                            "strength": config.get("strength", 1.0),
                        }
                    )
                if config.get("low"):
                    low_loras.append(
                        {"name": config["low"], "strength": config.get("strength", 1.0)}
                    )
                elif config.get("high"):  # If no low specified, use high for both
                    low_loras.append(
                        {
                            "name": config["high"],
                            "strength": config.get("strength", 1.0),
                        }
                    )

            # Apply high noise LoRAs (chain from node 7's output)
            if high_loras:
                current_high_model = ["7", 0]  # Start from SageAttn output
                high_node_id = 170  # Start numbering at 170 for high noise LoRAs
                for i, lora in enumerate(high_loras):
                    node_id = str(high_node_id + i)
                    workflow[node_id] = {
                        "class_type": "LoraLoaderModelOnly",
                        "inputs": {
                            "lora_name": lora["name"],
                            "strength_model": lora["strength"],
                            "model": current_high_model,
                        },
                    }
                    current_high_model = [node_id, 0]
                    logger.info(
                        f"🎨 High Noise LoRA #{i + 1}: {lora['name']} @ {lora['strength']}"
                    )
                # Connect sampler to last LoRA in chain
                workflow["13"]["inputs"]["model"] = current_high_model
                # Remove unused node 17
                if "17" in workflow:
                    del workflow["17"]
            else:
                # No high noise LoRAs - bypass to SageAttn directly
                workflow["13"]["inputs"]["model"] = ["7", 0]
                if "17" in workflow:
                    del workflow["17"]

            # Apply low noise LoRAs (chain from node 8's output)
            if low_loras:
                current_low_model = ["8", 0]  # Start from SageAttn output
                low_node_id = 180  # Start numbering at 180 for low noise LoRAs
                for i, lora in enumerate(low_loras):
                    node_id = str(low_node_id + i)
                    workflow[node_id] = {
                        "class_type": "LoraLoaderModelOnly",
                        "inputs": {
                            "lora_name": lora["name"],
                            "strength_model": lora["strength"],
                            "model": current_low_model,
                        },
                    }
                    current_low_model = [node_id, 0]
                    logger.info(
                        f"🎨 Low Noise LoRA #{i + 1}: {lora['name']} @ {lora['strength']}"
                    )
                # Connect sampler to last LoRA in chain
                workflow["14"]["inputs"]["model"] = current_low_model
                # Remove unused node 18
                if "18" in workflow:
                    del workflow["18"]
            else:
                # No low noise LoRAs - bypass to SageAttn directly
                workflow["14"]["inputs"]["model"] = ["8", 0]
                if "18" in workflow:
                    del workflow["18"]
        else:
            # No LoRAs - bypass LoRA nodes, connect samplers directly to SageAttn
            workflow["13"]["inputs"]["model"] = ["7", 0]
            workflow["14"]["inputs"]["model"] = ["8", 0]
            # Remove unused LoRA nodes
            if "17" in workflow:
                del workflow["17"]
            if "18" in workflow:
                del workflow["18"]

        # Node 13: Sampler 1 (High Noise) - steps 0 to high_noise_steps
        workflow["13"]["inputs"]["noise_seed"] = seed if seed >= 0 else 42
        workflow["13"]["inputs"]["steps"] = steps
        workflow["13"]["inputs"]["cfg"] = cfg
        workflow["13"]["inputs"]["sampler_name"] = sampler_name
        workflow["13"]["inputs"]["scheduler"] = scheduler
        workflow["13"]["inputs"]["end_at_step"] = high_noise_steps

        # Node 14: Sampler 2 (Low Noise) - steps high_noise_steps to end
        workflow["14"]["inputs"]["steps"] = steps
        workflow["14"]["inputs"]["cfg"] = cfg
        workflow["14"]["inputs"]["sampler_name"] = sampler_name
        workflow["14"]["inputs"]["scheduler"] = scheduler
        workflow["14"]["inputs"]["start_at_step"] = high_noise_steps

        # Node 16: Video output
        workflow["16"]["inputs"]["frame_rate"] = fps
        workflow["16"]["inputs"]["filename_prefix"] = output_prefix

        lora_info = ""
        if lora_configs and len(lora_configs) > 0:
            lora_count = len(lora_configs)
            lora_info = f", {lora_count} LoRA{'s' if lora_count > 1 else ''}"
        logger.info(
            f"🔧 Built DisTorch2 workflow: {aspect_ratio}@{long_edge}, {num_frames}f, {steps} steps (switch@{high_noise_steps}), cfg={cfg}{lora_info}"
        )
        return workflow

    def build_workflow(
        self,
        image_name: str,
        prompt: str,
        width: int = 480,
        height: int = 480,
        num_frames: int = 41,
        fps: int = 16,
        steps: int = 6,
        cfg: float = 5.0,
        seed: int = -1,
        output_prefix: str = "oelala_wan22",
        t2i_checkpoint_name: Optional[str] = None,
        t2i_prompt: Optional[str] = None,
        t2i_negative_prompt: str = "",
        t2i_steps: int = 20,
        t2i_cfg: float = 6.0,
        t2i_seed: int = -1,
        t2i_sampler_name: str = "euler",
        t2i_scheduler: str = "normal",
    ) -> Dict[str, Any]:
        """Build ComfyUI workflow - now uses API format directly"""
        return self.build_api_workflow(
            image_name=image_name,
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            steps=steps,
            cfg=cfg,
            seed=seed,
            output_prefix=output_prefix,
            t2i_checkpoint_name=t2i_checkpoint_name,
            t2i_prompt=t2i_prompt,
            t2i_negative_prompt=t2i_negative_prompt,
            t2i_steps=t2i_steps,
            t2i_cfg=t2i_cfg,
            t2i_seed=t2i_seed,
            t2i_sampler_name=t2i_sampler_name,
            t2i_scheduler=t2i_scheduler,
        )

    def queue_prompt(self, workflow: Dict[str, Any]) -> Optional[str]:
        """Queue workflow for execution, return prompt_id.

        Unloads the Guardian LLM from VRAM before queuing so ComfyUI has
        the full 28 GB available. Guardian auto-reloads on the next
        inference request.
        """
        # Free LLM VRAM before generation (non-fatal if Guardian is down)
        get_guardian().unload_sync()

        try:
            # Check if already in API format (has string keys like "1", "2")
            if (
                isinstance(list(workflow.keys())[0], str)
                and list(workflow.keys())[0].isdigit()
            ):
                api_workflow = workflow  # Already API format
            else:
                # Legacy: convert from node format
                api_workflow = self._convert_to_api_format(workflow)

            payload = {"prompt": api_workflow, "client_id": self.client_id}

            resp = requests.post(f"{self.base_url}/prompt", json=payload)

            if resp.status_code == 200:
                result = resp.json()
                prompt_id = result.get("prompt_id")
                logger.info(f"📋 Workflow queued: {prompt_id}")
                return prompt_id
            else:
                logger.error(f"Queue failed: {resp.status_code} - {resp.text}")
                return None
        except Exception as e:
            logger.error(f"Queue error: {e}")
            return None

    def queue_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Queue workflow for execution, return dict with prompt_id and status.

        This is a wrapper around queue_prompt that returns a dict suitable
        for the API response.
        """
        prompt_id = self.queue_prompt(workflow)

        if prompt_id:
            return {"success": True, "prompt_id": prompt_id, "status": "queued"}
        else:
            return {
                "success": False,
                "prompt_id": None,
                "status": "failed",
                "error": "Failed to queue workflow to ComfyUI",
            }

    def _convert_to_api_format(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Convert node-based workflow to API format"""
        api_format = {}

        for node in workflow.get("nodes", []):
            node_id = str(node["id"])

            api_node = {"class_type": node["type"], "inputs": {}}

            # Add widget values as inputs
            widgets = node.get("widgets_values", [])
            node_type = node["type"]

            # Map widget values to input names based on node type
            if node_type == "LoadImage":
                if widgets:
                    api_node["inputs"]["image"] = (
                        widgets[0] if isinstance(widgets[0], str) else "input_image.png"
                    )

            elif node_type == "LoadWanVideoT5TextEncoderMultiGPU":
                if len(widgets) >= 3:
                    api_node["inputs"]["t5"] = widgets[0]
                    api_node["inputs"]["dtype"] = widgets[1]
                    api_node["inputs"]["device"] = widgets[2]

            elif node_type == "WanVideoTextEncodeMultiGPU":
                if len(widgets) >= 3:
                    api_node["inputs"]["prompt"] = widgets[0]
                    api_node["inputs"]["negative_prompt"] = widgets[1]
                    api_node["inputs"]["force_offload"] = widgets[2]

            elif node_type == "WanVideoVAELoaderMultiGPU":
                if len(widgets) >= 3:
                    api_node["inputs"]["vae"] = widgets[0]
                    api_node["inputs"]["device"] = widgets[1]
                    api_node["inputs"]["dtype"] = widgets[2]

            elif node_type == "CLIPVisionLoader":
                if widgets:
                    api_node["inputs"]["clip_name"] = widgets[0]

            elif node_type == "WanVideoClipVisionEncode":
                if len(widgets) >= 5:
                    api_node["inputs"]["strength_1"] = widgets[0]
                    api_node["inputs"]["strength_2"] = widgets[1]
                    api_node["inputs"]["crop"] = widgets[2]
                    api_node["inputs"]["resize_mode"] = widgets[3]
                    api_node["inputs"]["force_offload"] = widgets[4]

            elif node_type == "WanVideoBlockSwapMultiGPU":
                if len(widgets) >= 4:
                    api_node["inputs"]["blocks_to_swap"] = widgets[0]
                    api_node["inputs"]["offload_txt_in"] = widgets[1]
                    api_node["inputs"]["offload_img_in"] = widgets[2]
                    api_node["inputs"]["offload_device"] = widgets[3]

            elif node_type == "WanVideoModelLoaderMultiGPU":
                if len(widgets) >= 7:
                    api_node["inputs"]["model"] = widgets[0]
                    api_node["inputs"]["base_precision"] = widgets[1]
                    api_node["inputs"]["quantization"] = widgets[2]
                    api_node["inputs"]["load_device"] = widgets[3]
                    api_node["inputs"]["compute_device"] = widgets[4]
                    api_node["inputs"]["attention"] = widgets[5]
                    api_node["inputs"]["blocks_to_swap"] = (
                        widgets[6] if len(widgets) > 6 else "default"
                    )

            elif node_type == "WanVideoImageToVideoEncodeMultiGPU":
                if len(widgets) >= 7:
                    api_node["inputs"]["width"] = widgets[0]
                    api_node["inputs"]["height"] = widgets[1]
                    api_node["inputs"]["num_frames"] = widgets[2]
                    api_node["inputs"]["sample_start_frame_percent"] = widgets[3]
                    api_node["inputs"]["sample_end_frame_percent"] = widgets[4]
                    api_node["inputs"]["strength"] = widgets[5]
                    api_node["inputs"]["force_offload"] = widgets[6]

            elif node_type == "WanVideoSamplerMultiGPU":
                if len(widgets) >= 14:
                    api_node["inputs"]["steps"] = widgets[0]
                    api_node["inputs"]["cfg"] = widgets[1]
                    api_node["inputs"]["shift"] = widgets[2]
                    api_node["inputs"]["seed"] = widgets[3]
                    api_node["inputs"]["seed_mode"] = widgets[4]
                    api_node["inputs"]["force_offload"] = widgets[5]
                    api_node["inputs"]["scheduler"] = widgets[6]

            elif node_type == "WanVideoDecodeMultiGPU":
                if len(widgets) >= 5:
                    api_node["inputs"]["enable_vae_tiling"] = widgets[0]
                    api_node["inputs"]["tile_sample_min_height"] = widgets[1]
                    api_node["inputs"]["tile_sample_min_width"] = widgets[2]
                    api_node["inputs"]["tile_overlap_factor_height"] = widgets[3]
                    api_node["inputs"]["tile_overlap_factor_width"] = widgets[4]

            elif node_type == "VHS_VideoCombine":
                if isinstance(widgets, dict):
                    api_node["inputs"]["frame_rate"] = widgets.get("frame_rate", 16)
                    api_node["inputs"]["loop_count"] = widgets.get("loop_count", 0)
                    api_node["inputs"]["filename_prefix"] = widgets.get(
                        "filename_prefix", "oelala"
                    )
                    api_node["inputs"]["format"] = widgets.get(
                        "format", "video/h264-mp4"
                    )
                    api_node["inputs"]["pingpong"] = widgets.get("pingpong", False)
                    api_node["inputs"]["save_output"] = widgets.get("save_output", True)

            # Add linked inputs from other nodes
            for inp in node.get("inputs", []):
                if inp.get("link") is not None:
                    # Find source node for this link
                    for link in workflow.get("links", []):
                        if link[0] == inp["link"]:
                            source_node_id = str(link[1])
                            source_slot = link[2]
                            api_node["inputs"][inp["name"]] = [
                                source_node_id,
                                source_slot,
                            ]
                            break

            api_format[node_id] = api_node

        return api_format

    def register_job(
        self,
        prompt_id: str,
        user_id: str,
        prompt: str = "",
        settings: Optional[Dict[str, Any]] = None,
    ):
        """Register job metadata for tracking and auto-upload on completion.

        Args:
            prompt_id: ComfyUI prompt ID
            user_id: User ID from JWT token
            prompt: Generation prompt
            settings: Additional settings (resolution, frames, etc.)
        """
        with self._metadata_lock:
            self.job_metadata[prompt_id] = {
                "user_id": user_id,
                "prompt": prompt,
                "settings": (settings or {}).copy(),
                "started_at": datetime.now().isoformat(),
            }
        logger.info(f"📝 Registered job {prompt_id} for user {user_id}")

    def get_job_metadata(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get job metadata for a prompt ID.

        Returns a deep copy to prevent external mutation of internal state,
        including nested structures such as the `settings` dictionary.
        """
        with self._metadata_lock:
            metadata = self.job_metadata.get(prompt_id)
            return copy.deepcopy(metadata) if metadata is not None else None

    def clear_job_metadata(self, prompt_id: str):
        """Clear job metadata after completion."""
        with self._metadata_lock:
            self.job_metadata.pop(prompt_id, None)

    def on_job_complete(
        self,
        prompt_id: str,
        output_path: str,
        output_type: str = "video",
    ) -> Optional[str]:
        """Auto-upload generated content to user storage on job completion.

        Args:
            prompt_id: ComfyUI prompt ID
            output_path: Local path to generated file
            output_type: Type of output ('video', 'image', 'audio')

        Returns:
            Storage path if upload succeeded in the format "{media_type}/{filename}",
            for example "images/1234567890_output.png", or None if the upload failed.
        """
        # Get job metadata
        metadata = self.get_job_metadata(prompt_id)
        if not metadata:
            logger.warning(
                f"⚠️ No job metadata found for {prompt_id}, skipping auto-upload"
            )
            return None

        user_id = metadata.get("user_id")
        if not user_id:
            logger.warning(f"⚠️ No user_id in job metadata for {prompt_id}")
            return None

        file_data = None
        try:
            # Read file content
            try:
                with open(output_path, "rb") as f:
                    file_data = f.read()
            except (IOError, OSError) as e:
                logger.error(f"❌ Failed to read file {output_path}: {e}")
                # Clear metadata on file read failure to prevent memory leak
                self.clear_job_metadata(prompt_id)
                return None

            # Generate storage filename with high-precision timestamp (ms) to avoid collisions
            timestamp = int(datetime.now().timestamp() * 1000)
            original_filename = Path(output_path).name
            storage_filename = f"{timestamp}_{original_filename}"

            # Determine media type folder
            media_type_map = {
                "video": "videos",
                "image": "images",
                "audio": "audio",
            }
            media_type = media_type_map.get(output_type, "generated")

            # Determine content type
            ext = Path(output_path).suffix.lower()
            content_type_map = {
                ".mp4": "video/mp4",
                ".webm": "video/webm",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".mp3": "audio/mpeg",
                ".wav": "audio/wav",
            }
            content_type = content_type_map.get(ext, "application/octet-stream")

            # Upload to user storage
            storage_client = get_storage_client()
            logger.info(
                f"📤 Uploading {output_type} to user storage: {user_id}/{media_type}/{storage_filename}"
            )

            try:
                storage_client.put_user_media(
                    user_id=user_id,
                    media_type=media_type,
                    filename=storage_filename,
                    data=file_data,
                    content_type=content_type,
                )
            except Exception as e:
                logger.error(f"❌ Storage upload failed for {prompt_id}: {e}")
                # Don't clear metadata on upload failure - allows retry
                return None

            storage_path = f"{media_type}/{storage_filename}"
            logger.info(
                f"✅ Auto-uploaded to storage: {storage_path} ({len(file_data)} bytes)"
            )

            # Clear job metadata after successful upload
            self.clear_job_metadata(prompt_id)

            return storage_path

        except Exception as e:
            logger.error(f"❌ Unexpected error during auto-upload for {prompt_id}: {e}")
            # Don't raise - we don't want to break the user flow if upload fails
            return None

    async def on_job_complete_async(
        self,
        prompt_id: str,
        output_path: str,
        output_type: str = "video",
    ) -> Optional[str]:
        """
        Async version of on_job_complete that uses MediaService for upload + Supabase sync.

        This should be called from async contexts (like FastAPI endpoints) for better
        integration with Supabase metadata tracking and signed URL generation.

        Args:
            prompt_id: ComfyUI prompt ID
            output_path: Local path to generated file
            output_type: Type of output ('video', 'image', 'audio')

        Returns:
            Full storage path (e.g., users/{user_id}/videos/file.mp4) or None if failed
        """
        if get_media_service is None:
            logger.warning("⚠️ MediaService not available, falling back to sync upload")
            return self.on_job_complete(prompt_id, output_path, output_type)

        # Get job metadata
        metadata = self.get_job_metadata(prompt_id)
        if not metadata:
            logger.warning(
                f"⚠️ No job metadata found for {prompt_id}, skipping auto-upload"
            )
            return None

        user_id = metadata.get("user_id")
        if not user_id:
            logger.warning(f"⚠️ No user_id in job metadata for {prompt_id}")
            return None

        try:
            # Read file content
            file_path = Path(output_path)
            if not file_path.exists():
                logger.error(f"❌ Output file not found: {output_path}")
                self.clear_job_metadata(prompt_id)
                return None

            file_data = file_path.read_bytes()

            # Map output_type to generation_type
            gen_type_map = {
                "video": "t2v" if "t2v" in metadata.get("type", "") else "i2v",
                "image": "t2i",
                "audio": "audio",
            }
            generation_type = gen_type_map.get(output_type, output_type)

            # Extract metadata for Supabase
            extra_metadata = {
                "model_name": metadata.get("model_name") or metadata.get("model_type"),
                "resolution": metadata.get("resolution"),
                "aspect_ratio": metadata.get("aspect_ratio"),
                "num_frames": metadata.get("num_frames"),
                "fps": metadata.get("fps"),
                "seed": metadata.get("seed"),
                "steps": metadata.get("steps"),
                "cfg": metadata.get("cfg"),
                "size_bytes": len(file_data),
            }
            # Remove None values
            extra_metadata = {k: v for k, v in extra_metadata.items() if v is not None}

            # Upload via MediaService (storage + Supabase sync)
            media_service = get_media_service()
            record = await media_service.upload(
                user_id=user_id,
                file_data=file_data,
                filename=file_path.name,
                generation_type=generation_type,
                prompt=metadata.get("prompt", ""),
                workflow_id=prompt_id,
                extra_metadata=extra_metadata,
            )

            logger.info(
                f"✅ Async uploaded to storage: {record.storage_path} ({len(file_data)} bytes)"
            )

            # Clear job metadata after successful upload
            self.clear_job_metadata(prompt_id)

            return record.storage_path

        except Exception as e:
            logger.error(f"❌ Async upload failed for {prompt_id}: {e}")
            # Fallback to sync upload
            logger.info("🔄 Falling back to sync upload...")
            return self.on_job_complete(prompt_id, output_path, output_type)

    def wait_for_completion(
        self,
        prompt_id: str,
        timeout: int = 1800,  # 30 minutes for longer generations
        progress_callback=None,
    ) -> Optional[Dict[str, Any]]:
        """Wait for workflow completion using websocket"""
        # Node ID to friendly name mapping for progress display
        NODE_NAMES = {
            "1": "📷 Load Image",
            "2": "🔧 Load GGUF Model",
            "3": "📝 T5 Text Encoder",
            "4": "🎨 VAE Loader",
            "5": "💬 Text Encode",
            "6": "🖼️ Image Encode",
            "7": "🎬 Video Sampler",
            "8": "🔄 VAE Decode",
            "9": "💾 Video Combine",
            "10": "📊 CLIP Vision",
            "11": "🎯 Sampler Stage 2",
            "12": "🎥 Video Output",
        }

        current_node = None
        current_node_name = "Starting..."

        try:
            ws_url = f"ws://{self.host}:{self.port}/ws?clientId={self.client_id}"
            ws = websocket.create_connection(ws_url, timeout=30)

            start_time = time.time()

            while time.time() - start_time < timeout:
                try:
                    message = ws.recv()
                    data = json.loads(message)

                    msg_type = data.get("type")
                    msg_data = data.get("data", {})

                    if msg_type == "progress":
                        value = msg_data.get("value", 0)
                        max_val = msg_data.get("max", 100)
                        pct = int(100 * value / max_val) if max_val > 0 else 0
                        node_id = str(msg_data.get("node", ""))
                        node_name = NODE_NAMES.get(node_id, f"Node {node_id}")
                        logger.info(
                            f"📊 [{node_name}] Progress: {pct}% ({value}/{max_val})"
                        )
                        if progress_callback:
                            # Pass both percentage and process name
                            progress_callback(pct, node_name)

                    elif msg_type == "executing":
                        node_id = msg_data.get("node")
                        if node_id is None and msg_data.get("prompt_id") == prompt_id:
                            logger.info("✅ Workflow execution complete")
                            ws.close()
                            return self._get_history(prompt_id)
                        elif node_id:
                            current_node = str(node_id)
                            current_node_name = NODE_NAMES.get(
                                current_node, f"Node {current_node}"
                            )
                            logger.info(f"🔄 Executing: {current_node_name}")

                    elif msg_type == "execution_error":
                        logger.error(f"❌ Execution error: {msg_data}")
                        ws.close()
                        return None

                except websocket.WebSocketTimeoutException:
                    continue

            ws.close()
            logger.error("⏰ Timeout waiting for completion")
            return None

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            return None

    def _get_history(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get execution history for prompt"""
        try:
            resp = requests.get(f"{self.base_url}/history/{prompt_id}")
            if resp.status_code == 200:
                return resp.json().get(prompt_id, {})
            return None
        except Exception as e:
            logger.error(f"History error: {e}")
            return None

    def get_output_video(
        self, history: Dict[str, Any], output_dir: str, prompt_id: Optional[str] = None
    ) -> Optional[str]:
        """Extract output video path from history and auto-upload to user storage.

        Args:
            history: ComfyUI execution history
            output_dir: Local directory to save video
            prompt_id: Optional prompt ID for auto-upload tracking

        Returns:
            Local path to downloaded video
        """
        try:
            outputs = history.get("outputs", {})

            # Find VHS_VideoCombine node output
            for node_id, node_output in outputs.items():
                if "gifs" in node_output:
                    for gif in node_output["gifs"]:
                        filename = gif.get("filename")
                        subfolder = gif.get("subfolder", "")

                        # Download the video
                        params = {
                            "filename": filename,
                            "subfolder": subfolder,
                            "type": "output",
                        }
                        resp = requests.get(f"{self.base_url}/view", params=params)

                        if resp.status_code == 200:
                            output_path = Path(output_dir) / filename
                            with open(output_path, "wb") as f:
                                f.write(resp.content)
                            logger.info(f"📥 Video downloaded: {output_path}")

                            # Auto-upload to user storage if prompt_id is provided
                            if prompt_id:
                                storage_path = self.on_job_complete(
                                    prompt_id=prompt_id,
                                    output_path=str(output_path),
                                    output_type="video",
                                )
                                if storage_path:
                                    logger.info(
                                        f"📤 Auto-uploaded video to: {storage_path}"
                                    )

                            return str(output_path)

            logger.warning("No video output found in history")
            return None

        except Exception as e:
            logger.error(f"Output extraction error: {e}")
            return None

    def get_output_image(
        self, history: Dict[str, Any], output_dir: str, prompt_id: Optional[str] = None
    ) -> Optional[str]:
        """Extract output image path from history and auto-upload to user storage.

        Args:
            history: ComfyUI execution history
            output_dir: Local directory to save image
            prompt_id: Optional prompt ID for auto-upload tracking

        Returns:
            Local path to downloaded image
        """
        try:
            outputs = history.get("outputs", {})

            # Find SaveImage node output
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    for img in node_output["images"]:
                        filename = img.get("filename")
                        subfolder = img.get("subfolder", "")

                        # Download the image from ComfyUI
                        params = {
                            "filename": filename,
                            "subfolder": subfolder,
                            "type": "output",
                        }
                        resp = requests.get(f"{self.base_url}/view", params=params)

                        if resp.status_code == 200:
                            output_path = Path(output_dir) / filename
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(output_path, "wb") as f:
                                f.write(resp.content)
                            logger.info(f"📥 Image downloaded: {output_path}")

                            # Auto-upload to user storage if prompt_id is provided
                            if prompt_id:
                                storage_path = self.on_job_complete(
                                    prompt_id=prompt_id,
                                    output_path=str(output_path),
                                    output_type="image",
                                )
                                if storage_path:
                                    logger.info(
                                        f"📤 Auto-uploaded image to: {storage_path}"
                                    )

                            return str(output_path)

            logger.warning("No image output found in history")
            return None

        except Exception as e:
            logger.error(f"Image extraction error: {e}")
            return None

    def wait_and_download_image(
        self,
        prompt_id: str,
        output_dir: str,
        timeout: int = 300,
        progress_callback=None,
    ) -> Optional[str]:
        """Wait for workflow completion and download resulting image.

        Args:
            prompt_id: ComfyUI prompt ID
            output_dir: Directory to save downloaded image
            timeout: Timeout in seconds
            progress_callback: Optional callback(percent, node_name)

        Returns:
            Path to downloaded image, or None on failure
        """
        history = self.wait_for_completion(prompt_id, timeout, progress_callback)
        if not history:
            return None
        return self.get_output_image(history, output_dir, prompt_id)

    # ─────────────────────────────────────────────────────────────────────────
    # SDXL Text-to-Image Generation
    # ─────────────────────────────────────────────────────────────────────────

    def generate_sdxl_image(
        self,
        prompt: str,
        output_dir: str,
        negative_prompt: str = "ugly, deformed, blurry, low quality, bad anatomy, watermark, signature, text",
        checkpoint: str = "CyberRealistic_Pony_v14.1_FP16.safetensors",
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        cfg: float = 7.5,
        seed: int = -1,
        sampler_name: str = "dpmpp_2m",
        scheduler: str = "karras",
        lora_configs: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """
        Generate image using SDXL checkpoint via ComfyUI.

        Args:
            prompt: Text prompt for image generation
            output_dir: Directory to save output
            negative_prompt: Negative prompt
            checkpoint: SDXL checkpoint filename
            width, height: Image dimensions
            steps: Number of sampling steps
            cfg: CFG scale
            seed: Random seed (-1 for random)
            sampler_name: Sampler name (dpmpp_2m, euler, etc.)
            scheduler: Scheduler (karras, normal, etc.)
            lora_configs: Optional list of LoRA configs [{name, strength}, ...]

        Returns:
            Path to generated image, or None on failure
        """
        import random

        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        # Build the SDXL workflow
        workflow = {
            "1": {
                "inputs": {"ckpt_name": checkpoint},
                "class_type": "CheckpointLoaderSimple",
                "_meta": {"title": "Load Checkpoint"},
            },
            "2": {
                "inputs": {"text": prompt, "clip": ["9", 1]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt"},
            },
            "3": {
                "inputs": {"text": negative_prompt, "clip": ["9", 1]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Negative Prompt"},
            },
            "4": {
                "inputs": {"width": width, "height": height, "batch_size": 1},
                "class_type": "EmptyLatentImage",
                "_meta": {"title": "Empty Latent Image"},
            },
            "5": {
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "denoise": 1,
                    "model": ["9", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0],
                },
                "class_type": "KSampler",
                "_meta": {"title": "KSampler"},
            },
            "6": {
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
                "class_type": "VAEDecode",
                "_meta": {"title": "VAE Decode"},
            },
            "8": {
                "inputs": {"filename_prefix": "SDXL_T2I", "images": ["6", 0]},
                "class_type": "SaveImage",
                "_meta": {"title": "Save Image"},
            },
            "9": {
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
                    "➕ Add Lora": "",
                    "model": ["1", 0],
                    "clip": ["1", 1],
                },
                "class_type": "Power Lora Loader (rgthree)",
                "_meta": {"title": "Power LoRA Loader"},
            },
        }

        # Apply LoRA configs if provided
        if lora_configs:
            for i, lora_cfg in enumerate(lora_configs[:6], 1):
                if lora_cfg.get("name") and lora_cfg.get("name") != "None":
                    workflow["9"]["inputs"][f"lora_{i}"] = {
                        "on": True,
                        "lora": lora_cfg["name"],
                        "strength": lora_cfg.get("strength", 1.0),
                    }

        logger.info(f"🎨 SDXL T2I: {prompt[:50]}... ({width}x{height}, {checkpoint})")

        # Queue and wait for completion
        prompt_id = self.queue_prompt(workflow)
        if not prompt_id:
            logger.error("Failed to queue SDXL workflow")
            return None

        # Wait for completion with timeout
        output_path = self.wait_and_download_image(prompt_id, output_dir, timeout=300)

        if output_path:
            logger.info(f"✅ SDXL image generated: {output_path}")
        else:
            logger.error("SDXL generation failed or timed out")

        return output_path

    # ─────────────────────────────────────────────────────────────────────────
    # Flux Dev Text-to-Image Generation
    # ─────────────────────────────────────────────────────────────────────────

    def generate_flux_image(
        self,
        prompt: str,
        output_dir: str,
        checkpoint: str = "flux1-dev-fp8.safetensors",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        guidance: float = 3.5,
        seed: int = -1,
        lora_configs: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """
        Generate image using Flux Dev via ComfyUI.
        Note: Flux doesn't use negative prompts or CFG in the traditional sense.

        Args:
            prompt: Text prompt for image generation
            output_dir: Directory to save output
            checkpoint: Flux checkpoint filename
            width, height: Image dimensions
            steps: Number of sampling steps (20 recommended)
            guidance: Flux guidance scale (3.5 recommended)
            seed: Random seed (-1 for random)
            lora_configs: Optional list of LoRA configs [{name, strength}, ...]

        Returns:
            Path to generated image, or None on failure
        """
        import random

        if seed == -1:
            seed = random.randint(0, 2**63 - 1)

        # Build the Flux workflow with Power LoRA Loader
        workflow = {
            "1": {
                "inputs": {"ckpt_name": checkpoint},
                "class_type": "CheckpointLoaderSimple",
                "_meta": {"title": "Load Checkpoint"},
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
                    "➕ Add Lora": "",
                    "model": ["1", 0],
                    "clip": ["1", 1],
                },
                "class_type": "Power Lora Loader (rgthree)",
                "_meta": {"title": "Power LoRA Loader"},
            },
            "3": {
                "inputs": {"text": prompt, "clip": ["2", 1]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt"},
            },
            "4": {
                "inputs": {"text": "", "clip": ["2", 1]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Negative Prompt (empty for Flux)"},
            },
            "5": {
                "inputs": {"guidance": guidance, "conditioning": ["3", 0]},
                "class_type": "FluxGuidance",
                "_meta": {"title": "FluxGuidance"},
            },
            "6": {
                "inputs": {"width": width, "height": height, "batch_size": 1},
                "class_type": "EmptySD3LatentImage",
                "_meta": {"title": "EmptySD3LatentImage"},
            },
            "7": {
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": 1,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1,
                    "model": ["2", 0],
                    "positive": ["5", 0],
                    "negative": ["4", 0],
                    "latent_image": ["6", 0],
                },
                "class_type": "KSampler",
                "_meta": {"title": "KSampler"},
            },
            "8": {
                "inputs": {"samples": ["7", 0], "vae": ["1", 2]},
                "class_type": "VAEDecode",
                "_meta": {"title": "VAE Decode"},
            },
            "9": {
                "inputs": {"filename_prefix": "Flux_T2I", "images": ["8", 0]},
                "class_type": "SaveImage",
                "_meta": {"title": "Save Image"},
            },
        }

        # Apply LoRA configs if provided
        if lora_configs:
            for i, lora_cfg in enumerate(lora_configs[:4], 1):
                if lora_cfg.get("name") and lora_cfg.get("name") != "None":
                    workflow["2"]["inputs"][f"lora_{i}"] = {
                        "on": True,
                        "lora": lora_cfg["name"],
                        "strength": lora_cfg.get("strength", 1.0),
                    }

        logger.info(
            f"⚡ Flux T2I: {prompt[:50]}... ({width}x{height}, guidance={guidance})"
        )

        # Queue and wait for completion
        prompt_id = self.queue_prompt(workflow)
        if not prompt_id:
            logger.error("Failed to queue Flux workflow")
            return None

        # Wait for completion with timeout (Flux is slower)
        output_path = self.wait_and_download_image(prompt_id, output_dir, timeout=600)

        if output_path:
            logger.info(f"✅ Flux image generated: {output_path}")
        else:
            logger.error("Flux generation failed or timed out")

        return output_path

    # ─────────────────────────────────────────────────────────────────────────
    # SD 1.5 Text-to-Image Generation
    # ─────────────────────────────────────────────────────────────────────────

    def generate_sd15_image(
        self,
        prompt: str,
        output_dir: str,
        negative_prompt: str = "(deformed, blurry, bad anatomy, extra fingers, mutated hands, poorly drawn face, low quality:1.4)",
        checkpoint: str = "Realistic_Vision_V5.1.safetensors",
        width: int = 512,
        height: int = 768,
        steps: int = 25,
        cfg: float = 7.0,
        seed: int = -1,
        sampler_name: str = "dpmpp_sde",
        scheduler: str = "karras",
        lora_configs: Optional[List[Dict[str, Any]]] = None,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate image using SD 1.5 checkpoint via ComfyUI.

        Args:
            prompt: Text prompt for image generation
            output_dir: Directory to save output
            negative_prompt: Negative prompt
            checkpoint: SD 1.5 checkpoint filename
            width, height: Image dimensions (512x512 to 768x768 optimal)
            steps: Number of sampling steps
            cfg: CFG scale
            seed: Random seed (-1 for random)
            sampler_name: Sampler name
            scheduler: Scheduler
            lora_configs: Optional list of LoRA configs [{name, strength}, ...]

        Returns:
            Path to generated image, or None on failure
        """
        import random

        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        # Build the SD 1.5 workflow with Power LoRA Loader
        workflow = {
            "1": {
                "inputs": {"ckpt_name": checkpoint},
                "class_type": "CheckpointLoaderSimple",
                "_meta": {"title": "Load Checkpoint"},
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
                    "➕ Add Lora": "",
                    "model": ["1", 0],
                    "clip": ["1", 1],
                },
                "class_type": "Power Lora Loader (rgthree)",
                "_meta": {"title": "Power LoRA Loader"},
            },
            "3": {
                "inputs": {"text": prompt, "clip": ["2", 1]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt"},
            },
            "4": {
                "inputs": {"text": negative_prompt, "clip": ["2", 1]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Negative Prompt"},
            },
            "5": {
                "inputs": {"width": width, "height": height, "batch_size": 1},
                "class_type": "EmptyLatentImage",
                "_meta": {"title": "Empty Latent Image"},
            },
            "6": {
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "denoise": 1,
                    "model": ["2", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0],
                },
                "class_type": "KSampler",
                "_meta": {"title": "KSampler"},
            },
            "7": {
                "inputs": {"samples": ["6", 0], "vae": ["1", 2]},
                "class_type": "VAEDecode",
                "_meta": {"title": "VAE Decode"},
            },
            "8": {
                "inputs": {"filename_prefix": "SD15_T2I", "images": ["7", 0]},
                "class_type": "SaveImage",
                "_meta": {"title": "Save Image"},
            },
        }

        # Apply LoRA configs if provided
        if lora_configs:
            for i, lora_cfg in enumerate(lora_configs[:6], 1):
                if lora_cfg.get("name") and lora_cfg.get("name") != "None":
                    workflow["2"]["inputs"][f"lora_{i}"] = {
                        "on": True,
                        "lora": lora_cfg["name"],
                        "strength": lora_cfg.get("strength", 1.0),
                    }

        logger.info(f"🖼️ SD1.5 T2I: {prompt[:50]}... ({width}x{height}, {checkpoint})")

        # Queue and wait for completion
        prompt_id = self.queue_prompt(workflow)
        if not prompt_id:
            logger.error("Failed to queue SD1.5 workflow")
            return None

        # Register job for auto-upload tracking
        if user_id:
            self.register_job(
                prompt_id=prompt_id,
                user_id=user_id,
                prompt=prompt,
                settings={
                    "checkpoint": checkpoint,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "cfg": cfg,
                    "seed": seed,
                    "sampler": sampler_name,
                    "scheduler": scheduler,
                },
            )

        # Wait for completion with timeout
        output_path = self.wait_and_download_image(prompt_id, output_dir, timeout=180)

        if output_path:
            logger.info(f"✅ SD1.5 image generated: {output_path}")
        else:
            logger.error("SD1.5 generation failed or timed out")

        return output_path

    # ─────────────────────────────────────────────────────────────────────────
    # Wan2.2 Text-to-Image Generation (DisTorch2 Multi-GPU)
    # ─────────────────────────────────────────────────────────────────────────

    def generate_wan22_t2i(
        self,
        prompt: str,
        output_dir: str,
        width: int = 512,
        height: int = 512,
        steps: int = 8,
        seed: int = -1,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate image using Wan2.2 T2V model in T2I mode via ComfyUI.
        Uses DisTorch2 multi-GPU setup with high/low noise models.

        Args:
            prompt: Text prompt for image generation
            output_dir: Directory to save output
            width, height: Image dimensions
            steps: Total number of sampling steps (split between high/low noise)
            seed: Random seed (-1 for random)

        Returns:
            Path to generated image, or None on failure
        """
        import random

        if seed == -1:
            seed = random.randint(0, 2**63 - 1)
        seed2 = random.randint(0, 2**63 - 1)

        half_steps = steps // 2

        # Build the Wan2.2 T2I workflow (DisTorch2 multi-GPU)
        workflow = {
            "3": {
                "inputs": {"text": prompt, "clip": ["29", 1]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt"},
            },
            "4": {
                "inputs": {"text": "", "clip": ["29", 1]},
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
                    "compute_device": "cuda:0",
                    "virtual_vram_gb": 5,
                    "donor_device": "cuda:1",
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
                    "device": "cuda:0",
                },
                "class_type": "CLIPLoaderMultiGPU",
                "_meta": {"title": "CLIPLoaderMultiGPU"},
            },
            "52": {
                "inputs": {
                    "unet_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
                    "weight_dtype": "default",
                    "compute_device": "cuda:0",
                    "virtual_vram_gb": 5,
                    "donor_device": "cuda:1",
                    "expert_mode_allocations": "",
                    "eject_models": True,
                },
                "class_type": "UNETLoaderDisTorch2MultiGPU",
                "_meta": {"title": "UNETLoaderDisTorch2MultiGPU"},
            },
            "55": {
                "inputs": {
                    "vae_name": "wan_2.1_vae.safetensors",
                    "compute_device": "cuda:0",
                    "virtual_vram_gb": 0,
                    "donor_device": "cuda:1",
                    "expert_mode_allocations": "",
                    "eject_models": True,
                },
                "class_type": "VAELoaderDisTorch2MultiGPU",
                "_meta": {"title": "VAELoaderDisTorch2MultiGPU"},
            },
        }

        logger.info(
            f"🎬 Wan2.2 T2I: {prompt[:50]}... ({width}x{height}, {steps} steps)"
        )

        # Queue and wait for completion
        prompt_id = self.queue_prompt(workflow)
        if not prompt_id:
            logger.error("Failed to queue Wan2.2 T2I workflow")
            return None

        # Register job for auto-upload tracking
        if user_id:
            self.register_job(
                prompt_id=prompt_id,
                user_id=user_id,
                prompt=prompt,
                settings={
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "seed": seed,
                },
            )

        # Wait for completion with timeout (Wan2.2 is slower)
        output_path = self.wait_and_download_image(prompt_id, output_dir, timeout=600)

        if output_path:
            logger.info(f"✅ Wan2.2 T2I image generated: {output_path}")
        else:
            logger.error("Wan2.2 T2I generation failed or timed out")

        return output_path

    def generate_video(
        self,
        image_path: Optional[str],
        prompt: str,
        output_dir: str,
        resolution: str = "480p",
        aspect_ratio: str = "1:1",
        num_frames: int = 41,
        fps: int = 16,
        steps: int = 6,
        cfg: float = 1.0,
        seed: int = -1,
        output_prefix: str = "oelala_distorch2",
        progress_callback=None,
        unet_high_noise: str = "wan2.2_i2v_high_noise_14B_Q6_K.gguf",
        unet_low_noise: str = "wan2.2_i2v_low_noise_14B_Q6_K.gguf",
        lora_configs: Optional[List[Dict[str, Any]]] = None,
        generation_mode: str = "standard",  # "standard" or "nsfw_lora"
        # Legacy T2I params (kept for compatibility)
        t2i_checkpoint_name: Optional[str] = None,
        t2i_prompt: Optional[str] = None,
        t2i_negative_prompt: str = "",
        t2i_steps: int = 20,
        t2i_cfg: float = 6.0,
        t2i_seed: int = -1,
        t2i_sampler_name: str = "euler",
        t2i_scheduler: str = "normal",
    ) -> Optional[str]:
        """
        Full pipeline: upload image → build workflow → execute → return video path
        Now uses DisTorch2 dual-pass workflow with optional LoRA support.

        Args:
            generation_mode: Which workflow preset to use ("standard", "nsfw_lora")
        """
        # Delegate to generate_q6_video which has full DisTorch2 support
        return self.generate_q6_video(
            image_path=image_path,
            prompt=prompt,
            output_dir=output_dir,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            num_frames=num_frames,
            fps=fps,
            steps=steps,
            cfg=cfg,
            seed=seed,
            output_prefix=output_prefix,
            unet_high_noise=unet_high_noise,
            unet_low_noise=unet_low_noise,
            lora_configs=lora_configs,
            progress_callback=progress_callback,
            generation_mode=generation_mode,
        )

    def generate_q6_video(
        self,
        image_path: str,
        prompt: str,
        output_dir: str,
        negative_prompt: str = "low quality, blurry, out of focus, unstable camera, artifacts, distortion, low resolution, overexposed, underexposed, color banding, missing details, unrealistic lighting, flickering shadows, frame stutter, ghosting, bad reflections, unrealistic motion, pixelated textures, wrong physics, broken animation, rendering artifacts, compression noise, jitter, visual glitches",
        resolution: str = "480p",
        aspect_ratio: str = "1:1",
        num_frames: int = 41,
        fps: int = 16,
        steps: int = 6,
        cfg: float = 1.0,
        seed: int = -1,
        output_prefix: str = "oelala_distorch2",
        high_noise_steps: int = 3,
        sampler_name: str = "uni_pc",
        scheduler: str = "normal",
        unet_high_noise: str = "wan2.2_i2v_high_noise_14B_Q6_K.gguf",
        unet_low_noise: str = "wan2.2_i2v_low_noise_14B_Q6_K.gguf",
        lora_configs: Optional[List[Dict[str, Any]]] = None,
        progress_callback=None,
        generation_mode: str = "standard",  # "standard" or "nsfw_lora"
    ) -> Optional[str]:
        """
        Full pipeline for WAN 2.2 DisTorch2 Dual-Pass workflow.
        Memory-efficient via expert_mode_allocations, uses CONVERTED T5.

        Args:
            high_noise_steps: Steps where high noise model switches to low noise (default 3)
            sampler_name: "uni_pc", "euler", "dpmpp_2m", etc.
            scheduler: "normal", "simple", "karras"
            unet_high_noise: GGUF model for high noise pass
            unet_low_noise: GGUF model for low noise pass
            lora_configs: List of LoRA configs [{high, low, strength}, ...] for stacking
            generation_mode: Which workflow preset to use ("standard", "nsfw_lora")
        """
        if not self.is_available():
            logger.error("❌ ComfyUI not available")
            return None

        # 1. Upload image
        logger.info(f"📤 Uploading image: {image_path}")
        image_name = self.upload_image(image_path)
        if not image_name:
            return None

        # 2. Map resolution to long_edge for AspectRatioResolution_Warper
        resolution_map = {"480p": 480, "576p": 576, "720p": 720, "1080p": 1080}
        long_edge = resolution_map.get(resolution, 480)
        logger.info(
            f"📐 Resolution: {resolution} ({aspect_ratio}), long_edge={long_edge}"
        )

        # 3. Build DisTorch2 workflow
        lora_info = ""
        if lora_configs and len(lora_configs) > 0:
            lora_info = (
                f", {len(lora_configs)} LoRA{'s' if len(lora_configs) > 1 else ''}"
            )
        logger.info(
            f"🔧 Building DisTorch2 workflow (mode={generation_mode}): {num_frames}f @ {fps}fps, {steps} steps (switch@{high_noise_steps}), cfg={cfg}{lora_info}"
        )

        # For preset modes, load workflow from file and customize
        if generation_mode != "standard" and generation_mode in I2V_GENERATION_MODES:
            mode_config = I2V_GENERATION_MODES[generation_mode]
            loaded_workflow = load_workflow_from_file(mode_config["workflow_file"])
            if loaded_workflow:
                workflow = copy.deepcopy(loaded_workflow)
                logger.info(
                    f"📂 Loaded preset workflow: {mode_config['workflow_file']}"
                )

                # Update dynamic parameters in the loaded workflow
                # Find and update image node
                for node_id, node in workflow.items():
                    if node.get("class_type") == "LoadImage":
                        node["inputs"]["image"] = image_name
                    elif node.get("class_type") in ["CLIPTextEncode"]:
                        # Update prompts - node 7 is usually positive, 8 is negative
                        if node_id in ["7", "9"]:
                            node["inputs"]["text"] = prompt
                        elif node_id in ["8", "10"]:
                            node["inputs"]["text"] = negative_prompt
                    elif node.get("class_type") == "VHS_VideoCombine":
                        node["inputs"]["frame_rate"] = fps
                        node["inputs"]["filename_prefix"] = output_prefix
                    elif node.get("class_type") == "KSamplerAdvanced":
                        node["inputs"]["noise_seed"] = (
                            seed if seed >= 0 else random.randint(0, 2**32 - 1)
                        )
                        node["inputs"]["cfg"] = cfg
                    elif node.get("class_type") == "WanImageToVideo":
                        node["inputs"]["length"] = num_frames
            else:
                # Fallback to standard build if file not found
                logger.warning("⚠️ Preset workflow not found, using standard build")
                workflow = self.build_q6_workflow(
                    image_name=image_name,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    fps=fps,
                    steps=steps,
                    cfg=cfg,
                    seed=seed,
                    output_prefix=output_prefix,
                    high_noise_steps=high_noise_steps,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    aspect_ratio=aspect_ratio,
                    long_edge=long_edge,
                    unet_high_noise=unet_high_noise,
                    unet_low_noise=unet_low_noise,
                    lora_configs=lora_configs,
                )
        else:
            # Standard mode: build workflow dynamically
            workflow = self.build_q6_workflow(
                image_name=image_name,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                fps=fps,
                steps=steps,
                cfg=cfg,
                seed=seed,
                output_prefix=output_prefix,
                high_noise_steps=high_noise_steps,
                sampler_name=sampler_name,
                scheduler=scheduler,
                aspect_ratio=aspect_ratio,
                long_edge=long_edge,
                unet_high_noise=unet_high_noise,
                unet_low_noise=unet_low_noise,
                lora_configs=lora_configs,
            )

        # 4. Queue workflow
        prompt_id = self.queue_prompt(workflow)
        if not prompt_id:
            return None

        # 5. Wait for completion
        logger.info("⏳ Executing Q6 workflow...")
        history = self.wait_for_completion(
            prompt_id, progress_callback=progress_callback
        )
        if not history:
            return None

        # 6. Get output video
        return self.get_output_video(history, output_dir, prompt_id)

    def generate_enhanced_video(
        self,
        image_path: str,
        prompt: str,
        output_dir: str,
        resolution: str = "480p",
        aspect_ratio: str = "1:1",
        num_frames: int = 41,
        fps: int = 16,
        steps: int = 4,
        cfg: float = 1.0,
        seed: int = -1,
        output_prefix: str = "oelala_wan22enh",
        model_variant: str = "HIGH",
        progress_callback=None,
    ) -> Optional[str]:
        """
        Full pipeline for WAN 2.2 Enhanced NSFW model (Lightning edition).
        Uses Q4KM GGUF with default Lightning settings (steps=4, cfg=1).
        """
        if not self.is_available():
            logger.error("❌ ComfyUI not available")
            return None

        # 1. Upload image
        logger.info(f"📤 Uploading image: {image_path}")
        image_name = self.upload_image(image_path)
        if not image_name:
            return None

        # 2. Calculate dimensions
        width, height = self.get_resolution_dimensions(resolution, aspect_ratio)
        logger.info(f"📐 Resolution: {width}x{height} ({resolution}, {aspect_ratio})")

        # 3. Build Enhanced workflow
        logger.info(
            f"🔧 Building Enhanced workflow: {num_frames}f @ {fps}fps, {steps} steps, cfg={cfg}"
        )
        workflow = self.build_enhanced_workflow(
            image_name=image_name,
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            steps=steps,
            cfg=cfg,
            seed=seed,
            output_prefix=output_prefix,
            model_variant=model_variant,
        )

        # 4. Queue workflow
        prompt_id = self.queue_prompt(workflow)
        if not prompt_id:
            return None

        # 5. Wait for completion
        logger.info("⏳ Executing Enhanced workflow...")
        history = self.wait_for_completion(
            prompt_id, progress_callback=progress_callback
        )
        if not history:
            return None

        # 6. Get output video
        return self.get_output_video(history, output_dir, prompt_id)

    def generate_sequential_video(
        self,
        image_path: str,
        prompt: str,
        output_dir: str,
        clip_count: int = 2,
        resolution: str = "480p",
        aspect_ratio: str = "16:9",
        num_frames: int = 41,
        fps: int = 16,
        steps: int = 6,
        cfg: float = 1.0,
        seed: int = -1,
        output_prefix: str = "sequential",
        unet_high_noise: str = "wan2.2_i2v_high_noise_14B_Q6_K.gguf",
        unet_low_noise: str = "wan2.2_i2v_low_noise_14B_Q6_K.gguf",
        lora_configs: list = None,
        progress_callback=None,
    ) -> Optional[str]:
        """
        Generate a sequential video by chaining multiple clips together.
        Each clip starts with the last frame of the previous clip.

        Args:
            clip_count: Number of clips to chain (1-5)
            Other args: Same as generate_q6_video

        Returns:
            Path to the final combined video
        """
        if not self.is_available():
            logger.error("❌ ComfyUI not available")
            return None

        clip_count = max(1, min(5, clip_count))  # Clamp to 1-5

        if clip_count == 1:
            # Just run normal generation
            return self.generate_q6_video(
                image_path=image_path,
                prompt=prompt,
                output_dir=output_dir,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                num_frames=num_frames,
                fps=fps,
                steps=steps,
                cfg=cfg,
                seed=seed,
                output_prefix=output_prefix,
                unet_high_noise=unet_high_noise,
                unet_low_noise=unet_low_noise,
                lora_configs=lora_configs,
                progress_callback=progress_callback,
            )

        logger.info(f"🎬 Starting sequential generation: {clip_count} clips")

        # 1. Upload initial image
        logger.info(f"📤 Uploading initial image: {image_path}")
        image_name = self.upload_image(image_path)
        if not image_name:
            return None

        # 2. Build and execute sequential workflow
        workflow = self._build_sequential_workflow(
            image_name=image_name,
            prompt=prompt,
            clip_count=clip_count,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            num_frames=num_frames,
            fps=fps,
            steps=steps,
            cfg=cfg,
            seed=seed if seed >= 0 else random.randint(0, 2**32 - 1),
            output_prefix=output_prefix,
            unet_high_noise=unet_high_noise,
            unet_low_noise=unet_low_noise,
            lora_configs=lora_configs or [],
        )

        # 3. Queue workflow
        prompt_id = self.queue_prompt(workflow)
        if not prompt_id:
            return None

        # 4. Wait for completion
        total_frames = num_frames * clip_count
        logger.info(
            f"⏳ Executing sequential workflow... ({total_frames} total frames)"
        )
        history = self.wait_for_completion(
            prompt_id, progress_callback=progress_callback
        )
        if not history:
            return None

        # 5. Get output video (the combined one)
        return self.get_output_video(history, output_dir, prompt_id)

    def _build_sequential_workflow(
        self,
        image_name: str,
        prompt: str,
        clip_count: int,
        resolution: str,
        aspect_ratio: str,
        num_frames: int,
        fps: int,
        steps: int,
        cfg: float,
        seed: int,
        output_prefix: str,
        unet_high_noise: str,
        unet_low_noise: str,
        lora_configs: list,
    ) -> Dict:
        """
        Build a sequential workflow for N clips dynamically.

        Structure:
        - Shared: Loaders (Unet High/Low, VAE, CLIP), SageAttention, LoRAs, Prompts
        - Per clip: WanImageToVideo → KSampler(High) → KSampler(Low) → VAEDecode → VideoCombine
        - Extract: VHS_SelectImages (indexes="-1") for last frame
        - Merge: Chain of VHS_MergeImages to combine all clips
        - Final: VHS_VideoCombine for combined output
        """
        # Get dimensions
        width, height = self.get_resolution_dimensions(resolution, aspect_ratio)
        split_step = steps // 2

        workflow = {}
        node_id = 1

        # === SHARED LOADERS (nodes 1-8) ===

        # 1: Unet High Noise Loader
        workflow[str(node_id)] = {
            "inputs": {
                "unet_name": unet_high_noise,
                "dequant_dtype": "default",
                "patch_dtype": "default",
                "patch_on_device": False,
                "compute_device": "cuda:0",
                "virtual_vram_gb": 16,
                "donor_device": "cuda:1",
                "expert_mode_allocations": "cuda:0,11gb;cuda:1,15gb;cpu,1gb",
                "eject_models": True,
            },
            "class_type": "UnetLoaderGGUFAdvancedDisTorch2MultiGPU",
            "_meta": {"title": "GGUF High Noise"},
        }
        unet_high_id = str(node_id)
        node_id += 1

        # 2: Unet Low Noise Loader
        workflow[str(node_id)] = {
            "inputs": {
                "unet_name": unet_low_noise,
                "dequant_dtype": "default",
                "patch_dtype": "default",
                "patch_on_device": False,
                "compute_device": "cuda:0",
                "virtual_vram_gb": 16,
                "donor_device": "cuda:1",
                "expert_mode_allocations": "cuda:0,11gb;cuda:1,15gb;cpu,1gb",
                "eject_models": True,
            },
            "class_type": "UnetLoaderGGUFAdvancedDisTorch2MultiGPU",
            "_meta": {"title": "GGUF Low Noise"},
        }
        unet_low_id = str(node_id)
        node_id += 1

        # 3: VAE Loader
        workflow[str(node_id)] = {
            "inputs": {
                "vae_name": "wan_2.1_vae.safetensors",
                "compute_device": "cuda:0",
                "virtual_vram_gb": 4,
                "donor_device": "cuda:1",
                "expert_mode_allocations": "cuda:0,11gb;cuda:1,15gb;cpu,1gb",
                "eject_models": True,
            },
            "class_type": "VAELoaderDisTorch2MultiGPU",
            "_meta": {"title": "VAE"},
        }
        vae_id = str(node_id)
        node_id += 1

        # 4: CLIP Loader
        workflow[str(node_id)] = {
            "inputs": {
                "clip_name": "umt5-xxl-enc-bf16.safetensors",
                "type": "wan",
                "device": "cuda:0",
                "virtual_vram_gb": 4,
                "donor_device": "cuda:1",
                "expert_mode_allocations": "cuda:0,11gb;cuda:1,15gb;cpu,1gb",
                "eject_models": True,
            },
            "class_type": "CLIPLoaderDisTorch2MultiGPU",
            "_meta": {"title": "T5-XXL"},
        }
        clip_id = str(node_id)
        node_id += 1

        # 5: ModelSamplingSD3 High
        workflow[str(node_id)] = {
            "inputs": {"shift": 8, "model": [unet_high_id, 0]},
            "class_type": "ModelSamplingSD3",
            "_meta": {"title": "ModelSampling High"},
        }
        model_shift_high_id = str(node_id)
        node_id += 1

        # 6: ModelSamplingSD3 Low
        workflow[str(node_id)] = {
            "inputs": {"shift": 8, "model": [unet_low_id, 0]},
            "class_type": "ModelSamplingSD3",
            "_meta": {"title": "ModelSampling Low"},
        }
        model_shift_low_id = str(node_id)
        node_id += 1

        # 7: SageAttention High
        workflow[str(node_id)] = {
            "inputs": {
                "sage_attention": "sageattn_qk_int8_pv_fp16_triton",
                "allow_compile": False,
                "model": [model_shift_high_id, 0],
            },
            "class_type": "PathchSageAttentionKJ",
            "_meta": {"title": "SageAttn High"},
        }
        sage_high_id = str(node_id)
        node_id += 1

        # 8: SageAttention Low
        workflow[str(node_id)] = {
            "inputs": {
                "sage_attention": "sageattn_qk_int8_pv_fp16_triton",
                "allow_compile": False,
                "model": [model_shift_low_id, 0],
            },
            "class_type": "PathchSageAttentionKJ",
            "_meta": {"title": "SageAttn Low"},
        }
        sage_low_id = str(node_id)
        node_id += 1

        # 9: Power Lora High
        lora_high_inputs = {
            "PowerLoraLoaderHeaderWidget": {"type": "PowerLoraLoaderHeaderWidget"},
            "model": [sage_high_id, 0],
            "clip": [clip_id, 0],
        }
        # Add LoRAs
        for i, lora in enumerate(lora_configs[:4], 1):
            lora_high_inputs[f"lora_{i}"] = {
                "on": True,
                "lora": lora.get("high", lora.get("name", "")),
                "strength": lora.get("strength", 1.5),
            }
        workflow[str(node_id)] = {
            "inputs": lora_high_inputs,
            "class_type": "Power Lora Loader (rgthree)",
            "_meta": {"title": "Power Lora High"},
        }
        lora_high_id = str(node_id)
        node_id += 1

        # 10: Power Lora Low
        lora_low_inputs = {
            "PowerLoraLoaderHeaderWidget": {"type": "PowerLoraLoaderHeaderWidget"},
            "model": [sage_low_id, 0],
        }
        for i, lora in enumerate(lora_configs[:4], 1):
            lora_low_inputs[f"lora_{i}"] = {
                "on": True,
                "lora": lora.get("low", lora.get("high", lora.get("name", ""))),
                "strength": lora.get("strength", 1.5),
            }
        workflow[str(node_id)] = {
            "inputs": lora_low_inputs,
            "class_type": "Power Lora Loader (rgthree)",
            "_meta": {"title": "Power Lora Low"},
        }
        lora_low_id = str(node_id)
        node_id += 1

        # 11: Positive Prompt
        workflow[str(node_id)] = {
            "inputs": {"text": prompt, "clip": [lora_high_id, 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive Prompt"},
        }
        pos_prompt_id = str(node_id)
        node_id += 1

        # 12: Negative Prompt
        workflow[str(node_id)] = {
            "inputs": {
                "text": "low quality, blurry, unstable, artifacts, flickering, jitter, sudden changes",
                "clip": [lora_high_id, 1],
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Negative Prompt"},
        }
        neg_prompt_id = str(node_id)
        node_id += 1

        # 13: Load Initial Image
        workflow[str(node_id)] = {
            "inputs": {"image": image_name},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Start Image"},
        }
        load_image_id = str(node_id)
        node_id += 1

        # === PER-CLIP NODES ===
        clip_decode_ids = []  # Store VAEDecode output IDs for merging
        current_image_id = load_image_id
        current_image_slot = 0

        for clip_idx in range(clip_count):
            clip_seed = seed + clip_idx

            # WanImageToVideo
            workflow[str(node_id)] = {
                "inputs": {
                    "width": width,
                    "height": height,
                    "length": num_frames,
                    "batch_size": 1,
                    "positive": [pos_prompt_id, 0],
                    "negative": [neg_prompt_id, 0],
                    "vae": [vae_id, 0],
                    "start_image": [current_image_id, current_image_slot],
                },
                "class_type": "WanImageToVideo",
                "_meta": {"title": f"WanI2V Clip {clip_idx + 1}"},
            }
            wan_i2v_id = str(node_id)
            node_id += 1

            # KSampler High Noise
            workflow[str(node_id)] = {
                "inputs": {
                    "add_noise": "enable",
                    "noise_seed": clip_seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "uni_pc",
                    "scheduler": "normal",
                    "start_at_step": 0,
                    "end_at_step": split_step,
                    "return_with_leftover_noise": "enable",
                    "model": [lora_high_id, 0],
                    "positive": [wan_i2v_id, 0],
                    "negative": [wan_i2v_id, 1],
                    "latent_image": [wan_i2v_id, 2],
                },
                "class_type": "KSamplerAdvanced",
                "_meta": {"title": f"Sampler High Clip {clip_idx + 1}"},
            }
            sampler_high_id = str(node_id)
            node_id += 1

            # KSampler Low Noise
            workflow[str(node_id)] = {
                "inputs": {
                    "add_noise": "disable",
                    "noise_seed": clip_seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "uni_pc",
                    "scheduler": "normal",
                    "start_at_step": split_step,
                    "end_at_step": 10000,
                    "return_with_leftover_noise": "disable",
                    "model": [lora_low_id, 0],
                    "positive": [wan_i2v_id, 0],
                    "negative": [wan_i2v_id, 1],
                    "latent_image": [sampler_high_id, 0],
                },
                "class_type": "KSamplerAdvanced",
                "_meta": {"title": f"Sampler Low Clip {clip_idx + 1}"},
            }
            sampler_low_id = str(node_id)
            node_id += 1

            # VAE Decode
            workflow[str(node_id)] = {
                "inputs": {"samples": [sampler_low_id, 0], "vae": [vae_id, 0]},
                "class_type": "VAEDecode",
                "_meta": {"title": f"VAEDecode Clip {clip_idx + 1}"},
            }
            vae_decode_id = str(node_id)
            clip_decode_ids.append(vae_decode_id)
            node_id += 1

            # Save individual clip
            workflow[str(node_id)] = {
                "inputs": {
                    "frame_rate": fps,
                    "loop_count": 0,
                    "filename_prefix": f"{output_prefix}/clip{clip_idx + 1}",
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 19,
                    "save_metadata": True,
                    "trim_to_audio": False,
                    "pingpong": False,
                    "save_output": True,
                    "images": [vae_decode_id, 0],
                },
                "class_type": "VHS_VideoCombine",
                "_meta": {"title": f"Save Clip {clip_idx + 1}"},
            }
            node_id += 1

            # Extract last frame for next clip (except for last clip)
            if clip_idx < clip_count - 1:
                workflow[str(node_id)] = {
                    "inputs": {
                        "image": [vae_decode_id, 0],
                        "indexes": "-1",
                        "err_if_missing": True,
                        "err_if_empty": True,
                    },
                    "class_type": "VHS_SelectImages",
                    "_meta": {"title": f"Last Frame Clip {clip_idx + 1}"},
                }
                current_image_id = str(node_id)
                current_image_slot = 0
                node_id += 1

        # === MERGE ALL CLIPS ===
        if clip_count >= 2:
            # Chain merge: merge clip1+clip2, then result+clip3, etc.
            current_merge_id = None

            for i in range(clip_count - 1):
                if i == 0:
                    # First merge: clip1 + clip2
                    workflow[str(node_id)] = {
                        "inputs": {
                            "images_A": [clip_decode_ids[0], 0],
                            "images_B": [clip_decode_ids[1], 0],
                            "merge_strategy": "match A",
                            "scale_method": "bilinear",
                            "crop": "disabled",
                        },
                        "class_type": "VHS_MergeImages",
                        "_meta": {"title": "Merge Clips 1+2"},
                    }
                else:
                    # Subsequent merges: previous_result + next_clip
                    workflow[str(node_id)] = {
                        "inputs": {
                            "images_A": [current_merge_id, 0],
                            "images_B": [clip_decode_ids[i + 1], 0],
                            "merge_strategy": "match A",
                            "scale_method": "bilinear",
                            "crop": "disabled",
                        },
                        "class_type": "VHS_MergeImages",
                        "_meta": {"title": f"Merge +Clip {i + 2}"},
                    }
                current_merge_id = str(node_id)
                node_id += 1

            # Final combined video output
            workflow[str(node_id)] = {
                "inputs": {
                    "frame_rate": fps,
                    "loop_count": 0,
                    "filename_prefix": f"{output_prefix}/combined",
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 19,
                    "save_metadata": True,
                    "trim_to_audio": False,
                    "pingpong": False,
                    "save_output": True,
                    "images": [current_merge_id, 0],
                },
                "class_type": "VHS_VideoCombine",
                "_meta": {"title": "Combined Video"},
            }

        logger.info(
            f"🔧 Built sequential workflow with {len(workflow)} nodes for {clip_count} clips"
        )
        return workflow

    def build_distorch2_workflow(
        self,
        image_name: str,
        prompt: str,
        negative_prompt: str = "low quality, blurry, out of focus, unstable camera, artifacts, distortion",
        width: int = 480,
        height: int = 480,
        num_frames: int = 41,
        fps: int = 16,
        steps: int = 6,
        cfg: float = 1.0,
        seed: int = -1,
        output_prefix: str = "wan22_distorch2",
        lora_strength: float = 1.5,
        lora_config: list = None,  # Dynamic LoRA list: [{name, high, low}, ...]
        generation_mode: str = "standard",  # "standard" or "nsfw_lora"
    ) -> Dict:
        """
        Build DisTorch2 dual-noise workflow with Power Lora Loader.
        Uses high_noise model for first half of steps, low_noise for second half.

        Args:
            lora_config: List of dicts with 'high' and 'low' paths for each LoRA
            generation_mode: Which workflow preset to use ("standard", "nsfw_lora")
        """
        import copy

        # Load workflow from file based on generation mode
        mode_config = I2V_GENERATION_MODES.get(
            generation_mode, I2V_GENERATION_MODES["standard"]
        )
        loaded_workflow = load_workflow_from_file(mode_config["workflow_file"])

        if loaded_workflow:
            workflow = copy.deepcopy(loaded_workflow)
            logger.info(f"📂 Loaded workflow from file: {mode_config['workflow_file']}")
        else:
            # Fallback to hardcoded workflow
            workflow = copy.deepcopy(WAN22_I2V_DISTORCH2_API_WORKFLOW)
            logger.warning(
                f"⚠️ Using fallback hardcoded workflow (mode: {generation_mode})"
            )

        # Set seed
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        # Calculate split step (half of total steps)
        split_step = steps // 2

        # Update prompts
        workflow["7"]["inputs"]["text"] = prompt
        workflow["8"]["inputs"]["text"] = negative_prompt

        # Update dimensions and frames
        workflow["16"]["inputs"]["width"] = width
        workflow["16"]["inputs"]["height"] = height
        workflow["16"]["inputs"]["length"] = num_frames

        # Update samplers
        workflow["10"]["inputs"]["noise_seed"] = seed
        workflow["10"]["inputs"]["steps"] = steps
        workflow["10"]["inputs"]["cfg"] = cfg
        workflow["10"]["inputs"]["end_at_step"] = split_step

        workflow["11"]["inputs"]["noise_seed"] = seed + 1
        workflow["11"]["inputs"]["steps"] = steps
        workflow["11"]["inputs"]["cfg"] = cfg
        workflow["11"]["inputs"]["start_at_step"] = split_step

        # Update output settings
        workflow["13"]["inputs"]["frame_rate"] = fps
        workflow["13"]["inputs"]["filename_prefix"] = output_prefix

        # Update image
        workflow["18"]["inputs"]["image"] = image_name

        # Configure LoRAs dynamically
        lora_config = lora_config or []

        # Clear all existing LoRA slots first (disable them)
        for node_id in ["19", "20"]:
            for i in range(1, 9):  # Support up to 8 LoRAs
                lora_key = f"lora_{i}"
                if lora_key in workflow[node_id]["inputs"]:
                    workflow[node_id]["inputs"][lora_key]["on"] = False

        # Configure LoRAs from dynamic config
        for idx, lora in enumerate(lora_config[:8]):  # Max 8 LoRAs
            lora_key = f"lora_{idx + 1}"
            high_path = lora.get("high", "")
            low_path = lora.get("low", high_path)  # Use high if low not specified

            # Node 19: High-noise model LoRAs
            if lora_key in workflow["19"]["inputs"]:
                workflow["19"]["inputs"][lora_key] = {
                    "on": True,
                    "lora": high_path,
                    "strength": lora_strength,
                }
            else:
                # Add new LoRA slot if it doesn't exist
                workflow["19"]["inputs"][lora_key] = {
                    "on": True,
                    "lora": high_path,
                    "strength": lora_strength,
                }

            # Node 20: Low-noise model LoRAs
            if lora_key in workflow["20"]["inputs"]:
                workflow["20"]["inputs"][lora_key] = {
                    "on": True,
                    "lora": low_path,
                    "strength": lora_strength,
                }
            else:
                workflow["20"]["inputs"][lora_key] = {
                    "on": True,
                    "lora": low_path,
                    "strength": lora_strength,
                }

            logger.info(f"   🎨 LoRA {idx + 1}: HIGH={high_path}, LOW={low_path}")

        logger.info(
            f"🔧 DisTorch2 workflow built: {width}x{height}, {num_frames}f, steps={steps} (split@{split_step}), cfg={cfg}, loras={len(lora_config)}"
        )
        return workflow

    def generate_distorch2_video(
        self,
        image_path: str,
        prompt: str,
        output_dir: str,
        negative_prompt: str = "low quality, blurry, out of focus, unstable camera, artifacts, distortion",
        resolution: str = "480p",
        aspect_ratio: str = "1:1",
        num_frames: int = 41,
        fps: int = 16,
        steps: int = 6,
        cfg: float = 1.0,
        seed: int = -1,
        output_prefix: str = "oelala_distorch2",
        lora_strength: float = 1.5,
        lora_config: list = None,  # Dynamic LoRA list: [{name, high, low}, ...]
        # Legacy parameters for backwards compatibility
        enable_nsfw_lora: bool = None,
        enable_dreamlay_lora: bool = None,
        enable_lightx2v_lora: bool = None,
        enable_cumshot_lora: bool = None,
        progress_callback=None,
    ) -> Optional[str]:
        """
        Full pipeline for WAN 2.2 DisTorch2 dual-noise models.
        High-quality Q6_K GGUF with dual-GPU distribution.

        Uses:
        - high_noise model for first half of sampling steps
        - low_noise model for second half
        - Power Lora Loader with configurable LoRAs
        - DisTorch2 multi-GPU distribution (cuda:0,12gb;cuda:1,16gb;cpu,*)

        Args:
            lora_config: List of LoRA configs, each with 'high' and 'low' paths
                         Example: [{'name': 'NSFW', 'high': 'path/to/high.safetensors', 'low': 'path/to/low.safetensors'}]
        """
        if not self.is_available():
            logger.error("❌ ComfyUI not available")
            return None

        # Handle legacy boolean parameters
        if lora_config is None:
            lora_config = []
            # Convert legacy booleans to lora_config if provided
            if enable_dreamlay_lora:
                lora_config.append(
                    {
                        "name": "DR34ML4Y",
                        "high": "wan 2.2/DR34ML4Y_I2V_14B_HIGH.safetensors",
                        "low": "wan 2.2/DR34ML4Y_I2V_14B_LOW.safetensors",
                    }
                )
            if enable_nsfw_lora:
                lora_config.append(
                    {
                        "name": "NSFW",
                        "high": "wan 2.2/NSFW-22-H-e8.safetensors",
                        "low": "wan 2.2/NSFW-22-L-e8.safetensors",
                    }
                )
            if enable_lightx2v_lora:
                lora_config.append(
                    {
                        "name": "LightX2V",
                        "high": "wan/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors",
                        "low": "wan/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors",
                    }
                )
            if enable_cumshot_lora:
                lora_config.append(
                    {
                        "name": "Cumshot",
                        "high": "masturbation_cumshot_v1.1_e310.safetensors",
                        "low": "masturbation_cumshot_v1.1_e310.safetensors",
                    }
                )

        # 1. Upload image
        logger.info(f"📤 Uploading image: {image_path}")
        image_name = self.upload_image(image_path)
        if not image_name:
            return None

        # 2. Calculate dimensions
        width, height = self.get_resolution_dimensions(resolution, aspect_ratio)
        logger.info(f"📐 Resolution: {width}x{height} ({resolution}, {aspect_ratio})")

        # 3. Build DisTorch2 workflow
        logger.info(
            f"🔧 Building DisTorch2 workflow: {num_frames}f @ {fps}fps, {steps} steps, cfg={cfg}, lora={lora_strength}"
        )
        logger.info(
            f"   🎨 LoRAs ({len(lora_config)}): {[l.get('name', 'unknown') for l in lora_config]}"
        )
        workflow = self.build_distorch2_workflow(
            image_name=image_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            steps=steps,
            cfg=cfg,
            seed=seed,
            output_prefix=output_prefix,
            lora_strength=lora_strength,
            lora_config=lora_config,
        )

        # 4. Queue workflow
        prompt_id = self.queue_prompt(workflow)
        if not prompt_id:
            return None

        # 5. Wait for completion
        logger.info("⏳ Executing DisTorch2 workflow...")
        history = self.wait_for_completion(
            prompt_id, progress_callback=progress_callback
        )
        if not history:
            return None

        # 6. Get output video
        return self.get_output_video(history, output_dir, prompt_id)

    def build_video_upscale_workflow(
        self,
        video_path: str,
        scale: int = 2,
        output_prefix: str = "upscaled",
        model: str = "realesrgan-x4plus",
        quality_preset: str = "balanced",
    ) -> Optional[Dict]:
        """
        Build a video upscaling workflow using Real-ESRGAN.

        Args:
            video_path: Path to input video
            scale: Upscale factor (2 or 4)
            output_prefix: Prefix for output filename
            model: Upscale model to use
            quality_preset: Quality preset (fast, balanced, quality)

        Returns:
            ComfyUI workflow dict or None if video doesn't exist
        """
        if not Path(video_path).exists():
            logger.error(f"Video not found: {video_path}")
            return None

        # Upload video to ComfyUI
        video_name = self.upload_video(video_path)
        if not video_name:
            logger.error("Failed to upload video for upscaling")
            return None

        # Map quality preset to CRF value
        crf_map = {"fast": 28, "balanced": 19, "quality": 14}
        crf = crf_map.get(quality_preset, 19)

        workflow = {
            "1": {
                "inputs": {
                    "video": video_name,
                    "force_rate": 0,
                    "force_size": "Disabled",
                    "custom_width": 0,
                    "custom_height": 0,
                    "frame_load_cap": 0,
                    "skip_first_frames": 0,
                    "select_every_nth": 1,
                },
                "class_type": "VHS_LoadVideo",
            },
            "2": {
                "inputs": {
                    "model_name": f"{model}.pth",
                },
                "class_type": "UpscaleModelLoader",
            },
            "3": {
                "inputs": {
                    "upscale_model": ["2", 0],
                    "image": ["1", 0],
                },
                "class_type": "ImageUpscaleWithModel",
            },
            "4": {
                "inputs": {
                    "frame_rate": ["1", 2],
                    "loop_count": 0,
                    "filename_prefix": output_prefix,
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": crf,
                    "save_metadata": True,
                    "images": ["3", 0],
                    "audio": ["1", 1],
                },
                "class_type": "VHS_VideoCombine",
            },
        }

        logger.info(
            f"🔧 Built video upscale workflow: {scale}x with {model}, quality={quality_preset}"
        )
        return workflow

    def build_rife_workflow(
        self,
        video_path: str,
        target_fps: int = 60,
        output_prefix: str = "interpolated",
        multiplier: int = 2,
    ) -> Optional[Dict]:
        """
        Build a RIFE frame interpolation workflow.

        Args:
            video_path: Path to input video
            target_fps: Target framerate
            output_prefix: Prefix for output filename
            multiplier: Frame multiplier (2 = double frames, 4 = quadruple)

        Returns:
            ComfyUI workflow dict or None if video doesn't exist
        """
        if not Path(video_path).exists():
            logger.error(f"Video not found: {video_path}")
            return None

        # Upload video to ComfyUI
        video_name = self.upload_video(video_path)
        if not video_name:
            logger.error("Failed to upload video for interpolation")
            return None

        workflow = {
            "1": {
                "inputs": {
                    "video": video_name,
                    "force_rate": 0,
                    "force_size": "Disabled",
                    "custom_width": 0,
                    "custom_height": 0,
                    "frame_load_cap": 0,
                    "skip_first_frames": 0,
                    "select_every_nth": 1,
                },
                "class_type": "VHS_LoadVideo",
            },
            "2": {
                "inputs": {
                    "ckpt_name": "rife49.pth",
                    "clear_cache_after_n_frames": 10,
                    "multiplier": multiplier,
                    "fast_mode": True,
                    "ensemble": True,
                    "scale_factor": 1.0,
                    "frames": ["1", 0],
                },
                "class_type": "RIFE VFI",
            },
            "3": {
                "inputs": {
                    "frame_rate": target_fps,
                    "loop_count": 0,
                    "filename_prefix": output_prefix,
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 19,
                    "save_metadata": True,
                    "images": ["2", 0],
                    "audio": ["1", 1],
                },
                "class_type": "VHS_VideoCombine",
            },
        }

        logger.info(f"🔧 Built RIFE workflow: {multiplier}x → {target_fps}fps")
        return workflow

    def build_video_concat_workflow(
        self,
        video_paths: list,
        output_prefix: str = "concatenated",
        transition: str = "none",
    ) -> Optional[Dict]:
        """
        Build a video concatenation workflow to join multiple videos.

        Args:
            video_paths: List of video paths to concatenate
            output_prefix: Prefix for output filename
            transition: Transition type between clips ("none", "crossfade")

        Returns:
            ComfyUI workflow dict or None
        """
        if not video_paths or len(video_paths) < 2:
            logger.error("Need at least 2 videos to concatenate")
            return None

        # Upload all videos
        video_names = []
        for vp in video_paths:
            if not Path(vp).exists():
                logger.error(f"Video not found: {vp}")
                return None
            name = self.upload_video(vp)
            if not name:
                return None
            video_names.append(name)

        # Build workflow with video loaders and batch
        workflow = {}
        node_id = 1

        # Load each video
        load_nodes = []
        for i, vname in enumerate(video_names):
            workflow[str(node_id)] = {
                "inputs": {
                    "video": vname,
                    "force_rate": 0,
                    "force_size": "Disabled",
                    "custom_width": 0,
                    "custom_height": 0,
                    "frame_load_cap": 0,
                    "skip_first_frames": 0,
                    "select_every_nth": 1,
                },
                "class_type": "VHS_LoadVideo",
            }
            load_nodes.append(str(node_id))
            node_id += 1

        # Batch images together
        workflow[str(node_id)] = {
            "inputs": {
                "images": [load_nodes[0], 0],
            },
            "class_type": "ImageBatch",
        }
        batch_node = str(node_id)

        # Chain additional videos
        for ln in load_nodes[1:]:
            node_id += 1
            workflow[str(node_id)] = {
                "inputs": {
                    "images1": [batch_node, 0],
                    "images2": [ln, 0],
                },
                "class_type": "ImageBatch",
            }
            batch_node = str(node_id)

        node_id += 1

        # Output combined video
        workflow[str(node_id)] = {
            "inputs": {
                "frame_rate": 16,  # Will be overridden by first video's rate
                "loop_count": 0,
                "filename_prefix": output_prefix,
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "images": [batch_node, 0],
            },
            "class_type": "VHS_VideoCombine",
        }

        logger.info(f"🔧 Built video concat workflow: {len(video_paths)} videos")
        return workflow


# WAN 2.2 I2V DisTorch2 Dual-Noise Workflow (Q6_K 14B models)
# Uses separate high_noise and low_noise models with 2-stage KSampler
WAN22_I2V_DISTORCH2_API_WORKFLOW = {
    "1": {
        "inputs": {
            "unet_name": "wan2.2_i2v_high_noise_14B_Q6_K.gguf",
            "dequant_dtype": "default",
            "patch_dtype": "default",
            "patch_on_device": False,
            "compute_device": "cuda:0",
            "virtual_vram_gb": 16,
            "donor_device": "cuda:1",
            "expert_mode_allocations": "cuda:0,12gb;cuda:1,16gb;cpu,*",
            "eject_models": True,
        },
        "class_type": "UnetLoaderGGUFAdvancedDisTorch2MultiGPU",
    },
    "2": {
        "inputs": {
            "unet_name": "wan2.2_i2v_low_noise_14B_Q6_K.gguf",
            "dequant_dtype": "default",
            "patch_dtype": "default",
            "patch_on_device": False,
            "compute_device": "cuda:0",
            "virtual_vram_gb": 16,
            "donor_device": "cuda:1",
            "expert_mode_allocations": "cuda:0,12gb;cuda:1,16gb;cpu,*",
            "eject_models": True,
        },
        "class_type": "UnetLoaderGGUFAdvancedDisTorch2MultiGPU",
    },
    "3": {
        "inputs": {
            "vae_name": "wan_2.1_vae.safetensors",
            "compute_device": "cuda:0",
            "virtual_vram_gb": 4,
            "donor_device": "cuda:1",
            "expert_mode_allocations": "cuda:0,12gb;cuda:1,16gb;cpu,*",
            "eject_models": True,
        },
        "class_type": "VAELoaderDisTorch2MultiGPU",
    },
    "4": {
        "inputs": {
            "clip_name": "umt5-xxl-enc-bf16.safetensors",
            "type": "wan",
            "device": "cuda:0",
            "virtual_vram_gb": 4,
            "donor_device": "cuda:1",
            "expert_mode_allocations": "cuda:0,12gb;cuda:1,16gb;cpu,*",
            "eject_models": True,
        },
        "class_type": "CLIPLoaderDisTorch2MultiGPU",
    },
    "5": {
        "inputs": {
            "sage_attention": "sageattn_qk_int8_pv_fp16_triton",
            "allow_compile": False,
            "model": ["14", 0],
        },
        "class_type": "PathchSageAttentionKJ",
    },
    "6": {
        "inputs": {
            "sage_attention": "sageattn_qk_int8_pv_fp16_triton",
            "allow_compile": False,
            "model": ["15", 0],
        },
        "class_type": "PathchSageAttentionKJ",
    },
    "7": {"inputs": {"text": "", "clip": ["19", 1]}, "class_type": "CLIPTextEncode"},
    "8": {
        "inputs": {
            "text": "low quality, blurry, out of focus, unstable camera, artifacts, distortion",
            "clip": ["19", 1],
        },
        "class_type": "CLIPTextEncode",
    },
    "10": {
        "inputs": {
            "add_noise": "enable",
            "noise_seed": 0,
            "steps": 6,
            "cfg": 1,
            "sampler_name": "uni_pc",
            "scheduler": "normal",
            "start_at_step": 0,
            "end_at_step": 3,
            "return_with_leftover_noise": "enable",
            "model": ["19", 0],
            "positive": ["16", 0],
            "negative": ["16", 1],
            "latent_image": ["16", 2],
        },
        "class_type": "KSamplerAdvanced",
    },
    "11": {
        "inputs": {
            "add_noise": "disable",
            "noise_seed": 0,
            "steps": 6,
            "cfg": 1,
            "sampler_name": "uni_pc",
            "scheduler": "normal",
            "start_at_step": 3,
            "end_at_step": 10000,
            "return_with_leftover_noise": "disable",
            "model": ["20", 0],
            "positive": ["16", 0],
            "negative": ["16", 1],
            "latent_image": ["10", 0],
        },
        "class_type": "KSamplerAdvanced",
    },
    "12": {
        "inputs": {"samples": ["11", 0], "vae": ["3", 0]},
        "class_type": "VAEDecode",
    },
    "13": {
        "inputs": {
            "frame_rate": 16,
            "loop_count": 0,
            "filename_prefix": "wan22_distorch2",
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 19,
            "save_metadata": True,
            "trim_to_audio": False,
            "pingpong": False,
            "save_output": True,
            "images": ["12", 0],
        },
        "class_type": "VHS_VideoCombine",
    },
    "14": {"inputs": {"shift": 8, "model": ["1", 0]}, "class_type": "ModelSamplingSD3"},
    "15": {"inputs": {"shift": 8, "model": ["2", 0]}, "class_type": "ModelSamplingSD3"},
    "16": {
        "inputs": {
            "width": 480,
            "height": 480,
            "length": 41,
            "batch_size": 1,
            "positive": ["7", 0],
            "negative": ["8", 0],
            "vae": ["3", 0],
            "start_image": ["18", 0],
        },
        "class_type": "WanImageToVideo",
    },
    "18": {"inputs": {"image": "example.png"}, "class_type": "LoadImage"},
    "19": {
        "inputs": {
            "PowerLoraLoaderHeaderWidget": {"type": "PowerLoraLoaderHeaderWidget"},
            "lora_1": {
                "on": True,
                "lora": "wan 2.2/DR34ML4Y_I2V_14B_HIGH.safetensors",
                "strength": 1.5,
            },
            "lora_2": {
                "on": True,
                "lora": "wan 2.2/NSFW-22-H-e8.safetensors",
                "strength": 1.5,
            },
            "lora_3": {
                "on": True,
                "lora": "wan/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors",
                "strength": 1.5,
            },
            "lora_4": {
                "on": True,
                "lora": "masturbation_cumshot_v1.1_e310.safetensors",
                "strength": 1.5,
            },
            "➕ Add Lora": "",
            "model": ["5", 0],
            "clip": ["4", 0],
        },
        "class_type": "Power Lora Loader (rgthree)",
    },
    "20": {
        "inputs": {
            "PowerLoraLoaderHeaderWidget": {"type": "PowerLoraLoaderHeaderWidget"},
            "lora_1": {
                "on": True,
                "lora": "wan 2.2/DR34ML4Y_I2V_14B_LOW.safetensors",
                "strength": 1.5,
            },
            "lora_2": {
                "on": True,
                "lora": "wan 2.2/NSFW-22-L-e8.safetensors",
                "strength": 1.5,
            },
            "lora_3": {
                "on": True,
                "lora": "wan/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors",
                "strength": 1.5,
            },
            "lora_4": {
                "on": True,
                "lora": "masturbation_cumshot_v1.1_e310.safetensors",
                "strength": 1.5,
            },
            "➕ Add Lora": "",
            "model": ["6", 0],
        },
        "class_type": "Power Lora Loader (rgthree)",
    },
}


# Singleton instance
_comfyui_client: Optional[ComfyUIClient] = None


def get_comfyui_client() -> ComfyUIClient:
    """Get or create ComfyUI client singleton"""
    global _comfyui_client
    if _comfyui_client is None:
        _comfyui_client = ComfyUIClient()
    return _comfyui_client
