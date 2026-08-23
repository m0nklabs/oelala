#!/usr/bin/env python3
"""
ComfyUI API Client for Oelala Backend
Enables integration with ComfyUI for Wan2.2 Q5 GGUF workflows
"""

import json
import os
import uuid
import time
import requests
import websocket
import random
import threading
import math
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
    "blockswap_q8": {
        "name": "BlockSwap Q8 Experimental",
        "description": "Q8_0 quality with BlockSwap VRAM optimization, Lightning LoRA, NAG, MagCache, TorchCompile, Florence2 auto-captioning",
        "workflow_file": "ImageToVideo/wan22_i2v_blockswap_q8_api.json",
        "default_steps": 8,
        "default_cfg": 1.0,
        "features": [
            "Q8_0 dual-pass (higher quality than Q6_K)",
            "BlockSwap VRAM reduction",
            "Lightning LoRA (4-8 step fast generation)",
            "NAG (Normalized Attention Guidance)",
            "MagCache (magnitude-aware caching)",
            "TorchCompile JIT compilation",
            "Florence2 auto-captioning",
            "EnhanceAVideo quality boost",
            "CFGZeroStar guidance trick",
            "Optional 4x upscale + RIFE 2x interpolation",
        ],
    },
    "distorch2_q8": {
        "name": "DisTorch2 Q8 Experimental",
        "description": "Q8_0 quality with DisTorch2 multi-GPU distribution, user-selectable LoRAs, NAG, TorchCompile, Florence2",
        "workflow_file": "ImageToVideo/wan22_i2v_distorch2_q8_api.json",
        "default_steps": 8,
        "default_cfg": 1.0,
        "features": [
            "Q8_0 dual-pass (higher quality than Q6_K)",
            "DisTorch2 multi-GPU (cuda:0,10gb;cuda:1,14.5gb;cpu,*)",
            "User-selectable LoRAs (no forced Lightning)",
            "NAG (Normalized Attention Guidance)",
            "TorchCompile JIT compilation",
            "Florence2 auto-captioning",
            "EnhanceAVideo quality boost",
            "CFGZeroStar guidance trick",
            "Optional 4x upscale + RIFE 2x interpolation",
        ],
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
        "max_frames": 481,
        "default_frames": 41,
    },
    "ltx2": {
        "name": "LTX-2.3 22B",
        "description": "Lightricks LTX-2.3 22B distilled, fast 8-step generation",
        "workflow_file": None,  # Uses cloud workflow builder
        "model_type": "ltx2",
        "default_steps": 8,
        "default_cfg": 1.0,
        "max_frames": 481,
        "default_frames": 97,
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
            "device": "cuda:1",
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
            "device": "cuda:1",
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
            "compute_device": "cuda:1",
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
            "compute_device": "cuda:1",
            "virtual_vram_gb": 16,
            "donor_device": "cuda:0",
            "expert_mode_allocations": "cuda:1,0.25gb;cuda:0,8gb;cpu,*",
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
            "compute_device": "cuda:1",
            "virtual_vram_gb": 16,
            "donor_device": "cuda:0",
            "expert_mode_allocations": "cuda:1,0.25gb;cuda:0,8gb;cpu,*",
            "eject_models": True,
        },
    },
    # Node 3: VAE Loader (DisTorch2)
    "3": {
        "class_type": "VAELoaderDisTorch2MultiGPU",
        "inputs": {
            "vae_name": "wan_2.1_vae.safetensors",
            "compute_device": "cuda:1",
            "virtual_vram_gb": 16,
            "donor_device": "cuda:0",
            "expert_mode_allocations": "cuda:1,0.25gb;cuda:0,8gb;cpu,*",
            "eject_models": True,
        },
    },
    # Node 4: T5-XXL CLIP Loader (DisTorch2) - CONVERTED model!
    "4": {
        "class_type": "CLIPLoaderDisTorch2MultiGPU",
        "inputs": {
            "clip_name": "umt5-xxl-enc-bf16.safetensors",
            "type": "wan",
            "device": "cuda:1",
            "virtual_vram_gb": 16,
            "donor_device": "cuda:0",
            "expert_mode_allocations": "cuda:1,0.25gb;cuda:0,8gb;cpu,*",
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
# T2V (Text-to-Video) DisTorch2 Dual-Pass Q6_K workflow
# Identical architecture to I2V Q6 but WITHOUT image conditioning:
#   - Uses T2V GGUF models (16 in_channels, pure text-to-video)
#   - WanImageToVideo without start_image creates empty 5D video latent
#   - No LoadImage/ImageResize nodes needed
#   - Same dual-pass KSamplerAdvanced: High noise → Low noise
# ─────────────────────────────────────────────────────────────────────────────
WAN22_T2V_Q6_API_WORKFLOW = {
    # Node 1: High Noise GGUF Model Loader (DisTorch2) — T2V model
    "1": {
        "class_type": "UnetLoaderGGUFAdvancedDisTorch2MultiGPU",
        "inputs": {
            "unet_name": "wan2.2_t2v_high_noise_14B_Q6_K.gguf",
            "dequant_dtype": "default",
            "patch_dtype": "default",
            "patch_on_device": False,
            "compute_device": "cuda:1",
            "virtual_vram_gb": 16,
            "donor_device": "cuda:0",
            "expert_mode_allocations": "cuda:1,0.25gb;cuda:0,8gb;cpu,*",
            "eject_models": True,
        },
    },
    # Node 2: Low Noise GGUF Model Loader (DisTorch2) — T2V model
    "2": {
        "class_type": "UnetLoaderGGUFAdvancedDisTorch2MultiGPU",
        "inputs": {
            "unet_name": "wan2.2_t2v_low_noise_14B_Q6_K.gguf",
            "dequant_dtype": "default",
            "patch_dtype": "default",
            "patch_on_device": False,
            "compute_device": "cuda:1",
            "virtual_vram_gb": 16,
            "donor_device": "cuda:0",
            "expert_mode_allocations": "cuda:1,0.25gb;cuda:0,8gb;cpu,*",
            "eject_models": True,
        },
    },
    # Node 3: VAE Loader (DisTorch2)
    "3": {
        "class_type": "VAELoaderDisTorch2MultiGPU",
        "inputs": {
            "vae_name": "wan_2.1_vae.safetensors",
            "compute_device": "cuda:1",
            "virtual_vram_gb": 16,
            "donor_device": "cuda:0",
            "expert_mode_allocations": "cuda:1,0.25gb;cuda:0,8gb;cpu,*",
            "eject_models": True,
        },
    },
    # Node 4: T5-XXL CLIP Loader (DisTorch2)
    "4": {
        "class_type": "CLIPLoaderDisTorch2MultiGPU",
        "inputs": {
            "clip_name": "umt5-xxl-enc-bf16.safetensors",
            "type": "wan",
            "device": "cuda:1",
            "virtual_vram_gb": 16,
            "donor_device": "cuda:0",
            "expert_mode_allocations": "cuda:1,0.25gb;cuda:0,8gb;cpu,*",
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
    # Node 12: WanImageToVideo — T2V mode (NO start_image)
    # Without start_image this creates an empty 5D video latent [1, 16, F, H/8, W/8]
    # and passes through text conditioning unchanged
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
        },
    },
    # Node 13: KSamplerAdvanced - Pass 1 (High Noise) steps 0-3
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
            "model": ["7", 0],
            "positive": ["12", 0],
            "negative": ["12", 1],
            "latent_image": ["12", 2],
        },
    },
    # Node 14: KSamplerAdvanced - Pass 2 (Low Noise) steps 3+
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
            "model": ["8", 0],
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
            "filename_prefix": "oelala_t2v",
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
            "device": "cuda:1",
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
            "device": "cuda:1",
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
            "compute_device": "cuda:1",
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
            "widgets_values": ["umt5-xxl-enc-bf16.safetensors", "bf16", "cuda:1"],
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
            "widgets_values": ["Wan2.1_VAE.safetensors", "cuda:1", "bf16"],
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
                "cuda:1",
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

    def get_model_names(self, loader_type: str = "UnetLoaderGGUF") -> list[str]:
        """Fetch available models dynamically from ComfyUI"""
        try:
            import requests

            resp = requests.get(f"{self.base_url}/object_info/{loader_type}", timeout=5)
            data = resp.json()
            if loader_type in data:
                return data[loader_type]["input"]["required"].get("unet_name", [[]])[0]
        except Exception as e:
            logger.error(f"Error fetching {loader_type} models: {e}")
        return []

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

    def build_t2v_q6_workflow(
        self,
        prompt: str,
        negative_prompt: str = "low quality, blurry, out of focus, unstable camera, artifacts, distortion, low resolution, overexposed, underexposed, color banding, missing details, unrealistic lighting, flickering shadows, frame stutter, ghosting, bad reflections, unrealistic motion, pixelated textures, wrong physics, broken animation, rendering artifacts, compression noise, jitter, visual glitches",
        width: int = 480,
        height: int = 848,
        num_frames: int = 41,
        fps: int = 16,
        steps: int = 6,
        cfg: float = 1.0,
        seed: int = -1,
        output_prefix: str = "oelala_t2v",
        high_noise_steps: int = 3,
        sampler_name: str = "uni_pc",
        scheduler: str = "normal",
        aspect_ratio: str = "9:16",
        long_edge: int = 480,
        unet_high_noise: str = "wan2.2_t2v_high_noise_14B_Q6_K.gguf",
        unet_low_noise: str = "wan2.2_t2v_low_noise_14B_Q6_K.gguf",
        lora_configs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Build native Text-to-Video DisTorch2 dual-pass Q6_K workflow.

        Pure T2V — no image conditioning, no T2I step. Uses WanImageToVideo
        without start_image to create empty 5D video latent, then dual-pass
        sampling with high noise (steps 0→N) and low noise (steps N→end).

        Same proven architecture as I2V Q6 (build_q6_workflow) but with
        T2V-specific GGUF models and no image loading nodes.

        Args:
            prompt: Text description of the video to generate.
            negative_prompt: What to avoid in the video.
            width: Video width in pixels.
            height: Video height in pixels.
            num_frames: Number of frames (will be adjusted to 4k+1 format).
            fps: Frames per second for output video.
            steps: Total sampling steps (split between high/low noise).
            cfg: Classifier-free guidance scale.
            seed: Random seed (-1 for random).
            output_prefix: Filename prefix for output video.
            high_noise_steps: Step at which to switch from high to low noise model.
            sampler_name: Sampler algorithm (uni_pc, euler, dpmpp_2m, etc.).
            scheduler: Noise scheduler (normal, simple, karras, etc.).
            aspect_ratio: Video aspect ratio (9:16, 16:9, 1:1, etc.).
            long_edge: Resolution for the short side (480, 720).
            unet_high_noise: T2V GGUF model for high noise pass.
            unet_low_noise: T2V GGUF model for low noise pass.
            lora_configs: Optional LoRA configs [{high, low, strength}, ...].
        """
        workflow = copy.deepcopy(WAN22_T2V_Q6_API_WORKFLOW)

        # Wan2.2 requires num_frames in format 4k+1
        k = round((num_frames - 1) / 4)
        k = max(1, k)
        num_frames = 4 * k + 1

        # Node 1 & 2: Unet Loaders - set model names
        workflow["1"]["inputs"]["unet_name"] = unet_high_noise
        workflow["2"]["inputs"]["unet_name"] = unet_low_noise
        logger.info(f"🔧 T2V Unet High: {unet_high_noise}, Low: {unet_low_noise}")

        # Node 9: Positive Prompt
        workflow["9"]["inputs"]["text"] = prompt

        # Node 10: Negative Prompt
        workflow["10"]["inputs"]["text"] = negative_prompt

        # Calculate width/height from aspect ratio and long_edge
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

        if ar_w >= ar_h:
            # Landscape or square
            height = long_edge
            width = int(long_edge * ar_w / ar_h)
        else:
            # Portrait
            width = long_edge
            height = int(long_edge * ar_h / ar_w)

        # Ensure dimensions are multiples of 8 for VAE
        width = (width // 8) * 8
        height = (height // 8) * 8

        # Node 12: WanImageToVideo (T2V mode — no start_image)
        workflow["12"]["inputs"]["width"] = width
        workflow["12"]["inputs"]["height"] = height
        workflow["12"]["inputs"]["length"] = num_frames
        logger.info(
            f"📐 T2V Resolution: {width}x{height} (aspect {aspect_ratio}, long_edge {long_edge})"
        )

        # Handle LoRA loading — same pattern as I2V Q6
        if lora_configs and len(lora_configs) > 0:
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
                elif config.get("high"):
                    low_loras.append(
                        {
                            "name": config["high"],
                            "strength": config.get("strength", 1.0),
                        }
                    )

            # High noise LoRAs
            if high_loras:
                current_high_model = ["7", 0]
                high_node_id = 170
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
                        f"🎨 T2V High LoRA #{i + 1}: {lora['name']} @ {lora['strength']}"
                    )
                workflow["13"]["inputs"]["model"] = current_high_model
            else:
                workflow["13"]["inputs"]["model"] = ["7", 0]

            # Low noise LoRAs
            if low_loras:
                current_low_model = ["8", 0]
                low_node_id = 180
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
                        f"🎨 T2V Low LoRA #{i + 1}: {lora['name']} @ {lora['strength']}"
                    )
                workflow["14"]["inputs"]["model"] = current_low_model
            else:
                workflow["14"]["inputs"]["model"] = ["8", 0]
        else:
            # No LoRAs — connect samplers directly to SageAttn
            workflow["13"]["inputs"]["model"] = ["7", 0]
            workflow["14"]["inputs"]["model"] = ["8", 0]

        # Node 13: Sampler 1 (High Noise) — steps 0 to high_noise_steps
        workflow["13"]["inputs"]["noise_seed"] = seed if seed >= 0 else 42
        workflow["13"]["inputs"]["steps"] = steps
        workflow["13"]["inputs"]["cfg"] = cfg
        workflow["13"]["inputs"]["sampler_name"] = sampler_name
        workflow["13"]["inputs"]["scheduler"] = scheduler
        workflow["13"]["inputs"]["end_at_step"] = high_noise_steps

        # Node 14: Sampler 2 (Low Noise) — steps high_noise_steps to end
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
            lora_info = (
                f", {len(lora_configs)} LoRA{'s' if len(lora_configs) > 1 else ''}"
            )
        logger.info(
            f"🔧 Built T2V DisTorch2 workflow: {aspect_ratio}@{long_edge}, {num_frames}f, "
            f"{steps} steps (switch@{high_noise_steps}), cfg={cfg}{lora_info}"
        )
        return workflow

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
        **kwargs,
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
        # Dynamically map HIGH and LOW noise models based on what's available
        ggufs = self.get_model_names("UnetLoaderGGUF")
        high_model = next(
            (
                m
                for m in ggufs
                if "HIGH" in m.upper()
                and ("wan22EnhancedNSFW" in m or "wan2.2_i2v" in m)
            ),
            "wan22EnhancedNSFW_V2_Q6K_HIGH.gguf",
        )
        low_model = next(
            (
                m
                for m in ggufs
                if "LOW" in m.upper()
                and ("wan22EnhancedNSFW" in m or "wan2.2_i2v" in m)
            ),
            "wan22EnhancedNSFW_V2_Q6K_LOW.gguf",
        )

        model_map = {
            "HIGH": high_model,
            "LOW": low_model,
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

    def build_blockswap_q8_workflow(
        self,
        image_name: str,
        prompt: str,
        negative_prompt: str = "low quality, blurry, distorted, artifacts",
        width: int = 720,
        height: int = 1280,
        num_frames: int = 121,
        fps: int = 24,
        steps: int = 8,
        cfg: float = 1.0,
        seed: int = -1,
        output_prefix: str = "oelala_blockswap_q8",
        high_noise_steps: int = 4,
        shift: float = 9.0,
        nag_scale: float = 11.0,
        nag_alpha: float = 0.25,
        nag_tau: float = 2.373,
        enhance_weight: float = 1.0,
        enable_upscale: bool = False,
        enable_interpolation: bool = False,
        enable_florence2: bool = True,
        lora_configs: Optional[List[Dict[str, Any]]] = None,
        aspect_ratio: str = "9:16",
        long_edge: int = 720,
    ) -> Optional[Dict[str, Any]]:
        """
        Build BlockSwap Q8 experimental workflow.

        Uses Q8_0 GGUF models with BlockSwap VRAM optimization, Lightning LoRA,
        NAG, TorchCompile, EnhanceAVideo, CFGZeroStar, and optional Florence2
        auto-captioning + post-processing (upscale/interpolation).

        Key differences from standard Q6 mode:
        - Q8_0 models (higher quality, larger)
        - BlockSwap VRAM reduction (enables Q8 on limited VRAM)
        - Lightning LoRA always-on (4-8 step fast generation)
        - NAG (Normalized Attention Guidance) for better coherence
        - TorchCompile JIT for speed
        - MagCache for caching optimization
        - Florence2 auto-captioning of input image
        - EnhanceAVideo quality enhancement
        - CFGZeroStar guidance trick
        - Optional 4x upscale + RIFE 2x frame interpolation

        Args:
            high_noise_steps: Steps where high noise model switches to low noise
            shift: ModelSamplingSD3 shift parameter (higher = more creative)
            nag_scale: NAG guidance scale
            nag_alpha: NAG alpha parameter
            nag_tau: NAG tau parameter
            enhance_weight: EnhanceAVideo weight (0 to disable)
            enable_upscale: Enable 4x upscale post-processing
            enable_interpolation: Enable RIFE 2x frame interpolation
            enable_florence2: Enable Florence2 auto-captioning
            lora_configs: Additional LoRAs [{high, low, strength}, ...]
        """
        # Load template
        workflow = load_workflow_from_file(
            "ImageToVideo/wan22_i2v_blockswap_q8_api.json"
        )
        if not workflow:
            logger.error("❌ Failed to load BlockSwap Q8 workflow template")
            return None

        workflow = copy.deepcopy(workflow)

        # Wan2.2 requires num_frames in format 4k+1
        k = round((num_frames - 1) / 4)
        k = max(1, k)
        num_frames = 4 * k + 1

        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        # ─── Calculate dimensions from aspect ratio ───
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

        if ar_w >= ar_h:
            height = long_edge
            width = int(long_edge * ar_w / ar_h)
        else:
            width = long_edge
            height = int(long_edge * ar_h / ar_w)

        width = (width // 8) * 8
        height = (height // 8) * 8

        logger.info(
            f"🧪 Building BlockSwap Q8 workflow: {width}x{height}, {num_frames}f, "
            f"{steps} steps, cfg={cfg}, shift={shift}, NAG={nag_scale}"
        )

        # ─── Input Image ───
        workflow["88"]["inputs"]["image"] = image_name

        # ─── Image Resize ───
        workflow["401"]["inputs"]["width"] = width
        workflow["401"]["inputs"]["height"] = height

        # ─── WanImageToVideo ───
        workflow["464"]["inputs"]["width"] = width
        workflow["464"]["inputs"]["height"] = height
        workflow["464"]["inputs"]["length"] = num_frames

        # ─── Florence2 captioning ───
        if enable_florence2:
            # Florence2 captions the image, output goes through text replacements
            # to StringConcatenate which combines with user prompt
            workflow["451"]["inputs"]["string_b"] = prompt
        else:
            # Bypass Florence2: feed prompt directly to positive encode
            workflow["462"]["inputs"]["text"] = prompt
            # Remove Florence2 nodes to save VRAM
            for nid in list(workflow.keys()):
                ct = workflow[nid].get("class_type", "")
                if ct in (
                    "DownloadAndLoadFlorence2Model",
                    "Florence2Run",
                    "Text Find and Replace",
                    "StringConcatenate",
                ):
                    del workflow[nid]
            # Connect positive encode directly to prompt string
            workflow["462"]["inputs"].pop("text", None)  # Remove connection
            workflow["462"]["inputs"]["text"] = prompt

        # ─── Negative Prompt ───
        workflow["463"]["inputs"]["text"] = negative_prompt

        # ─── Seed ───
        workflow["73"]["inputs"]["noise_seed"] = seed
        workflow["466"]["inputs"]["noise_seed"] = seed
        workflow["465"]["inputs"]["noise_seed"] = seed

        # ─── Sampling Parameters ───
        workflow["466"]["inputs"]["steps"] = steps
        workflow["466"]["inputs"]["cfg"] = cfg
        workflow["466"]["inputs"]["end_at_step"] = high_noise_steps
        workflow["465"]["inputs"]["steps"] = steps
        workflow["465"]["inputs"]["cfg"] = cfg
        workflow["465"]["inputs"]["start_at_step"] = high_noise_steps

        # ─── Model Sampling Shift ───
        workflow["467"]["inputs"]["shift"] = shift
        workflow["468"]["inputs"]["shift"] = shift

        # ─── NAG Parameters ───
        workflow["485"]["inputs"]["nag_scale"] = nag_scale
        workflow["485"]["inputs"]["nag_alpha"] = nag_alpha
        workflow["485"]["inputs"]["nag_tau"] = nag_tau
        workflow["486"]["inputs"]["nag_scale"] = nag_scale
        workflow["486"]["inputs"]["nag_alpha"] = nag_alpha
        workflow["486"]["inputs"]["nag_tau"] = nag_tau

        # ─── EnhanceAVideo ───
        workflow["481"]["inputs"]["weight"] = enhance_weight
        workflow["482"]["inputs"]["weight"] = enhance_weight

        # ─── LoRAs via Power Lora Loader (rgthree) ───
        # Nodes 416 (high noise) and 471 (low noise) are Power Lora Loader nodes
        # with up to 6 slots each. Template has them all off by default.
        if lora_configs and len(lora_configs) > 0:
            for i, config in enumerate(lora_configs[:6], 1):
                # High noise LoRA slot
                high_name = config.get("high", "")
                if high_name:
                    workflow["416"]["inputs"][f"lora_{i}"] = {
                        "on": True,
                        "lora": high_name,
                        "strength": config.get("strength", 1.0),
                    }
                    logger.info(
                        f"🎨 BS-Q8 High LoRA #{i}: {high_name} @ {config.get('strength', 1.0)}"
                    )

                # Low noise LoRA slot (falls back to high if not specified)
                low_name = config.get("low", high_name)
                if low_name:
                    workflow["471"]["inputs"][f"lora_{i}"] = {
                        "on": True,
                        "lora": low_name,
                        "strength": config.get("strength", 1.0),
                    }
        # Chain: UnetLoader(495/496) → PowerLora(416/471) → EnhanceAVideo(481/482)

        # ─── Output ───
        workflow["398"]["inputs"]["frame_rate"] = fps
        workflow["398"]["inputs"]["filename_prefix"] = output_prefix

        # ─── Post-processing (upscale + interpolation) ───
        if not enable_upscale:
            # Remove upscale nodes
            for nid in ["384", "385", "418", "419"]:
                if nid in workflow:
                    del workflow[nid]

        if not enable_interpolation:
            # Remove interpolation nodes
            for nid in ["431", "433"]:
                if nid in workflow:
                    del workflow[nid]
        else:
            # Set interpolated output frame rate
            workflow["433"]["inputs"]["frame_rate"] = fps * 2

        if enable_upscale:
            workflow["419"]["inputs"]["frame_rate"] = fps
            workflow["419"]["inputs"]["filename_prefix"] = f"{output_prefix}_upscaled"

        if enable_interpolation:
            workflow["433"]["inputs"]["filename_prefix"] = (
                f"{output_prefix}_interpolated"
            )

        lora_info = ""
        if lora_configs and len(lora_configs) > 0:
            lora_info = f", {len(lora_configs)} extra LoRAs"
        pp_info = []
        if enable_upscale:
            pp_info.append("upscale")
        if enable_interpolation:
            pp_info.append("interpolate")
        if enable_florence2:
            pp_info.append("florence2")
        pp_str = f", post=[{','.join(pp_info)}]" if pp_info else ""

        logger.info(
            f"🧪 Built BlockSwap Q8: {width}x{height}, {num_frames}f@{fps}fps, "
            f"{steps} steps (switch@{high_noise_steps}), cfg={cfg}, "
            f"shift={shift}, NAG={nag_scale}{lora_info}{pp_str}"
        )
        return workflow

    def build_distorch2_q8_workflow(
        self,
        image_name: str,
        prompt: str,
        negative_prompt: str = "low quality, blurry, distorted, artifacts",
        width: int = 0,
        height: int = 0,
        num_frames: int = 161,
        fps: int = 16,
        steps: int = 8,
        cfg: float = 1.0,
        seed: int = -1,
        output_prefix: str = "oelala_distorch2_q8",
        high_noise_steps: int = 4,
        shift: float = 9.0,
        nag_scale: float = 11.0,
        nag_alpha: float = 0.25,
        nag_tau: float = 2.373,
        enhance_weight: float = 1.0,
        enable_upscale: bool = False,
        enable_interpolation: bool = False,
        enable_florence2: bool = True,
        lora_configs: Optional[List[Dict[str, Any]]] = None,
        aspect_ratio: str = "9:16",
        long_edge: int = 480,
        distorch2_alloc: str = "cuda:0,10gb;cuda:1,14.5gb;cpu,*",
    ) -> Optional[Dict[str, Any]]:
        """
        Build DisTorch2 Q8 experimental workflow.

        Same processing chain as BlockSwap Q8 (NAG, EnhanceAVideo, CFGZeroStar,
        TorchCompile, Florence2) but uses DisTorch2 multi-GPU distribution
        instead of BlockSwap VRAM offloading. No forced Lightning LoRA —
        all LoRAs are user-selectable.

        Key differences from BlockSwap Q8:
        - DisTorch2 multi-GPU loaders (UnetGGUFAdvanced, CLIP, VAE)
        - No wanBlockSwap nodes (DisTorch2 handles VRAM)
        - No forced Lightning LoRA (user chooses all LoRAs)
        - Configurable allocation string

        Args:
            distorch2_alloc: DisTorch2 expert_mode_allocations string
            (all other args same as build_blockswap_q8_workflow)
        """
        workflow = load_workflow_from_file(
            "ImageToVideo/wan22_i2v_distorch2_q8_api.json"
        )
        if not workflow:
            logger.error("❌ Failed to load DisTorch2 Q8 workflow template")
            return None

        workflow = copy.deepcopy(workflow)

        # Wan2.2 requires num_frames in format 4k+1
        k = round((num_frames - 1) / 4)
        k = max(1, k)
        num_frames = 4 * k + 1

        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        # ─── Calculate dimensions from aspect ratio ───
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

        if ar_w >= ar_h:
            height = long_edge
            width = int(long_edge * ar_w / ar_h)
        else:
            width = long_edge
            height = int(long_edge * ar_h / ar_w)

        width = (width // 8) * 8
        height = (height // 8) * 8

        logger.info(
            f"🧪 Building DisTorch2 Q8 workflow: {width}x{height}, {num_frames}f, "
            f"{steps} steps, cfg={cfg}, shift={shift}, NAG={nag_scale}, alloc={distorch2_alloc}"
        )

        # ─── DisTorch2 allocation on all loader nodes ───
        for loader_id in ["495", "496", "460", "461"]:
            if loader_id in workflow:
                workflow[loader_id]["inputs"]["expert_mode_allocations"] = (
                    distorch2_alloc
                )

        # ─── Input Image ───
        workflow["88"]["inputs"]["image"] = image_name

        # ─── Image Resize ───
        workflow["401"]["inputs"]["width"] = width
        workflow["401"]["inputs"]["height"] = height

        # ─── WanImageToVideo ───
        workflow["464"]["inputs"]["width"] = width
        workflow["464"]["inputs"]["height"] = height
        workflow["464"]["inputs"]["length"] = num_frames

        # ─── Florence2 captioning ───
        if enable_florence2:
            workflow["451"]["inputs"]["string_b"] = prompt
        else:
            workflow["462"]["inputs"]["text"] = prompt
            for nid in list(workflow.keys()):
                ct = workflow[nid].get("class_type", "")
                if ct in (
                    "DownloadAndLoadFlorence2Model",
                    "Florence2Run",
                    "Text Find and Replace",
                    "StringConcatenate",
                ):
                    del workflow[nid]
            workflow["462"]["inputs"].pop("text", None)
            workflow["462"]["inputs"]["text"] = prompt

        # ─── Negative Prompt ───
        workflow["463"]["inputs"]["text"] = negative_prompt

        # ─── Seed ───
        workflow["73"]["inputs"]["noise_seed"] = seed
        workflow["466"]["inputs"]["noise_seed"] = seed
        workflow["465"]["inputs"]["noise_seed"] = seed

        # ─── Sampling Parameters ───
        workflow["466"]["inputs"]["steps"] = steps
        workflow["466"]["inputs"]["cfg"] = cfg
        workflow["466"]["inputs"]["end_at_step"] = high_noise_steps
        workflow["465"]["inputs"]["steps"] = steps
        workflow["465"]["inputs"]["cfg"] = cfg
        workflow["465"]["inputs"]["start_at_step"] = high_noise_steps

        # ─── Model Sampling Shift ───
        workflow["467"]["inputs"]["shift"] = shift
        workflow["468"]["inputs"]["shift"] = shift

        # ─── NAG Parameters ───
        workflow["485"]["inputs"]["nag_scale"] = nag_scale
        workflow["485"]["inputs"]["nag_alpha"] = nag_alpha
        workflow["485"]["inputs"]["nag_tau"] = nag_tau
        workflow["486"]["inputs"]["nag_scale"] = nag_scale
        workflow["486"]["inputs"]["nag_alpha"] = nag_alpha
        workflow["486"]["inputs"]["nag_tau"] = nag_tau

        # ─── EnhanceAVideo ───
        workflow["481"]["inputs"]["weight"] = enhance_weight
        workflow["482"]["inputs"]["weight"] = enhance_weight

        # ─── LoRAs via Power Lora Loader (rgthree) ───
        # Nodes 416 (high noise) and 471 (low noise) are Power Lora Loader nodes
        # with up to 6 slots each. Template has them all off by default.
        if lora_configs and len(lora_configs) > 0:
            for i, config in enumerate(lora_configs[:6], 1):
                # High noise LoRA slot
                high_name = config.get("high", "")
                if high_name:
                    workflow["416"]["inputs"][f"lora_{i}"] = {
                        "on": True,
                        "lora": high_name,
                        "strength": config.get("strength", 1.0),
                    }
                    logger.info(
                        f"🎨 DT2-Q8 High LoRA #{i}: {high_name} @ {config.get('strength', 1.0)}"
                    )

                # Low noise LoRA slot (falls back to high if not specified)
                low_name = config.get("low", high_name)
                if low_name:
                    workflow["471"]["inputs"][f"lora_{i}"] = {
                        "on": True,
                        "lora": low_name,
                        "strength": config.get("strength", 1.0),
                    }
        # Chain: UnetLoader(495/496) → PowerLora(416/471) → EnhanceAVideo(481/482)

        # ─── Output ───
        workflow["398"]["inputs"]["frame_rate"] = fps
        workflow["398"]["inputs"]["filename_prefix"] = output_prefix

        # ─── Post-processing (upscale + interpolation) ───
        if not enable_upscale:
            for nid in ["384", "385", "418", "419"]:
                if nid in workflow:
                    del workflow[nid]

        if not enable_interpolation:
            for nid in ["431", "433"]:
                if nid in workflow:
                    del workflow[nid]
        else:
            workflow["433"]["inputs"]["frame_rate"] = fps * 2

        if enable_upscale:
            workflow["419"]["inputs"]["frame_rate"] = fps
            workflow["419"]["inputs"]["filename_prefix"] = f"{output_prefix}_upscaled"

        if enable_interpolation:
            workflow["433"]["inputs"]["filename_prefix"] = (
                f"{output_prefix}_interpolated"
            )

        lora_info = ""
        if lora_configs and len(lora_configs) > 0:
            lora_info = f", {len(lora_configs)} LoRAs"
        pp_info = []
        if enable_upscale:
            pp_info.append("upscale")
        if enable_interpolation:
            pp_info.append("interpolate")
        if enable_florence2:
            pp_info.append("florence2")
        pp_str = f", post=[{','.join(pp_info)}]" if pp_info else ""

        logger.info(
            f"🧪 Built DisTorch2 Q8: {width}x{height}, {num_frames}f@{fps}fps, "
            f"{steps} steps (switch@{high_noise_steps}), cfg={cfg}, "
            f"shift={shift}, NAG={nag_scale}, alloc={distorch2_alloc}{lora_info}{pp_str}"
        )
        return workflow

    def build_ultra_q8_workflow(
        self,
        image_name: str,
        prompt: str,
        negative_prompt: str = "low quality, blurry, distorted, artifacts",
        width: int = 0,
        height: int = 0,
        num_frames: int = 161,
        fps: int = 16,
        steps: int = 8,
        cfg: float = 1.0,
        seed: int = -1,
        output_prefix: str = "oelala_ultra_q8",
        high_noise_steps: int = 4,
        shift: float = 9.0,
        nag_scale: float = 11.0,
        nag_alpha: float = 0.25,
        nag_tau: float = 2.373,
        enhance_weight: float = 1.0,
        enable_upscale: bool = False,
        enable_interpolation: bool = False,
        enable_florence2: bool = True,
        lora_configs: Optional[List[Dict[str, Any]]] = None,
        aspect_ratio: str = "9:16",
        long_edge: int = 576,
    ) -> Optional[Dict[str, Any]]:
        """
        Build Ultra Q8 workflow — max VRAM + unlimited CPU RAM.

        Uses DisTorch2 with optimized allocation: 3060 as dedicated model cache
        (11GB), 5060 Ti 100% free for compute (~15.5GB), CPU for overflow.
        This keeps the compute GPU maximally free for activations/latents,
        allowing higher resolutions and more frames than other modes.

        Memory strategy:
        - UNet: cuda:0(3060)=11GB, cuda:1(5060Ti)=0.5GB, cpu=remainder
        - CLIP: Fully on CPU (saves GPU VRAM)
        - VAE: On compute device cuda:1 (small, needed for decode)

        Same processing chain as BlockSwap/DisTorch2 Q8:
        NAG, EnhanceAVideo, CFGZeroStar, TorchCompile, Florence2,
        Lightning LoRA built-in, optional upscale/interpolation.
        """
        workflow = load_workflow_from_file("ImageToVideo/wan22_i2v_ultra_q8_api.json")
        if not workflow:
            logger.error("❌ Failed to load Ultra Q8 workflow template")
            return None

        workflow = copy.deepcopy(workflow)

        # Wan2.2 requires num_frames in format 4k+1
        k = round((num_frames - 1) / 4)
        k = max(1, k)
        num_frames = 4 * k + 1

        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        # ─── Calculate dimensions from aspect ratio ───
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

        if ar_w >= ar_h:
            height = long_edge
            width = int(long_edge * ar_w / ar_h)
        else:
            width = long_edge
            height = int(long_edge * ar_h / ar_w)

        width = (width // 8) * 8
        height = (height // 8) * 8

        logger.info(
            f"⚡ Building Ultra Q8 workflow: {width}x{height}, {num_frames}f, "
            f"{steps} steps, cfg={cfg}, shift={shift}, NAG={nag_scale}"
        )

        # ─── Input Image ───
        workflow["88"]["inputs"]["image"] = image_name

        # ─── Image Resize ───
        workflow["401"]["inputs"]["width"] = width
        workflow["401"]["inputs"]["height"] = height

        # ─── WanImageToVideo ───
        workflow["464"]["inputs"]["width"] = width
        workflow["464"]["inputs"]["height"] = height
        workflow["464"]["inputs"]["length"] = num_frames

        # ─── Florence2 captioning ───
        if enable_florence2:
            workflow["451"]["inputs"]["string_b"] = prompt
        else:
            workflow["462"]["inputs"]["text"] = prompt
            for nid in list(workflow.keys()):
                ct = workflow[nid].get("class_type", "")
                if ct in (
                    "DownloadAndLoadFlorence2Model",
                    "Florence2Run",
                    "Text Find and Replace",
                    "StringConcatenate",
                ):
                    del workflow[nid]
            workflow["462"]["inputs"].pop("text", None)
            workflow["462"]["inputs"]["text"] = prompt

        # ─── Negative Prompt ───
        workflow["463"]["inputs"]["text"] = negative_prompt

        # ─── Seed ───
        workflow["73"]["inputs"]["noise_seed"] = seed
        workflow["466"]["inputs"]["noise_seed"] = seed
        workflow["465"]["inputs"]["noise_seed"] = seed

        # ─── Sampling Parameters ───
        workflow["466"]["inputs"]["steps"] = steps
        workflow["466"]["inputs"]["cfg"] = cfg
        workflow["466"]["inputs"]["end_at_step"] = high_noise_steps
        workflow["465"]["inputs"]["steps"] = steps
        workflow["465"]["inputs"]["cfg"] = cfg
        workflow["465"]["inputs"]["start_at_step"] = high_noise_steps

        # ─── Model Sampling Shift ───
        workflow["467"]["inputs"]["shift"] = shift
        workflow["468"]["inputs"]["shift"] = shift

        # ─── NAG Parameters ───
        workflow["485"]["inputs"]["nag_scale"] = nag_scale
        workflow["485"]["inputs"]["nag_alpha"] = nag_alpha
        workflow["485"]["inputs"]["nag_tau"] = nag_tau
        workflow["486"]["inputs"]["nag_scale"] = nag_scale
        workflow["486"]["inputs"]["nag_alpha"] = nag_alpha
        workflow["486"]["inputs"]["nag_tau"] = nag_tau

        # ─── EnhanceAVideo ───
        workflow["481"]["inputs"]["weight"] = enhance_weight
        workflow["482"]["inputs"]["weight"] = enhance_weight

        # ─── LoRAs via Power Lora Loader (rgthree) ───
        if lora_configs and len(lora_configs) > 0:
            for i, config in enumerate(lora_configs[:6], 1):
                high_name = config.get("high", "")
                if high_name:
                    workflow["416"]["inputs"][f"lora_{i}"] = {
                        "on": True,
                        "lora": high_name,
                        "strength": config.get("strength", 1.0),
                    }
                    logger.info(
                        f"🎨 Ultra-Q8 High LoRA #{i}: {high_name} @ {config.get('strength', 1.0)}"
                    )

                low_name = config.get("low", high_name)
                if low_name:
                    workflow["471"]["inputs"][f"lora_{i}"] = {
                        "on": True,
                        "lora": low_name,
                        "strength": config.get("strength", 1.0),
                    }

        # ─── Output ───
        workflow["398"]["inputs"]["frame_rate"] = fps
        workflow["398"]["inputs"]["filename_prefix"] = output_prefix

        # ─── Post-processing (upscale + interpolation) ───
        if not enable_upscale:
            for nid in ["384", "385", "418", "419"]:
                if nid in workflow:
                    del workflow[nid]

        if not enable_interpolation:
            for nid in ["431", "433"]:
                if nid in workflow:
                    del workflow[nid]
        else:
            workflow["433"]["inputs"]["frame_rate"] = fps * 2

        if enable_upscale:
            workflow["419"]["inputs"]["frame_rate"] = fps
            workflow["419"]["inputs"]["filename_prefix"] = f"{output_prefix}_upscaled"

        if enable_interpolation:
            workflow["433"]["inputs"]["filename_prefix"] = (
                f"{output_prefix}_interpolated"
            )

        lora_info = ""
        if lora_configs and len(lora_configs) > 0:
            lora_info = f", {len(lora_configs)} LoRAs"
        pp_info = []
        if enable_upscale:
            pp_info.append("upscale")
        if enable_interpolation:
            pp_info.append("interpolate")
        if enable_florence2:
            pp_info.append("florence2")
        pp_str = f", post=[{','.join(pp_info)}]" if pp_info else ""

        logger.info(
            f"⚡ Built Ultra Q8: {width}x{height}, {num_frames}f@{fps}fps, "
            f"{steps} steps (switch@{high_noise_steps}), cfg={cfg}, "
            f"shift={shift}, NAG={nag_scale}{lora_info}{pp_str}"
        )
        return workflow

    def build_cloud_wan22_i2v_workflow(
        self,
        image_name: str,
        prompt: str,
        negative_prompt: str = "low quality, blurry, distorted, artifacts, flickering, jitter",
        width: int = 1280,
        height: int = 720,
        num_frames: int = 81,
        fps: int = 16,
        steps: int = 15,
        cfg: float = 3.0,
        seed: int = -1,
        output_prefix: str = "oelala_cloud_wan22",
        high_noise_steps: int = 8,
        shift: float = 8.0,
        sampler_name: str = "dpmpp_2m",
        scheduler: str = "beta",
        lora_configs: Optional[List[Dict[str, Any]]] = None,
        aspect_ratio: str = "9:16",
        long_edge: int = 720,
        diffusion_model_high: str = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        diffusion_model_low: str = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        text_encoder: str = "umt5_xxl_fp16.safetensors",
        vae_model: str = "wan_2.1_vae.safetensors",
        clip_vision: str = "clip_vision_h.safetensors",
        # Legacy compat: if caller passes single diffusion_model, use it for both
        diffusion_model: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Build Cloud Wan22 I2V workflow — Wan 2.2 fp8_scaled for cloud GPUs (48GB+).

        Uses dedicated high/low noise models from Wan 2.2 for better temporal
        coherence and motion quality compared to Wan 2.1's single-model approach.
        fp8_scaled precision is near-lossless vs bf16, fits on 48GB GPUs.

        Designed for RunPod serverless with A6000/A40/L40S GPUs. Uses native
        ComfyUI UNETLoader nodes. Dual-pass sampling with dedicated models
        for high and low noise phases.

        Key features:
        - Wan 2.2 dedicated high/low noise models (better than 2.1 unified)
        - fp8_scaled precision (~14.3GB each, 28.6GB total, fits 48GB GPU)
        - Wan 2.2 VAE (improved decoder, 1.41GB)
        - Full precision text encoder (fp16 UMT5-XXL)
        - 15 steps with dpmpp_2m/beta (optimal quality/speed)
        - LoRA support with high/low noise split
        - No multi-GPU nodes (single powerful GPU)
        """
        # Wan2.2/2.1 requires num_frames in format 4k+1
        k = round((num_frames - 1) / 4)
        k = max(1, k)
        num_frames = 4 * k + 1

        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        # ─── Calculate dimensions from aspect ratio ───
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

        if ar_w >= ar_h:
            height = long_edge
            width = int(long_edge * ar_w / ar_h)
        else:
            width = long_edge
            height = int(long_edge * ar_h / ar_w)

        width = (width // 8) * 8
        height = (height // 8) * 8

        logger.info(
            f"☁️ Building Cloud Wan22 I2V: {width}x{height}, {num_frames}f, "
            f"{steps} steps, cfg={cfg}, shift={shift}, sampler={sampler_name}, "
            f"scheduler={scheduler}"
        )

        # ─── Build workflow dict (native ComfyUI nodes) ───
        workflow = {}

        # Node 1: Load input image
        workflow["1"] = {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        }

        # Node 2: Image resize to target resolution
        workflow["2"] = {
            "class_type": "ImageResizeKJv2",
            "inputs": {
                "width": width,
                "height": height,
                "upscale_method": "lanczos",
                "keep_proportion": "stretch",
                "divisible_by": 8,
                "crop_position": "center",
                "pad_color": "0, 0, 0",
                "image": ["1", 0],
            },
        }

        # Node 3: UNET Loader — HIGH NOISE model (Wan 2.2 fp8_scaled)
        workflow["3"] = {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": diffusion_model_high,
                "weight_dtype": "default",
            },
        }

        # Node 14: UNET Loader — LOW NOISE model (Wan 2.2 fp8_scaled)
        workflow["14"] = {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": diffusion_model_low,
                "weight_dtype": "default",
            },
        }

        # Node 4: CLIP Loader — full precision text encoder
        workflow["4"] = {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": text_encoder,
                "type": "wan",
            },
        }

        # Node 5: VAE Loader
        workflow["5"] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_model},
        }

        # Node 6: CLIP Vision Loader (for I2V conditioning)
        workflow["6"] = {
            "class_type": "CLIPVisionLoader",
            "inputs": {"clip_name": clip_vision},
        }

        # Node 12: CLIP Vision Encode (CLIP_VISION + IMAGE → CLIP_VISION_OUTPUT)
        workflow["12"] = {
            "class_type": "CLIPVisionEncode",
            "inputs": {
                "clip_vision": ["6", 0],
                "image": ["2", 0],
                "crop": "center",
            },
        }

        # Node 7: Positive prompt encoding
        workflow["7"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["4", 0]},
        }

        # Node 8: Negative prompt encoding
        workflow["8"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["4", 0]},
        }

        # Node 9: WanImageToVideo — I2V conditioning
        workflow["9"] = {
            "class_type": "WanImageToVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1,
                "positive": ["7", 0],
                "negative": ["8", 0],
                "vae": ["5", 0],
                "clip_vision_output": ["12", 0],
                "start_image": ["2", 0],
            },
        }

        # Node 10: ModelSamplingSD3 for high noise model
        workflow["10"] = {
            "class_type": "ModelSamplingSD3",
            "inputs": {"shift": shift, "model": ["3", 0]},
        }

        # Node 11: ModelSamplingSD3 for low noise model (SEPARATE model)
        workflow["11"] = {
            "class_type": "ModelSamplingSD3",
            "inputs": {"shift": shift, "model": ["14", 0]},
        }

        # ─── LoRA Loading (supports high/low split) ───
        # High noise chain starts from node 10
        current_high_model = ["10", 0]
        # Low noise chain starts from node 11
        current_low_model = ["11", 0]

        if lora_configs and len(lora_configs) > 0:
            high_node_id = 100
            low_node_id = 150
            for i, config in enumerate(lora_configs):
                strength = config.get("strength", 1.0)

                # High noise LoRA
                high_name = config.get("high", "")
                if high_name:
                    nid = str(high_node_id + i)
                    workflow[nid] = {
                        "class_type": "LoraLoaderModelOnly",
                        "inputs": {
                            "lora_name": high_name,
                            "strength_model": strength,
                            "model": current_high_model,
                        },
                    }
                    current_high_model = [nid, 0]
                    logger.info(
                        f"🎨 Cloud Wan22 High LoRA #{i + 1}: {high_name} @ {strength}"
                    )

                # Low noise LoRA (falls back to high if not specified)
                low_name = config.get("low", high_name)
                if low_name:
                    nid = str(low_node_id + i)
                    workflow[nid] = {
                        "class_type": "LoraLoaderModelOnly",
                        "inputs": {
                            "lora_name": low_name,
                            "strength_model": strength,
                            "model": current_low_model,
                        },
                    }
                    current_low_model = [nid, 0]
                    logger.info(
                        f"🎨 Cloud Wan22 Low LoRA #{i + 1}: {low_name} @ {strength}"
                    )

        # Node 20: KSamplerAdvanced — Pass 1 (High Noise)
        workflow["20"] = {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "add_noise": "enable",
                "noise_seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "start_at_step": 0,
                "end_at_step": high_noise_steps,
                "return_with_leftover_noise": "enable",
                "model": current_high_model,
                "positive": ["9", 0],
                "negative": ["9", 1],
                "latent_image": ["9", 2],
            },
        }

        # Node 21: KSamplerAdvanced — Pass 2 (Low Noise)
        workflow["21"] = {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "add_noise": "disable",
                "noise_seed": 0,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "start_at_step": high_noise_steps,
                "end_at_step": 10000,
                "return_with_leftover_noise": "disable",
                "model": current_low_model,
                "positive": ["9", 0],
                "negative": ["9", 1],
                "latent_image": ["20", 0],
            },
        }

        # Node 22: VAE Decode
        workflow["22"] = {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["21", 0], "vae": ["5", 0]},
        }

        # Node 23: Save Video (VHS_VideoCombine)
        workflow["23"] = {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": output_prefix,
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 17,  # Higher quality than local (19)
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
                "images": ["22", 0],
            },
        }

        lora_info = ""
        if lora_configs and len(lora_configs) > 0:
            lora_info = (
                f", {len(lora_configs)} LoRA{'s' if len(lora_configs) > 1 else ''}"
            )
        logger.info(
            f"☁️ Built Cloud Wan22 I2V: {width}x{height}, {num_frames}f@{fps}fps, "
            f"{steps} steps (switch@{high_noise_steps}), cfg={cfg}, "
            f"shift={shift}, {sampler_name}/{scheduler}{lora_info}"
        )
        return workflow

    def build_cloud_wan22_t2v_workflow(
        self,
        prompt: str,
        negative_prompt: str = "low quality, blurry, distorted, artifacts, flickering, jitter",
        width: int = 1280,
        height: int = 720,
        num_frames: int = 81,
        fps: int = 16,
        steps: int = 15,
        cfg: float = 3.0,
        seed: int = -1,
        output_prefix: str = "oelala_cloud_wan22_t2v",
        high_noise_steps: int = 8,
        shift: float = 8.0,
        sampler_name: str = "dpmpp_2m",
        scheduler: str = "beta",
        lora_configs: Optional[List[Dict[str, Any]]] = None,
        aspect_ratio: str = "9:16",
        long_edge: int = 720,
        diffusion_model_high: str = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        diffusion_model_low: str = "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        text_encoder: str = "umt5_xxl_fp16.safetensors",
        vae_model: str = "wan_2.1_vae.safetensors",
        # Legacy compat: if caller passes single diffusion_model, use it for both
        diffusion_model: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Build Cloud Wan22 T2V workflow — Wan 2.2 fp8_scaled for cloud GPUs (48GB+).

        Same architecture as Cloud Wan22 I2V but without image conditioning.
        Uses dedicated high/low noise T2V models from Wan 2.2 for better
        temporal coherence. fp8_scaled precision (~14.3GB each, fits 48GB GPU).
        """
        # Wan2.2/2.1 requires num_frames in format 4k+1
        k = round((num_frames - 1) / 4)
        k = max(1, k)
        num_frames = 4 * k + 1

        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        # Legacy compat: single diffusion_model param → use for both
        if diffusion_model is not None:
            diffusion_model_high = diffusion_model
            diffusion_model_low = diffusion_model

        # ─── Calculate dimensions from aspect ratio ───
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

        if ar_w >= ar_h:
            height = long_edge
            width = int(long_edge * ar_w / ar_h)
        else:
            width = long_edge
            height = int(long_edge * ar_h / ar_w)

        width = (width // 8) * 8
        height = (height // 8) * 8

        logger.info(
            f"☁️ Building Cloud Wan22 T2V: {width}x{height}, {num_frames}f, "
            f"{steps} steps, cfg={cfg}, shift={shift}"
        )

        workflow = {}

        # Node 1: UNET Loader — HIGH NOISE model (Wan 2.2 fp8_scaled)
        workflow["1"] = {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": diffusion_model_high,
                "weight_dtype": "default",
            },
        }

        # Node 9: UNET Loader — LOW NOISE model (Wan 2.2 fp8_scaled)
        workflow["9"] = {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": diffusion_model_low,
                "weight_dtype": "default",
            },
        }

        # Node 2: CLIP Loader — full precision text encoder
        workflow["2"] = {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": text_encoder,
                "type": "wan",
            },
        }

        # Node 3: VAE Loader
        workflow["3"] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_model},
        }

        # Node 4: Positive prompt encoding
        workflow["4"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["2", 0]},
        }

        # Node 5: Negative prompt encoding
        workflow["5"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["2", 0]},
        }

        # Node 6: WanImageToVideo — T2V mode (NO start_image, NO clip_vision)
        workflow["6"] = {
            "class_type": "WanImageToVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1,
                "positive": ["4", 0],
                "negative": ["5", 0],
                "vae": ["3", 0],
            },
        }

        # Node 7: ModelSamplingSD3 for high noise
        workflow["7"] = {
            "class_type": "ModelSamplingSD3",
            "inputs": {"shift": shift, "model": ["1", 0]},
        }

        # Node 8: ModelSamplingSD3 for low noise (SEPARATE model)
        workflow["8"] = {
            "class_type": "ModelSamplingSD3",
            "inputs": {"shift": shift, "model": ["9", 0]},
        }

        # ─── LoRA Loading ───
        current_high_model = ["7", 0]
        current_low_model = ["8", 0]

        if lora_configs and len(lora_configs) > 0:
            high_node_id = 100
            low_node_id = 150
            for i, config in enumerate(lora_configs):
                strength = config.get("strength", 1.0)
                high_name = config.get("high", "")
                if high_name:
                    nid = str(high_node_id + i)
                    workflow[nid] = {
                        "class_type": "LoraLoaderModelOnly",
                        "inputs": {
                            "lora_name": high_name,
                            "strength_model": strength,
                            "model": current_high_model,
                        },
                    }
                    current_high_model = [nid, 0]
                low_name = config.get("low", high_name)
                if low_name:
                    nid = str(low_node_id + i)
                    workflow[nid] = {
                        "class_type": "LoraLoaderModelOnly",
                        "inputs": {
                            "lora_name": low_name,
                            "strength_model": strength,
                            "model": current_low_model,
                        },
                    }
                    current_low_model = [nid, 0]

        # Node 10: KSamplerAdvanced — Pass 1 (High Noise)
        workflow["10"] = {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "add_noise": "enable",
                "noise_seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "start_at_step": 0,
                "end_at_step": high_noise_steps,
                "return_with_leftover_noise": "enable",
                "model": current_high_model,
                "positive": ["6", 0],
                "negative": ["6", 1],
                "latent_image": ["6", 2],
            },
        }

        # Node 11: KSamplerAdvanced — Pass 2 (Low Noise)
        workflow["11"] = {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "add_noise": "disable",
                "noise_seed": 0,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "start_at_step": high_noise_steps,
                "end_at_step": 10000,
                "return_with_leftover_noise": "disable",
                "model": current_low_model,
                "positive": ["6", 0],
                "negative": ["6", 1],
                "latent_image": ["10", 0],
            },
        }

        # Node 12: VAE Decode
        workflow["12"] = {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["11", 0], "vae": ["3", 0]},
        }

        # Node 13: Save Video
        workflow["13"] = {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": output_prefix,
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 17,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
                "images": ["12", 0],
            },
        }

        lora_info = ""
        if lora_configs and len(lora_configs) > 0:
            lora_info = (
                f", {len(lora_configs)} LoRA{'s' if len(lora_configs) > 1 else ''}"
            )
        logger.info(
            f"☁️ Built Cloud Wan22 T2V: {width}x{height}, {num_frames}f@{fps}fps, "
            f"{steps} steps (switch@{high_noise_steps}), cfg={cfg}, "
            f"shift={shift}, {sampler_name}/{scheduler}{lora_info}"
        )
        return workflow

    # ─────────────────────────────────────────────────────────────────────
    # LTX-2.3 22B Cloud Workflow Builders
    # Single-stage distilled pipeline: 8-step ManualSigmas, CFGGuider,
    # SamplerCustomAdvanced. Target: 80 GB+ cloud GPUs.
    # ─────────────────────────────────────────────────────────────────────

    def build_cloud_ltx23_t2v_workflow(
        self,
        prompt: str,
        negative_prompt: str = "low quality, blurry, distorted, artifacts, watermark",
        width: int = 768,
        height: int = 512,
        num_frames: int = 97,
        fps: int = 25,
        seed: int = -1,
        output_prefix: str = "oelala_ltx23_t2v",
        checkpoint: str = "ltx-2.3-22b-distilled.safetensors",
        text_encoder: str = "gemma_3_12B_it_fp8_scaled.safetensors",
        aspect_ratio: str = "9:16",
        long_edge: int = 768,
        lora_configs: Optional[list] = None,
        audio_prompt: Optional[str] = None,
        audio_vae_checkpoint: str = "ltx2_audio_vae.safetensors",
    ) -> Optional[Dict[str, Any]]:
        """
        Build LTX-2.3 22B Cloud T2V workflow — single-stage distilled pipeline.

        Uses the 8-step distilled sigma schedule. When audio_prompt is provided,
        generates audio-video using MultimodalGuider + audio VAE.
        """
        # LTX frame count must be 8k+1
        k = round((num_frames - 1) / 8)
        k = max(1, k)
        num_frames = 8 * k + 1

        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        # Calculate dimensions from aspect ratio
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

        if ar_w >= ar_h:
            height = long_edge
            width = int(long_edge * ar_w / ar_h)
        else:
            width = long_edge
            height = int(long_edge * ar_h / ar_w)

        # LTX requires dimensions divisible by 32
        width = (width // 32) * 32
        height = (height // 32) * 32

        logger.info(
            f"☁️ Building LTX-2.3 Cloud T2V: {width}x{height}, "
            f"{num_frames}f@{fps}fps, seed={seed}"
        )

        workflow = {}

        # Node 1: CheckpointLoaderSimple — loads MODEL + CLIP + VAE
        workflow["1"] = {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        }

        # Node 2: LTXAVTextEncoderLoader — Gemma 3 12B fp8 text encoder
        workflow["2"] = {
            "class_type": "LTXAVTextEncoderLoader",
            "inputs": {
                "text_encoder": text_encoder,
                "ckpt_name": checkpoint,
                "device": "default",
            },
        }

        # Build effective prompt (append audio description if provided)
        effective_prompt = prompt
        if audio_prompt:
            effective_prompt = f"{prompt}\nAudio: {audio_prompt}"

        # Node 3: Positive prompt
        workflow["3"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": effective_prompt, "clip": ["2", 0]},
        }

        # Node 4: Negative prompt
        workflow["4"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["2", 0]},
        }

        # Node 5: LTXVConditioning — set frame rate on conditioning
        workflow["5"] = {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["3", 0],
                "negative": ["4", 0],
                "frame_rate": float(fps),
            },
        }

        # Node 6: EmptyLTXVLatentVideo — create latent
        workflow["6"] = {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1,
            },
        }

        # Node 7: CFGGuider — cfg=1 for distilled model
        workflow["7"] = {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["1", 0],
                "positive": ["5", 0],
                "negative": ["5", 1],
                "cfg": 1.0,
            },
        }

        # Node 8: KSamplerSelect
        workflow["8"] = {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler_ancestral"},
        }

        # Node 9: ManualSigmas — 8-step distilled schedule
        workflow["9"] = {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0",
            },
        }

        # Node 10: RandomNoise
        workflow["10"] = {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        }

        # ── Audio-Video Pipeline (conditional) ──────────────────────
        sampler_latent_input = ["6", 0]  # default: empty video latent
        sampler_guider_input = ["7", 0]  # default: CFGGuider

        if audio_prompt:
            # Node 20: LTXVAudioVAELoader — load audio VAE
            workflow["20"] = {
                "class_type": "LTXVAudioVAELoader",
                "inputs": {"ckpt_name": audio_vae_checkpoint},
            }

            # Node 21: LTXVEmptyLatentAudio — empty audio latent matching video frames
            workflow["21"] = {
                "class_type": "LTXVEmptyLatentAudio",
                "inputs": {
                    "frames_number": num_frames,
                    "frame_rate": fps,
                    "batch_size": 1,
                    "audio_vae": ["20", 0],
                },
            }

            # Node 22: LTXVConcatAVLatent — combine video + audio latents
            workflow["22"] = {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {
                    "video_latent": ["6", 0],  # empty video latent
                    "audio_latent": ["21", 0],  # empty audio latent
                },
            }

            # Node 23: GuiderParameters — VIDEO modality (cfg=1.0 for distilled)
            workflow["23"] = {
                "class_type": "GuiderParameters",
                "inputs": {
                    "modality": "VIDEO",
                    "cfg": 1.0,
                    "stg": 0.0,
                    "perturb_attn": True,
                    "rescale": 0.0,
                    "modality_scale": 1.0,
                    "skip_step": 0,
                    "cross_attn": True,
                },
            }

            # Node 24: GuiderParameters — AUDIO modality (chained from VIDEO)
            workflow["24"] = {
                "class_type": "GuiderParameters",
                "inputs": {
                    "modality": "AUDIO",
                    "cfg": 1.0,
                    "stg": 0.0,
                    "perturb_attn": True,
                    "rescale": 0.0,
                    "modality_scale": 1.0,
                    "skip_step": 0,
                    "cross_attn": True,
                    "parameters": ["23", 0],  # chain from VIDEO params
                },
            }

            # Node 25: MultimodalGuider — replaces CFGGuider for AV generation
            workflow["25"] = {
                "class_type": "MultimodalGuider",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["5", 0],
                    "negative": ["5", 1],
                    "parameters": ["24", 0],
                    "skip_blocks": "",
                },
            }

            sampler_latent_input = ["22", 0]  # combined AV latent
            sampler_guider_input = ["25", 0]  # MultimodalGuider

            logger.info("🔊 T2V Audio pipeline enabled — MultimodalGuider + AV latents")

        # Node 11: SamplerCustomAdvanced — main sampling
        workflow["11"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["10", 0],
                "guider": sampler_guider_input,
                "sampler": ["8", 0],
                "sigmas": ["9", 0],
                "latent_image": sampler_latent_input,
            },
        }

        # ── Post-sampling: video decode (always) + audio decode (if AV) ──
        if audio_prompt:
            # Node 26: LTXVSeparateAVLatent — split denoised AV latent
            workflow["26"] = {
                "class_type": "LTXVSeparateAVLatent",
                "inputs": {
                    "av_latent": ["11", 1],  # denoised_output from sampler
                },
            }

            # Node 12: VAEDecodeTiled — decode VIDEO latent
            workflow["12"] = {
                "class_type": "VAEDecodeTiled",
                "inputs": {
                    "samples": ["26", 0],  # video_latent from separator
                    "vae": ["1", 2],
                    "tile_size": 512,
                    "overlap": 64,
                    "temporal_size": 512,
                    "temporal_overlap": 64,
                },
            }

            # Node 27: LTXVAudioVAEDecode — decode AUDIO latent
            workflow["27"] = {
                "class_type": "LTXVAudioVAEDecode",
                "inputs": {
                    "samples": ["26", 1],  # audio_latent from separator
                    "audio_vae": ["20", 0],  # audio VAE from loader
                },
            }

            # Node 13: VHS_VideoCombine — save video WITH audio
            workflow["13"] = {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "frame_rate": fps,
                    "loop_count": 0,
                    "filename_prefix": output_prefix,
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 17,
                    "save_metadata": True,
                    "trim_to_audio": True,
                    "pingpong": False,
                    "save_output": True,
                    "images": ["12", 0],
                    "audio": ["27", 0],  # decoded audio waveform
                },
            }
        else:
            # Node 12: VAEDecodeTiled — decode video-only latent
            workflow["12"] = {
                "class_type": "VAEDecodeTiled",
                "inputs": {
                    "samples": ["11", 1],  # denoised_output
                    "vae": ["1", 2],
                    "tile_size": 512,
                    "overlap": 64,
                    "temporal_size": 512,
                    "temporal_overlap": 64,
                },
            }

            # Node 13: VHS_VideoCombine — save video (no audio)
            workflow["13"] = {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "frame_rate": fps,
                    "loop_count": 0,
                    "filename_prefix": output_prefix,
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 17,
                    "save_metadata": True,
                    "trim_to_audio": False,
                    "pingpong": False,
                    "save_output": True,
                    "images": ["12", 0],
                },
            }

        # ── LoRA chain (single-stage: insert between checkpoint and CFGGuider) ──
        if lora_configs:
            current_model = ["1", 0]  # CheckpointLoaderSimple MODEL output
            lora_node_id = 100
            for i, config in enumerate(lora_configs):
                lora_name = config.get("high") or config.get("name", "")
                if not lora_name:
                    continue
                strength = config.get("strength", 1.0)
                node_id = str(lora_node_id + i)
                workflow[node_id] = {
                    "class_type": "LoraLoaderModelOnly",
                    "inputs": {
                        "lora_name": lora_name,
                        "strength_model": strength,
                        "model": current_model,
                    },
                }
                current_model = [node_id, 0]
                logger.info(f"🎨 LTX-2.3 T2V LoRA #{i + 1}: {lora_name} @ {strength}")
            # Wire final LoRA output to guider (CFGGuider or MultimodalGuider)
            guider_node = "25" if audio_prompt else "7"
            workflow[guider_node]["inputs"]["model"] = current_model

        lora_info = ""
        if lora_configs and len(lora_configs) > 0:
            lora_info = (
                f", {len(lora_configs)} LoRA{'s' if len(lora_configs) > 1 else ''}"
            )
        audio_info = " + audio" if audio_prompt else ""
        logger.info(
            f"☁️ Built LTX-2.3 Cloud T2V: {width}x{height}, "
            f"{num_frames}f@{fps}fps, 8-step distilled{lora_info}{audio_info}"
        )
        return workflow

    def build_cloud_ltx23_i2v_workflow(
        self,
        image_name: str,
        prompt: str,
        negative_prompt: str = "low quality, blurry, distorted, artifacts, watermark",
        width: int = 768,
        height: int = 512,
        num_frames: int = 97,
        fps: int = 25,
        seed: int = -1,
        strength: float = 1.0,
        output_prefix: str = "oelala_ltx23_i2v",
        checkpoint: str = "ltx-2.3-22b-distilled.safetensors",
        text_encoder: str = "gemma_3_12B_it_fp8_scaled.safetensors",
        lora_configs: Optional[list] = None,
        audio_prompt: Optional[str] = None,
        audio_vae_checkpoint: str = "ltx2_audio_vae.safetensors",
    ) -> Optional[Dict[str, Any]]:
        """
        Build LTX-2.3 22B Cloud I2V workflow — image-to-video with
        LTXVImgToVideoConditionOnly from ComfyUI-LTXVideo.

        Uses 8-step distilled sigma schedule. Image is preprocessed
        with LTXVPreprocess and used to condition the latent.

        When audio_prompt is provided, enables audio-video generation:
        appends audio description to prompt, adds audio VAE, empty audio
        latent, MultimodalGuider, and audio decode/mux nodes.
        """
        # LTX frame count must be 8k+1
        k = round((num_frames - 1) / 8)
        k = max(1, k)
        num_frames = 8 * k + 1

        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        # LTX requires dimensions divisible by 32
        width = (width // 32) * 32
        height = (height // 32) * 32

        logger.info(
            f"☁️ Building LTX-2.3 Cloud I2V: {width}x{height}, "
            f"{num_frames}f@{fps}fps, strength={strength}, seed={seed}"
        )

        workflow = {}

        # Node 1: CheckpointLoaderSimple — MODEL + CLIP + VAE
        workflow["1"] = {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        }

        # Node 2: LTXAVTextEncoderLoader — Gemma 3 12B fp8
        workflow["2"] = {
            "class_type": "LTXAVTextEncoderLoader",
            "inputs": {
                "text_encoder": text_encoder,
                "ckpt_name": checkpoint,
                "device": "default",
            },
        }

        # Node 3: LoadImage — input image
        workflow["3"] = {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        }

        # Node 4: LTXVPreprocess — compress/noise condition the image
        workflow["4"] = {
            "class_type": "LTXVPreprocess",
            "inputs": {
                "image": ["3", 0],
                "img_compression": 35,
            },
        }

        # Combine audio description into the main prompt when audio is enabled
        # LTX-2.3 AV model uses the same text encoder for both modalities
        effective_prompt = prompt
        if audio_prompt:
            effective_prompt = f"{prompt}\nAudio: {audio_prompt}"
            logger.info(f"🔊 Audio prompt appended: {audio_prompt[:80]}...")

        # Node 5: Positive prompt
        workflow["5"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": effective_prompt, "clip": ["2", 0]},
        }

        # Node 6: Negative prompt
        workflow["6"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["2", 0]},
        }

        # Node 7: LTXVConditioning — set frame rate
        workflow["7"] = {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["5", 0],
                "negative": ["6", 0],
                "frame_rate": float(fps),
            },
        }

        # Node 8: EmptyLTXVLatentVideo — create empty latent
        workflow["8"] = {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1,
            },
        }

        # Node 9: LTXVImgToVideoConditionOnly — condition latent on image
        workflow["9"] = {
            "class_type": "LTXVImgToVideoConditionOnly",
            "inputs": {
                "vae": ["1", 2],  # VAE from checkpoint
                "image": ["4", 0],  # preprocessed image
                "latent": ["8", 0],  # empty latent
                "strength": strength,
            },
        }

        # Node 10: CFGGuider — cfg=1 for distilled
        workflow["10"] = {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["1", 0],
                "positive": ["7", 0],
                "negative": ["7", 1],
                "cfg": 1.0,
            },
        }

        # Node 11: KSamplerSelect
        workflow["11"] = {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler_ancestral"},
        }

        # Node 12: ManualSigmas — 8-step distilled schedule
        workflow["12"] = {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0",
            },
        }

        # Node 13: RandomNoise
        workflow["13"] = {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        }

        # ── Audio-Video Pipeline (conditional) ──────────────────────
        # When audio_prompt is provided, we enable the full AV pipeline:
        # - Load audio VAE, create empty audio latents
        # - Concat video + audio latents
        # - Use MultimodalGuider instead of CFGGuider
        # - After sampling: separate, decode audio, mux into video
        sampler_latent_input = ["9", 0]  # default: image-conditioned video latent
        sampler_guider_input = ["10", 0]  # default: CFGGuider

        if audio_prompt:
            # Node 20: LTXVAudioVAELoader — load audio VAE
            workflow["20"] = {
                "class_type": "LTXVAudioVAELoader",
                "inputs": {"ckpt_name": audio_vae_checkpoint},
            }

            # Node 21: LTXVEmptyLatentAudio — empty audio latent matching video frames
            workflow["21"] = {
                "class_type": "LTXVEmptyLatentAudio",
                "inputs": {
                    "frames_number": num_frames,
                    "frame_rate": fps,
                    "batch_size": 1,
                    "audio_vae": ["20", 0],
                },
            }

            # Node 22: LTXVConcatAVLatent — combine video + audio latents
            workflow["22"] = {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {
                    "video_latent": ["9", 0],  # image-conditioned video latent
                    "audio_latent": ["21", 0],  # empty audio latent
                },
            }

            # Node 23: GuiderParameters — VIDEO modality (cfg=1.0 for distilled)
            workflow["23"] = {
                "class_type": "GuiderParameters",
                "inputs": {
                    "modality": "VIDEO",
                    "cfg": 1.0,
                    "stg": 0.0,
                    "perturb_attn": True,
                    "rescale": 0.0,
                    "modality_scale": 1.0,
                    "skip_step": 0,
                    "cross_attn": True,
                },
            }

            # Node 24: GuiderParameters — AUDIO modality (chained from VIDEO)
            workflow["24"] = {
                "class_type": "GuiderParameters",
                "inputs": {
                    "modality": "AUDIO",
                    "cfg": 1.0,
                    "stg": 0.0,
                    "perturb_attn": True,
                    "rescale": 0.0,
                    "modality_scale": 1.0,
                    "skip_step": 0,
                    "cross_attn": True,
                    "parameters": ["23", 0],  # chain from VIDEO params
                },
            }

            # Node 25: MultimodalGuider — replaces CFGGuider for AV generation
            workflow["25"] = {
                "class_type": "MultimodalGuider",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["7", 0],
                    "negative": ["7", 1],
                    "parameters": ["24", 0],
                    "skip_blocks": "",
                },
            }

            sampler_latent_input = ["22", 0]  # combined AV latent
            sampler_guider_input = ["25", 0]  # MultimodalGuider

            logger.info("🔊 Audio pipeline enabled — MultimodalGuider + AV latents")

        # Node 14: SamplerCustomAdvanced — main sampling
        workflow["14"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["13", 0],
                "guider": sampler_guider_input,
                "sampler": ["11", 0],
                "sigmas": ["12", 0],
                "latent_image": sampler_latent_input,
            },
        }

        # ── Post-sampling: video decode (always) + audio decode (if AV) ──
        if audio_prompt:
            # Node 26: LTXVSeparateAVLatent — split denoised AV latent
            workflow["26"] = {
                "class_type": "LTXVSeparateAVLatent",
                "inputs": {
                    "av_latent": ["14", 1],  # denoised_output from sampler
                },
            }

            # Node 15: VAEDecodeTiled — decode VIDEO latent (from separated output)
            workflow["15"] = {
                "class_type": "VAEDecodeTiled",
                "inputs": {
                    "samples": ["26", 0],  # video_latent from separator
                    "vae": ["1", 2],
                    "tile_size": 512,
                    "overlap": 64,
                    "temporal_size": 512,
                    "temporal_overlap": 64,
                },
            }

            # Node 27: LTXVAudioVAEDecode — decode AUDIO latent
            workflow["27"] = {
                "class_type": "LTXVAudioVAEDecode",
                "inputs": {
                    "samples": ["26", 1],  # audio_latent from separator
                    "audio_vae": ["20", 0],  # audio VAE from loader
                },
            }

            # Node 16: VHS_VideoCombine — save video WITH audio
            workflow["16"] = {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "frame_rate": fps,
                    "loop_count": 0,
                    "filename_prefix": output_prefix,
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 17,
                    "save_metadata": True,
                    "trim_to_audio": True,
                    "pingpong": False,
                    "save_output": True,
                    "images": ["15", 0],
                    "audio": ["27", 0],  # decoded audio waveform
                },
            }
        else:
            # Node 15: VAEDecodeTiled — decode video-only latent
            workflow["15"] = {
                "class_type": "VAEDecodeTiled",
                "inputs": {
                    "samples": ["14", 1],  # denoised_output
                    "vae": ["1", 2],
                    "tile_size": 512,
                    "overlap": 64,
                    "temporal_size": 512,
                    "temporal_overlap": 64,
                },
            }

            # Node 16: VHS_VideoCombine — save video (no audio)
            workflow["16"] = {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "frame_rate": fps,
                    "loop_count": 0,
                    "filename_prefix": output_prefix,
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 17,
                    "save_metadata": True,
                    "trim_to_audio": False,
                    "pingpong": False,
                    "save_output": True,
                    "images": ["15", 0],
                },
            }

        # ── LoRA chain (single-stage: insert between checkpoint and CFGGuider) ──
        if lora_configs:
            current_model = ["1", 0]  # CheckpointLoaderSimple MODEL output
            lora_node_id = 100
            for i, config in enumerate(lora_configs):
                lora_name = config.get("high") or config.get("name", "")
                if not lora_name:
                    continue
                strength_lora = config.get("strength", 1.0)
                node_id = str(lora_node_id + i)
                workflow[node_id] = {
                    "class_type": "LoraLoaderModelOnly",
                    "inputs": {
                        "lora_name": lora_name,
                        "strength_model": strength_lora,
                        "model": current_model,
                    },
                }
                current_model = [node_id, 0]
                logger.info(
                    f"🎨 LTX-2.3 I2V LoRA #{i + 1}: {lora_name} @ {strength_lora}"
                )
            # Wire final LoRA output to guider (CFGGuider or MultimodalGuider)
            guider_node = "25" if audio_prompt else "10"
            workflow[guider_node]["inputs"]["model"] = current_model

        lora_info = ""
        if lora_configs and len(lora_configs) > 0:
            lora_info = (
                f", {len(lora_configs)} LoRA{'s' if len(lora_configs) > 1 else ''}"
            )
        audio_info = " + audio" if audio_prompt else ""
        logger.info(
            f"☁️ Built LTX-2.3 Cloud I2V: {width}x{height}, "
            f"{num_frames}f@{fps}fps, strength={strength}, 8-step distilled{lora_info}{audio_info}"
        )
        return workflow

    # ── MiniMax-H3 cloud workflows ────────────────────────────────────
    #
    # MiniMax-H3 is a joint video+audio DiT (FL2VA). It always generates a
    # synchronized soundtrack: the AV latent carries both a video stream
    # (24 channels, 24 fps) and an audio stream (40 fps). The workflows below
    # mirror Comfy-Org's official "Image to Video (MiniMax H3)" template:
    #   UNETLoader(fl2va int8) + CLIPLoader(qwen3vl minimax) + 2× VAELoader
    #   → MiniMaxH3ImageToVideo (conditioning + AV latent)
    #   → BasicGuider + KSamplerSelect(res_multistep) + BasicScheduler(simple)
    #   → SamplerCustomAdvanced → VAEDecode (video) + VAEDecodeAudio (audio)
    #   → VHS_VideoCombine (mp4 with muxed audio).
    # The same fl2va checkpoint serves t2v AND i2v (i2v anchors the input
    # image as the first keyframe via MiniMaxH3ImageToVideo.first_frame).

    @staticmethod
    def _align_minimax_h3_frames(num_frames: int) -> int:
        """Snap a frame count to MiniMax-H3's 17k+5 grid at 24 fps."""
        n = max(5, int(num_frames))
        while n % 17 != 5:
            n += 1
        return n

    @staticmethod
    def _aspect_ratio_pair(aspect_ratio: str) -> Tuple[int, int]:
        """Resolve an aspect ratio string (e.g. '16:9') to a (w, h) pair."""
        pairs = {
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
        return pairs.get(aspect_ratio, (16, 9))

    @staticmethod
    def _minimax_h3_canvas(
        aspect_ratio: str, megapixels: Optional[float] = None
    ) -> Tuple[int, int]:
        """Compute the MiniMax-H3 output canvas (w, h), rounded to multiples of 32.

        Without *megapixels*: H3's native canvas — a 768px short edge capped at
        768×1344 pixels, using the exact same math as ComfyUI's official
        MiniMaxH3 nodes (`adapt_canvas` in nodes_minimax_h3.py).

        With *megapixels*: the official ResolutionSelector formula —
        target area = megapixels × 1024² px at the given aspect ratio, rounded
        to the nearest multiple of 32 (0.4 MP @16:9 → 864×480, 0.98 MP @16:9 →
        1344×768, 2.0 MP @16:9 → 1920×1088). Clamped to 0.2–2.0 MP.
        """
        ar_w, ar_h = ComfyUIClient._aspect_ratio_pair(aspect_ratio)

        if megapixels is None:
            ratio = ar_w / ar_h
            if ratio >= 1.0:
                nom_w, nom_h = 768.0 * ratio, 768.0
            else:
                nom_w, nom_h = 768.0, 768.0 / ratio
            max_pixels = 768 * 1344
            if nom_w * nom_h > max_pixels:
                s = math.sqrt(max_pixels / (nom_w * nom_h))
                nom_w, nom_h = nom_w * s, nom_h * s
            width = max(32, round(nom_w / 32) * 32)
            height = max(32, round(nom_h / 32) * 32)
            return width, height

        mp = max(0.2, min(2.0, float(megapixels)))
        total_pixels = mp * 1024 * 1024
        scale = math.sqrt(total_pixels / (ar_w * ar_h))
        width = max(32, round(ar_w * scale / 32) * 32)
        height = max(32, round(ar_h * scale / 32) * 32)
        return width, height

    def _build_minimax_h3_workflow(
        self,
        *,
        prompt: str,
        width: int,
        height: int,
        length: int,
        fps: int,
        seed: int,
        steps: int,
        output_prefix: str,
        checkpoint: str,
        text_encoder: str,
        video_vae: str,
        audio_vae: str,
        first_frame_link: Optional[list] = None,
        last_frame_link: Optional[list] = None,
        output_kind: str = "vhs",
    ) -> Dict[str, Any]:
        """Shared MiniMax-H3 sampling graph (model load → AV decode → mp4).

        *output_kind* selects how the decoded frames + audio are written:
          - "vhs" (default, cloud / ai-kvm2): the `VHS_VideoCombine` custom-node
            (VideoHelperSuite), which is available on the RunPod workers.
          - "savevideo" (local Windows PC): core `CreateVideo` + `SaveVideo`
            nodes, which exist on the Windows portable install where
            VideoHelperSuite (VHS_VideoCombine) is NOT installed.
        """
        workflow = {}

        # Node 1: UNETLoader — FL2VA diffusion model (t2v + i2v keyframes)
        workflow["1"] = {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": checkpoint,
                "weight_dtype": "default",
            },
        }

        # Node 2: CLIPLoader — Qwen3-VL-32B MiniMax text encoder
        workflow["2"] = {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": text_encoder,
                "type": "minimax",
                "device": "default",
            },
        }

        # Node 3: VAELoader — video VAE
        workflow["3"] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": video_vae},
        }

        # Node 4: VAELoader — audio VAE (H3 generates audio unconditionally)
        workflow["4"] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": audio_vae},
        }

        # Node 5: MiniMaxH3ImageToVideo — prompt → conditioning + AV latent
        h3_inputs = {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "prompt": prompt,
            "width": width,
            "height": height,
            "length": length,
        }
        if first_frame_link:
            h3_inputs["first_frame"] = first_frame_link
        if last_frame_link:
            h3_inputs["last_frame"] = last_frame_link
        workflow["5"] = {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": h3_inputs,
        }

        # Node 6: BasicGuider — no CFG / no negative prompt for H3
        workflow["6"] = {
            "class_type": "BasicGuider",
            "inputs": {
                "model": ["1", 0],
                "conditioning": ["5", 0],
            },
        }

        # Node 7: KSamplerSelect — res_multistep (official template)
        workflow["7"] = {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "res_multistep"},
        }

        # Node 8: BasicScheduler — simple schedule (official template)
        workflow["8"] = {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["1", 0],
                "scheduler": "simple",
                "steps": steps,
                "denoise": 1.0,
            },
        }

        # Node 9: RandomNoise
        workflow["9"] = {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        }

        # Node 10: SamplerCustomAdvanced — samples the flat AV pack
        workflow["10"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["9", 0],
                "guider": ["6", 0],
                "sampler": ["7", 0],
                "sigmas": ["8", 0],
                "latent_image": ["5", 1],
            },
        }

        # Node 11: VAEDecode — video frames from the AV latent
        workflow["11"] = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["10", 0],
                "vae": ["3", 0],
            },
        }

        # Node 12: VAEDecodeAudio — soundtrack from the AV latent
        workflow["12"] = {
            "class_type": "VAEDecodeAudio",
            "inputs": {
                "samples": ["10", 0],
                "vae": ["4", 0],
            },
        }

        # Node 13: output node — mp4 with muxed audio
        if output_kind == "savevideo":
            # Core CreateVideo (frames + audio -> VIDEO) + SaveVideo (-> mp4).
            # Used on the Windows PC portable ComfyUI where VideoHelperSuite
            # (VHS_VideoCombine) is not installed.
            #
            # Node 14 is reserved for the I2V LoadImage first-frame keyframe, so
            # SaveVideo lives on node 15 to avoid a collision.
            workflow["13"] = {
                "class_type": "CreateVideo",
                "inputs": {
                    "images": ["11", 0],
                    "fps": float(fps),
                    "audio": ["12", 0],
                },
            }
            workflow["15"] = {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["13", 0],
                    "filename_prefix": output_prefix,
                    "format": "auto",
                    "codec": "auto",
                },
            }
        else:
            # VHS_VideoCombine — VideoHelperSuite custom node (available on
            # the RunPod / ai-kvm2 workers).
            workflow["13"] = {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "frame_rate": fps,
                    "loop_count": 0,
                    "filename_prefix": output_prefix,
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 17,
                    "save_metadata": True,
                    "trim_to_audio": True,
                    "pingpong": False,
                    "save_output": True,
                    "images": ["11", 0],
                    "audio": ["12", 0],
                },
            }

        return workflow

    def build_cloud_minimax_h3_t2v_workflow(
        self,
        prompt: str,
        width: int = 1344,
        height: int = 768,
        num_frames: int = 124,
        fps: int = 24,
        seed: int = -1,
        steps: int = 20,
        output_prefix: str = "oelala_minimax_h3_t2v",
        checkpoint: str = "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        text_encoder: str = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        video_vae: str = "minimax_h3_video_vae_fp16.safetensors",
        audio_vae: str = "minimax_h3_audio_vae_fp32.safetensors",
        aspect_ratio: str = "16:9",
        megapixels: Optional[float] = None,
        long_edge: int = 768,
        output_kind: str = "vhs",
    ) -> Optional[Dict[str, Any]]:
        """
        Build MiniMax-H3 22B Cloud T2V workflow — text-to-video+audio.

        Matches Comfy-Org's official MiniMax-H3 template (simple/20-step
        schedule, res_multistep sampler, BasicGuider — no negative prompt).
        The model always generates a synchronized soundtrack.

        Canvas: native 768px short edge (capped 768×1344) unless *megapixels*
        is set, in which case the official ResolutionSelector formula is used
        (0.4 MP @16:9 → 864×480, 0.98 → 1344×768, 2.0 → 1920×1088).
        """
        length = self._align_minimax_h3_frames(num_frames)
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        width, height = self._minimax_h3_canvas(aspect_ratio, megapixels)

        logger.info(
            f"☁️ Building MiniMax-H3 Cloud T2V: {width}x{height}, "
            f"{length}f@{fps}fps, {steps} steps, seed={seed}"
        )

        return self._build_minimax_h3_workflow(
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            fps=fps,
            seed=seed,
            steps=steps,
            output_prefix=output_prefix,
            checkpoint=checkpoint,
            text_encoder=text_encoder,
            video_vae=video_vae,
            audio_vae=audio_vae,
            output_kind=output_kind,
        )

    def build_cloud_minimax_h3_i2v_workflow(
        self,
        image_name: str,
        prompt: str,
        width: int = 1344,
        height: int = 768,
        num_frames: int = 124,
        fps: int = 24,
        seed: int = -1,
        steps: int = 20,
        output_prefix: str = "oelala_minimax_h3_i2v",
        checkpoint: str = "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        text_encoder: str = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        video_vae: str = "minimax_h3_video_vae_fp16.safetensors",
        audio_vae: str = "minimax_h3_audio_vae_fp32.safetensors",
        aspect_ratio: str = "16:9",
        megapixels: Optional[float] = None,
        long_edge: int = 768,
        output_kind: str = "vhs",
    ) -> Optional[Dict[str, Any]]:
        """
        Build MiniMax-H3 22B Cloud I2V workflow — image-to-video+audio.

        Uses the same FL2VA checkpoint as T2V: the input image is anchored
        as the first keyframe of the video (MiniMaxH3ImageToVideo.first_frame).
        Canvas follows the same rule as T2V (native 768p short edge, or the
        official ResolutionSelector formula when *megapixels* is set).
        """
        length = self._align_minimax_h3_frames(num_frames)
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        width, height = self._minimax_h3_canvas(aspect_ratio, megapixels)

        logger.info(
            f"☁️ Building MiniMax-H3 Cloud I2V: {width}x{height}, "
            f"{length}f@{fps}fps, {steps} steps, seed={seed}, image={image_name}"
        )

        workflow = self._build_minimax_h3_workflow(
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            fps=fps,
            seed=seed,
            steps=steps,
            output_prefix=output_prefix,
            checkpoint=checkpoint,
            text_encoder=text_encoder,
            video_vae=video_vae,
            audio_vae=audio_vae,
            first_frame_link=["14", 0],
            output_kind=output_kind,
        )

        # Node 14: LoadImage — first-frame keyframe
        workflow["14"] = {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        }
        return workflow

    def build_local_minimax_h3_t2v_workflow(
        self,
        prompt: str,
        width: int = 1344,
        height: int = 768,
        num_frames: int = 124,
        fps: int = 24,
        seed: int = -1,
        steps: int = 20,
        output_prefix: str = "oelala_minimax_h3_t2v",
        checkpoint: str = "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        text_encoder: str = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        video_vae: str = "minimax_h3_video_vae_fp16.safetensors",
        audio_vae: str = "minimax_h3_audio_vae_fp32.safetensors",
        aspect_ratio: str = "16:9",
        megapixels: Optional[float] = None,
        long_edge: int = 768,
    ) -> Optional[Dict[str, Any]]:
        """
        Build MiniMax-H3 local (Windows PC ComfyUI) T2V workflow.

        Same MiniMax-H3 FL2VA graph as the cloud builder, but uses the model
        files on the user's Windows PC (ComfyUI portable). The local text
        encoder is the **int8_convrot** variant (`qwen3vl_32b_minimax_h3_int8_convrot.safetensors`),
        matching the Comfy-Org int8 pruned set that fits the 16 GB GPU —
        unlike the cloud worker's nvfp4_awq encoder.

        Always generates a synchronized soundtrack (H3 FL2VA).
        """
        return self.build_cloud_minimax_h3_t2v_workflow(
            prompt=prompt,
            num_frames=num_frames,
            fps=fps,
            seed=seed,
            steps=steps,
            output_prefix=output_prefix,
            checkpoint=checkpoint,
            text_encoder=text_encoder,
            video_vae=video_vae,
            audio_vae=audio_vae,
            aspect_ratio=aspect_ratio,
            megapixels=megapixels,
            output_kind="savevideo",
        )

    def build_local_minimax_h3_i2v_workflow(
        self,
        image_name: str,
        prompt: str,
        width: int = 1344,
        height: int = 768,
        num_frames: int = 124,
        fps: int = 24,
        seed: int = -1,
        steps: int = 20,
        output_prefix: str = "oelala_minimax_h3_i2v",
        checkpoint: str = "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        text_encoder: str = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        video_vae: str = "minimax_h3_video_vae_fp16.safetensors",
        audio_vae: str = "minimax_h3_audio_vae_fp32.safetensors",
        aspect_ratio: str = "16:9",
        megapixels: Optional[float] = None,
        long_edge: int = 768,
    ) -> Optional[Dict[str, Any]]:
        """
        Build MiniMax-H3 local (Windows PC ComfyUI) I2V workflow.

        Same FL2VA graph as the cloud I2V builder, but with the local
        int8_convrot text encoder that lives on the Windows PC. The input
        image is anchored as the first keyframe. Always generates audio.
        """
        return self.build_cloud_minimax_h3_i2v_workflow(
            image_name=image_name,
            prompt=prompt,
            num_frames=num_frames,
            fps=fps,
            seed=seed,
            steps=steps,
            output_prefix=output_prefix,
            checkpoint=checkpoint,
            text_encoder=text_encoder,
            video_vae=video_vae,
            audio_vae=audio_vae,
            aspect_ratio=aspect_ratio,
            megapixels=megapixels,
            output_kind="savevideo",
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
            # Read file content — try local first, fall back to storage
            file_path = Path(output_path)
            if file_path.exists():
                try:
                    with open(output_path, "rb") as f:
                        file_data = f.read()
                except (IOError, OSError) as e:
                    logger.error(f"❌ Failed to read file {output_path}: {e}")
                    self.clear_job_metadata(prompt_id)
                    return None
            else:
                # Try fetching from storage (path may be "generated/cloud-wan22/file.mp4")
                try:
                    sc = get_storage_client()
                    parts = output_path.replace("\\", "/").split("/", 1)
                    if len(parts) == 2:
                        file_data = sc.get(parts[0], parts[1])
                    if not file_data:
                        raise FileNotFoundError(output_path)
                    logger.info(
                        f"📦 Read {len(file_data)} bytes from storage: {output_path}"
                    )
                except Exception as e:
                    logger.error(
                        f"❌ File not found locally or in storage: {output_path} ({e})"
                    )
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

            # Cleanup local file to keep media dirs empty
            try:
                local_path = Path(output_path)
                minio_data_dir = Path(
                    os.environ.get("MINIO_DATA_DIR", "/home/flip/minio-data")
                )
                if local_path.exists() and str(local_path.parent) != str(
                    minio_data_dir
                ):
                    local_path.unlink()
                    logger.info(f"🗑️ Cleaned up auto-uploaded local file: {local_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup local file {output_path}: {e}")

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
            # Read file content — try local first, fall back to storage
            file_path = Path(output_path)
            if file_path.exists():
                file_data = file_path.read_bytes()
            else:
                # Try fetching from storage (path may be "generated/cloud-wan22/file.mp4")
                try:
                    sc = get_storage_client()
                    parts = output_path.replace("\\", "/").split("/", 1)
                    if len(parts) == 2:
                        file_data = sc.get(parts[0], parts[1])
                    else:
                        file_data = None
                    if not file_data:
                        raise FileNotFoundError(output_path)
                    logger.info(
                        f"📦 Read {len(file_data)} bytes from storage: {output_path}"
                    )
                except Exception as e:
                    logger.error(
                        f"❌ File not found locally or in storage: {output_path} ({e})"
                    )
                    self.clear_job_metadata(prompt_id)
                    return None

            # Map output_type to generation_type
            # Settings are nested under metadata["settings"]
            settings = metadata.get("settings", {})
            job_type = settings.get("job_type", "") or settings.get("type", "")
            gen_type_map = {
                "video": "t2v" if "t2v" in job_type else "i2v",
                "image": "t2i",
                "audio": "audio",
            }
            generation_type = gen_type_map.get(output_type, output_type)

            # Extract metadata for Supabase from nested settings
            extra_metadata = {
                "model_name": settings.get("model_name")
                or settings.get("model_type")
                or settings.get("model_mode"),
                "resolution": settings.get("resolution"),
                "aspect_ratio": settings.get("aspect_ratio"),
                "num_frames": settings.get("num_frames"),
                "fps": settings.get("fps"),
                "seed": settings.get("seed"),
                "steps": settings.get("steps"),
                "cfg": settings.get("cfg"),
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
                model_name=extra_metadata.get("model_name"),
                workflow_id=prompt_id,
            )

            logger.info(
                f"✅ Async uploaded to storage: {record.storage_path} ({len(file_data)} bytes)"
            )

            # Cleanup local file to keep media dirs empty
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"🗑️ Cleaned up async local file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup async local file {file_path}: {e}")

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

                            # Upload to generated bucket in MinIO
                            try:
                                storage_client = get_storage_client()
                                storage_client.put("generated", filename, resp.content)
                                logger.info(
                                    f"📤 Video uploaded to storage: generated/{filename}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"⚠️ Storage upload failed (non-fatal): {e}"
                                )

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

                            # Upload to generated bucket in MinIO
                            try:
                                storage_client = get_storage_client()
                                storage_client.put("generated", filename, resp.content)
                                logger.info(
                                    f"📤 Image uploaded to storage: generated/{filename}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"⚠️ Storage upload failed (non-fatal): {e}"
                                )

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
        - DisTorch2 multi-GPU distribution (cuda:1,12gb;cuda:0,16gb;cpu,*)

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
    ) -> Optional[Dict]:
        """
        Build a video upscaling workflow using Real-ESRGAN.

        Args:
            video_path: Path to input video
            scale: Upscale factor (2 or 4)
            output_prefix: Prefix for output filename
            model: Upscale model to use

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
                    "crf": 19,
                    "save_metadata": True,
                    "images": ["3", 0],
                    "audio": ["1", 1],
                },
                "class_type": "VHS_VideoCombine",
            },
        }

        logger.info(f"🔧 Built video upscale workflow: {scale}x with {model}")
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
            "compute_device": "cuda:1",
            "virtual_vram_gb": 16,
            "donor_device": "cuda:0",
            "expert_mode_allocations": "cuda:1,12gb;cuda:0,16gb;cpu,*",
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
            "compute_device": "cuda:1",
            "virtual_vram_gb": 16,
            "donor_device": "cuda:0",
            "expert_mode_allocations": "cuda:1,12gb;cuda:0,16gb;cpu,*",
            "eject_models": True,
        },
        "class_type": "UnetLoaderGGUFAdvancedDisTorch2MultiGPU",
    },
    "3": {
        "inputs": {
            "vae_name": "wan_2.1_vae.safetensors",
            "compute_device": "cuda:1",
            "virtual_vram_gb": 4,
            "donor_device": "cuda:0",
            "expert_mode_allocations": "cuda:1,12gb;cuda:0,16gb;cpu,*",
            "eject_models": True,
        },
        "class_type": "VAELoaderDisTorch2MultiGPU",
    },
    "4": {
        "inputs": {
            "clip_name": "umt5-xxl-enc-bf16.safetensors",
            "type": "wan",
            "device": "cuda:1",
            "virtual_vram_gb": 4,
            "donor_device": "cuda:0",
            "expert_mode_allocations": "cuda:1,12gb;cuda:0,16gb;cpu,*",
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

# Optional second ComfyUI server — the user's Windows PC (used for local
# MiniMax-H3 generation, which runs on that machine's ComfyUI rather than the
# one on ai-kvm2). Configured via COMFYUI_WINDOWS_HOST / COMFYUI_WINDOWS_PORT.
_windows_comfyui_client: Optional[ComfyUIClient] = None


def get_comfyui_client() -> ComfyUIClient:
    """Get or create ComfyUI client singleton"""
    global _comfyui_client
    if _comfyui_client is None:
        _comfyui_client = ComfyUIClient()
    return _comfyui_client


def get_windows_comfyui_client() -> Optional[ComfyUIClient]:
    """Get/create the ComfyUI client for the user's Windows PC.

    This is a separate ComfyUI server from the one on ai-kvm2 (which
    get_comfyui_client() targets). Local MiniMax-H3 generation is routed here
    because the H3 models + workflow live on the Windows machine
    (ComfyUI portable, e.g. 192.168.1.245:8188).

    Returns None when the host/port haven't been configured, so any
    dependent adapter/registration is skipped gracefully.
    """
    global _windows_comfyui_client
    if _windows_comfyui_client is not None:
        return _windows_comfyui_client

    host = os.getenv("COMFYUI_WINDOWS_HOST", "").strip()
    port = int(os.getenv("COMFYUI_WINDOWS_PORT", "8188") or "8188")
    if not host:
        logger.info(
            "⬛ COMFYUI_WINDOWS_HOST not set — Windows ComfyUI client unavailable"
        )
        return None

    _windows_comfyui_client = ComfyUIClient(host=host, port=port)
    logger.info(f"🪟 Windows ComfyUI client ready: {host}:{port}")
    return _windows_comfyui_client


# ── Generic per-backend ComfyUI clients ─────────────────────────────
# The Compute Backend Inventory (generation/compute_backends.py) drives which
# server an adapter targets. get_comfyui_client_for_backend() returns a
# (cached) ComfyUIClient for any configured 'comfyui' backend so adding a new
# server is purely a config change.
_backend_clients: dict = {}


def _parse_base_url(base_url: str):
    """Split 'http://host:port' -> (host, port). Defaults to localhost:8188."""
    host, port = "localhost", 8188
    clean = (base_url or "").strip()
    if clean:
        if "://" in clean:
            clean = clean.split("://", 1)[1]
        host = clean.split("/")[0].split(":")[0]
        if ":" in clean.split("/")[0]:
            try:
                port = int(clean.split("/")[0].split(":")[1])
            except (ValueError, IndexError):
                port = 8188
    return host, port


def get_comfyui_client_for_backend(backend) -> Optional[ComfyUIClient]:
    """Get or create a ComfyUIClient for a ComputeBackend.

    Accepts a ComputeBackend object (generation.compute_backends) or a plain
    dict with keys id/base_url/type. Returns None for runpod backends or when
    no base_url is configured.
    """
    if backend is None:
        return None
    if isinstance(backend, dict):
        backend_id = backend.get("id", "")
        base_url = backend.get("base_url", "")
        btype = backend.get("type", "comfyui")
    else:
        backend_id = backend.id
        base_url = backend.base_url
        btype = backend.type

    if btype == "runpod" or not base_url:
        return None

    # Cache by (backend_id, base_url) so an admin edit to a backend's base_url
    # takes effect on the next dispatch instead of returning the stale client.
    cache_key = (backend_id, base_url)
    if cache_key in _backend_clients:
        return _backend_clients[cache_key]

    host, port = _parse_base_url(base_url)
    client = ComfyUIClient(host=host, port=port)
    _backend_clients[cache_key] = client
    logger.info(f"⚙️ ComfyUI client ready for backend '{backend_id}': {host}:{port}")
    return client
