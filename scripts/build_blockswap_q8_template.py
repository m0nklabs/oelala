#!/usr/bin/env python3
"""
Build a clean API-format workflow template for the BlockSwap Q8 experimental mode.
This creates the template JSON that the oelala backend will load and inject parameters into.

Pipeline:
  LoadImage → ImageResize → [Florence2 captioning] → CLIPTextEncode
  UnetGGUF High/Low → LoRA → EnhanceAVideo → CFGZeroStar → ModelSamplingSD3 → NAG
    → PatcherOrder → TorchCompile → BlockSwap → Sampler
  VAEDecode → ColorMatch → VHS_VideoCombine
  [Optional: Upscale → RIFE interpolation → VHS_VideoCombine]
"""

import json


def build_blockswap_q8_template():
    """Build the clean API workflow template."""

    workflow = {}

    # ─────────────────────────────────────────────────────────────────────
    # Input Image
    # ─────────────────────────────────────────────────────────────────────
    workflow["88"] = {
        "class_type": "LoadImage",
        "inputs": {
            "image": "INPUT_IMAGE.jpg",  # Replaced by backend
        },
        "_meta": {"title": "Input Image"},
    }

    # Resize input image to target dimensions
    workflow["401"] = {
        "class_type": "ImageResizeKJv2",
        "inputs": {
            "image": ["88", 0],
            "width": 720,       # Replaced by backend
            "height": 1280,     # Replaced by backend
            "upscale_method": "lanczos",
            "keep_proportion": "True",
            "pad_color": "0, 0, 0",
            "crop_position": "center",
            "divisible_by": 8,
        },
        "_meta": {"title": "Resize Image"},
    }

    # ─────────────────────────────────────────────────────────────────────
    # Florence2 Auto-Captioning (visual description of input image)
    # ─────────────────────────────────────────────────────────────────────
    workflow["525"] = {
        "class_type": "DownloadAndLoadFlorence2Model",
        "inputs": {
            "model": "MiaoshouAI/Florence-2-base-PromptGen-v2.0",
            "precision": "fp16",
            "attention": "sdpa",
            "convert_to_safetensors": False,
        },
        "_meta": {"title": "Florence2 Model"},
    }

    workflow["526"] = {
        "class_type": "Florence2Run",
        "inputs": {
            "image": ["401", 0],
            "florence2_model": ["525", 0],
            "text_input": "",
            "task": "detailed_caption",
            "fill_mask": True,
            "keep_model_loaded": False,
            "max_new_tokens": 1024,
            "num_beams": 3,
            "do_sample": True,
            "output_mask_select": "",
            "seed": 777777777777777,
        },
        "_meta": {"title": "Florence2 Caption"},
    }

    # Text Find and Replace chain - clean up captioning output
    # Replaces art/photo terms with "video" for better motion prompts
    replacements = [
        ("painting", "video"),
        ("illustration", "video"),
        ("drawing", "video"),
        ("sketch", "video"),
        ("artwork", "video"),
        ("photograph", "video"),
        ("photo", "video"),
        ("image", "video"),
        ("picture", "video"),
        ("portrait", "video"),
        ("render", "video"),
        ("scene", "video"),
    ]

    prev_node = "526"  # Florence2Run outputs STRING at slot 0
    prev_slot = 0
    for i, (find, replace) in enumerate(replacements):
        node_id = str(535 + i)  # Start at 535
        workflow[node_id] = {
            "class_type": "Text Find and Replace",
            "inputs": {
                "text": [prev_node, prev_slot],
                "find": find,
                "replace": replace,
            },
            "_meta": {"title": f"Replace {find}→{replace}"},
        }
        prev_node = node_id
        prev_slot = 0

    last_replace_node = prev_node

    # Concatenate Florence2 caption with user prompt
    workflow["451"] = {
        "class_type": "StringConcatenate",
        "inputs": {
            "string_a": [last_replace_node, 0],
            "string_b": "USER_PROMPT",  # Replaced by backend
        },
        "_meta": {"title": "Combine Caption + Prompt"},
    }

    # ─────────────────────────────────────────────────────────────────────
    # Text Encoding
    # ─────────────────────────────────────────────────────────────────────
    workflow["460"] = {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "type": "wan",
            "device": "default",
        },
        "_meta": {"title": "CLIP"},
    }

    workflow["462"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["460", 0],
            "text": ["451", 0],
        },
        "_meta": {"title": "Positive Encode"},
    }

    workflow["463"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["460", 0],
            "text": "low quality, blurry, distorted, artifacts",  # Replaced by backend
        },
        "_meta": {"title": "Negative Encode"},
    }

    # ─────────────────────────────────────────────────────────────────────
    # VAE
    # ─────────────────────────────────────────────────────────────────────
    workflow["461"] = {
        "class_type": "VAELoader",
        "inputs": {
            "vae_name": "wan_2.1_vae.safetensors",
        },
        "_meta": {"title": "VAE"},
    }

    # ─────────────────────────────────────────────────────────────────────
    # Model Loading - Dual Q8_0 GGUF (High Noise + Low Noise)
    # ─────────────────────────────────────────────────────────────────────
    workflow["495"] = {
        "class_type": "UnetLoaderGGUF",
        "inputs": {
            "unet_name": "Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf",
        },
        "_meta": {"title": "Model - High noise"},
    }

    workflow["496"] = {
        "class_type": "UnetLoaderGGUF",
        "inputs": {
            "unet_name": "Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf",
        },
        "_meta": {"title": "Model - Low noise"},
    }

    # ─────────────────────────────────────────────────────────────────────
    # LoRA Loading - Lightning LoRA for speed (4-8 steps instead of 20+)
    # Always-on: enables fast generation. Additional LoRAs added by backend.
    # ─────────────────────────────────────────────────────────────────────
    workflow["416"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["495", 0],
            "lora_name": "Wan_2_2_I2V_A14B_HIGH_lightx2v_4step_lora_v1030_rank_64_bf16.safetensors",
            "strength_model": 1.0,
        },
        "_meta": {"title": "Lightning LoRA - High noise"},
    }

    workflow["471"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["496", 0],
            "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
            "strength_model": 1.0,
        },
        "_meta": {"title": "Lightning LoRA - Low noise"},
    }

    # Additional LoRA slots (dynamically populated by backend)
    # Backend chains LoraLoaderModelOnly nodes starting at ID 170 (high) and 180 (low)
    # These connect: 416 → 170 → 171 → ... → 481 (EnhanceAVideo High)
    #                471 → 180 → 181 → ... → 482 (EnhanceAVideo Low)

    # ─────────────────────────────────────────────────────────────────────
    # Model Optimization Chain - HIGH NOISE
    # ─────────────────────────────────────────────────────────────────────

    # EnhanceAVideo - video quality enhancement
    workflow["481"] = {
        "class_type": "WanVideoEnhanceAVideoKJ",
        "inputs": {
            "model": ["416", 0],
            "latent": ["464", 3],  # WanImageToVideo latent output
            "weight": 1.0,
        },
        "_meta": {"title": "Enhance High"},
    }

    # CFGZeroStar - CFG guidance trick
    workflow["483"] = {
        "class_type": "CFGZeroStarAndInit",
        "inputs": {
            "model": ["481", 0],
            "use_zero_init": False,
            "zero_init_steps": 2,
        },
        "_meta": {"title": "CFG Zero High"},
    }

    # ModelSamplingSD3 - shift parameter
    workflow["467"] = {
        "class_type": "ModelSamplingSD3",
        "inputs": {
            "model": ["483", 0],
            "shift": 8.0,  # Configurable shift
        },
        "_meta": {"title": "Sampling High"},
    }

    # NAG - Normalized Attention Guidance
    workflow["485"] = {
        "class_type": "WanVideoNAG",
        "inputs": {
            "model": ["467", 0],
            "conditioning": ["463", 0],  # Negative conditioning (used for NAG)
            "nag_scale": 11.0,
            "nag_alpha": 0.25,
            "nag_tau": 2.373,
            "input_type": "default",
        },
        "_meta": {"title": "NAG High"},
    }

    # PatchModelPatcherOrder
    workflow["491"] = {
        "class_type": "PatchModelPatcherOrder",
        "inputs": {
            "model": ["485", 0],
            "patch_order": "weight_patch_first",
            "full_load": "auto",
        },
        "_meta": {"title": "Patcher High"},
    }

    # TorchCompile
    workflow["492"] = {
        "class_type": "TorchCompileModelWanVideo",
        "inputs": {
            "model": ["491", 0],
        },
        "_meta": {"title": "Compile High"},
    }

    # BlockSwap - VRAM reduction
    workflow["500"] = {
        "class_type": "wanBlockSwap",
        "inputs": {
            "model": ["492", 0],
        },
        "_meta": {"title": "BlockSwap High"},
    }

    # ─────────────────────────────────────────────────────────────────────
    # Model Optimization Chain - LOW NOISE
    # ─────────────────────────────────────────────────────────────────────

    workflow["482"] = {
        "class_type": "WanVideoEnhanceAVideoKJ",
        "inputs": {
            "model": ["471", 0],
            "latent": ["464", 3],
            "weight": 1.0,
        },
        "_meta": {"title": "Enhance Low"},
    }

    workflow["484"] = {
        "class_type": "CFGZeroStarAndInit",
        "inputs": {
            "model": ["482", 0],
            "use_zero_init": False,
            "zero_init_steps": 2,
        },
        "_meta": {"title": "CFG Zero Low"},
    }

    workflow["468"] = {
        "class_type": "ModelSamplingSD3",
        "inputs": {
            "model": ["484", 0],
            "shift": 8.0,
        },
        "_meta": {"title": "Sampling Low"},
    }

    workflow["486"] = {
        "class_type": "WanVideoNAG",
        "inputs": {
            "model": ["468", 0],
            "conditioning": ["463", 0],
            "nag_scale": 11.0,
            "nag_alpha": 0.25,
            "nag_tau": 2.373,
            "input_type": "default",
        },
        "_meta": {"title": "NAG Low"},
    }

    workflow["493"] = {
        "class_type": "PatchModelPatcherOrder",
        "inputs": {
            "model": ["486", 0],
            "patch_order": "weight_patch_first",
            "full_load": "auto",
        },
        "_meta": {"title": "Patcher Low"},
    }

    workflow["494"] = {
        "class_type": "TorchCompileModelWanVideo",
        "inputs": {
            "model": ["493", 0],
        },
        "_meta": {"title": "Compile Low"},
    }

    workflow["501"] = {
        "class_type": "wanBlockSwap",
        "inputs": {
            "model": ["494", 0],
        },
        "_meta": {"title": "BlockSwap Low"},
    }

    # ─────────────────────────────────────────────────────────────────────
    # WanImageToVideo - I2V conditioning
    # ─────────────────────────────────────────────────────────────────────
    workflow["464"] = {
        "class_type": "WanImageToVideo",
        "inputs": {
            "positive": ["462", 0],
            "negative": ["463", 0],
            "vae": ["461", 0],
            "start_image": ["401", 0],  # Resized image
            "width": 720,       # Replaced by backend
            "height": 1280,     # Replaced by backend
            "length": 121,      # Replaced by backend (4k+1 format)
            "batch_size": 1,
        },
        "_meta": {"title": "WanImageToVideo"},
    }

    # ─────────────────────────────────────────────────────────────────────
    # Random Noise
    # ─────────────────────────────────────────────────────────────────────
    workflow["73"] = {
        "class_type": "RandomNoise",
        "inputs": {
            "noise_seed": 42,  # Replaced by backend
        },
        "_meta": {"title": "Noise"},
    }

    # ─────────────────────────────────────────────────────────────────────
    # Dual-Pass KSampler (High Noise → Low Noise handoff)
    # ─────────────────────────────────────────────────────────────────────
    workflow["466"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model": ["500", 0],          # BlockSwap High output
            "positive": ["464", 0],       # WanImageToVideo conditioning
            "negative": ["464", 1],       # WanImageToVideo negative
            "latent_image": ["464", 2],   # WanImageToVideo latent
            "add_noise": "enable",
            "noise_seed": 42,             # Replaced by backend
            "steps": 8,                   # Total steps (replaced by backend)
            "cfg": 3.5,                   # CFG scale (replaced by backend)
            "sampler_name": "euler",
            "scheduler": "simple",
            "start_at_step": 0,
            "end_at_step": 5,             # High noise steps (half of total)
            "return_with_leftover_noise": "enable",
        },
        "_meta": {"title": "Sampler High"},
    }

    workflow["465"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model": ["501", 0],          # BlockSwap Low output
            "positive": ["464", 0],
            "negative": ["464", 1],
            "latent_image": ["466", 0],   # High sampler output (handoff)
            "add_noise": "disable",
            "noise_seed": 42,             # Same seed
            "steps": 8,                   # Same total steps
            "cfg": 3.5,                   # Same CFG
            "sampler_name": "euler",
            "scheduler": "simple",
            "start_at_step": 5,           # Continue from high noise
            "end_at_step": 10000,         # Run to completion
            "return_with_leftover_noise": "disable",
        },
        "_meta": {"title": "Sampler Low"},
    }

    # ─────────────────────────────────────────────────────────────────────
    # VAE Decode
    # ─────────────────────────────────────────────────────────────────────
    workflow["469"] = {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["465", 0],  # Low sampler output
            "vae": ["461", 0],
        },
        "_meta": {"title": "VAE Decode"},
    }

    # Color matching to reference image
    workflow["400"] = {
        "class_type": "ColorMatch",
        "inputs": {
            "image_ref": ["88", 0],   # Original input image
            "image_target": ["469", 0],
        },
        "_meta": {"title": "Color Match"},
    }

    # ─────────────────────────────────────────────────────────────────────
    # Output - Primary video
    # ─────────────────────────────────────────────────────────────────────
    workflow["398"] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": ["400", 0],
            "frame_rate": 24,
            "loop_count": 0,
            "filename_prefix": "oelala_blockswap_q8",  # Replaced by backend
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 19,
            "save_metadata": True,
            "trim_to_audio": False,
            "pingpong": False,
            "save_output": True,
        },
        "_meta": {"title": "Output Video"},
    }

    # ─────────────────────────────────────────────────────────────────────
    # Post-processing: Upscale + Interpolation (optional chain)
    # ─────────────────────────────────────────────────────────────────────
    workflow["384"] = {
        "class_type": "UpscaleModelLoader",
        "inputs": {
            "model_name": "4x_foolhardy_Remacri.pth",
        },
        "_meta": {"title": "Upscale Model"},
    }

    workflow["385"] = {
        "class_type": "ImageUpscaleWithModel",
        "inputs": {
            "upscale_model": ["384", 0],
            "image": ["400", 0],  # Color-matched video frames
        },
        "_meta": {"title": "Upscale 4x"},
    }

    # Scale back down (4x upscale → scale by 0.25 to get 1x, or higher for final)
    workflow["418"] = {
        "class_type": "ImageScaleBy",
        "inputs": {
            "image": ["385", 0],
            "upscale_method": "lanczos",
            "scale_by": 0.5,  # 4x * 0.5 = 2x final upscale
        },
        "_meta": {"title": "Scale to 2x"},
    }

    # RIFE Frame Interpolation (24fps → 48fps)
    workflow["431"] = {
        "class_type": "RIFE VFI",
        "inputs": {
            "frames": ["400", 0],  # Original (non-upscaled) for interpolation
            "ckpt_name": "rife47.pth",
            "clear_cache_after_n_frames": 10,
            "multiplier": 2,
            "fast_mode": True,
            "ensemble": True,
            "scale_factor": 1.0,
            "dtype": "float32",
            "torch_compile": False,
            "batch_size": 1,
        },
        "_meta": {"title": "Interpolate 2x"},
    }

    # Upscaled + interpolated output
    workflow["419"] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": ["418", 0],
            "frame_rate": 24,
            "loop_count": 0,
            "filename_prefix": "oelala_blockswap_q8_upscaled",
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 19,
            "save_metadata": True,
            "trim_to_audio": False,
            "pingpong": False,
            "save_output": True,
        },
        "_meta": {"title": "Output Upscaled"},
    }

    workflow["433"] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": ["431", 0],
            "frame_rate": 48,  # 24fps * 2x interpolation
            "loop_count": 0,
            "filename_prefix": "oelala_blockswap_q8_interpolated",
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 19,
            "save_metadata": True,
            "trim_to_audio": False,
            "pingpong": False,
            "save_output": True,
        },
        "_meta": {"title": "Output Interpolated"},
    }

    return workflow


if __name__ == "__main__":
    workflow = build_blockswap_q8_template()
    output_path = "/home/flip/oelala/workflows/ImageToVideo/wan22_i2v_blockswap_q8_api.json"
    with open(output_path, "w") as f:
        json.dump(workflow, f, indent=2)

    print(f"✅ Built clean API template with {len(workflow)} nodes")
    print(f"📁 Saved to: {output_path}")

    # Summary of node types
    types = {}
    for nid, node in workflow.items():
        ct = node["class_type"]
        types[ct] = types.get(ct, 0) + 1
    print("\nNode types:")
    for ct, count in sorted(types.items()):
        print(f"  {ct}: {count}")
