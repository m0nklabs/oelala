#!/usr/bin/env python3
"""
Wan2.2 I2V - DisTorch MultiGPU v3 (all correct parameters)
"""
import json
import requests
import random

workflow = {
    # 1. Load input image
    "1": {
        "class_type": "LoadImage",
        "inputs": {"image": "wan_input.png"}
    },

    # 2. Load T5 Text Encoder (correct params)
    "2": {
        "class_type": "LoadWanVideoT5TextEncoderMultiGPU",
        "inputs": {
            "model_name": "umt5_xxl_fp8_e4m3fn.safetensors",
            "precision": "bf16",
            "device": "cuda:0",
            "quantization": "fp8_e4m3fn"
        }
    },

    # 3. Encode text
    "3": {
        "class_type": "WanVideoTextEncodeMultiGPU",
        "inputs": {
            "t5": ["2", 0],
            "positive_prompt": "gentle ocean waves lapping at the shore, peaceful sunset colors, cinematic quality, smooth motion",
            "negative_prompt": "blurry, distorted, noise, artifacts, static, low quality",
            "force_offload": True
        }
    },

    # 4. Load VAE
    "4": {
        "class_type": "WanVideoVAELoaderMultiGPU",
        "inputs": {
            "model_name": "Wan2.1_VAE.safetensors",
            "load_device": "cuda:0",
            "precision": "bf16"
        }
    },

    # 5. Load CLIP Vision
    "5": {
        "class_type": "CLIPVisionLoader",
        "inputs": {"clip_name": "clip-vit-large.safetensors"}
    },

    # 6. Encode image with CLIP Vision (use standard node - simpler)
    "6": {
        "class_type": "WanVideoClipVisionEncode",
        "inputs": {
            "clip_vision": ["5", 0],
            "image_1": ["1", 0],
            "strength_1": 1.0,
            "strength_2": 0.0,
            "crop": "center",
            "combine_embeds": "average",
            "force_offload": True
        }
    },

    # 7. BlockSwap - offload 40 blocks to cuda:1
    "7": {
        "class_type": "WanVideoBlockSwapMultiGPU",
        "inputs": {
            "blocks_to_swap": 40,
            "offload_img_emb": True,
            "offload_txt_emb": True,
            "swap_device": "cuda:1"
        }
    },

    # 8. Load model with BlockSwap
    "8": {
        "class_type": "WanVideoModelLoaderMultiGPU",
        "inputs": {
            "model": "Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf",
            "base_precision": "bf16",
            "quantization": "disabled",
            "load_device": "offload_device",
            "compute_device": "cuda:0",
            "attention_mode": "sdpa",
            "block_swap_args": ["7", 0]
        }
    },

    # 9. Encode image to video latents
    "9": {
        "class_type": "WanVideoImageToVideoEncodeMultiGPU",
        "inputs": {
            "vae": ["4", 0],
            "clip_embeds": ["6", 0],
            "start_image": ["1", 0],
            "width": 480,
            "height": 832,
            "num_frames": 41,
            "noise_aug_strength": 0.0,
            "start_latent_strength": 1.0,
            "end_latent_strength": 1.0,
            "force_offload": True,
            "load_device": ["8", 1]
        }
    },

    # 10. Sample video
    "10": {
        "class_type": "WanVideoSamplerMultiGPU",
        "inputs": {
            "model": ["8", 0],
            "compute_device": ["8", 1],
            "image_embeds": ["9", 0],
            "text_embeds": ["3", 0],
            "steps": 30,
            "cfg": 5.0,
            "shift": 5.0,
            "seed": random.randint(0, 2**32),
            "force_offload": True,
            "scheduler": "unipc",
            "riflex_freq_index": 0
        }
    },

    # 11. Decode to video frames
    "11": {
        "class_type": "WanVideoDecodeMultiGPU",
        "inputs": {
            "vae": ["4", 0],
            "samples": ["10", 0],
            "load_device": ["8", 1],
            "enable_vae_tiling": True,
            "tile_x": 272,
            "tile_y": 272,
            "tile_stride_x": 192,
            "tile_stride_y": 192
        }
    },

    # 12. Save video
    "12": {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": ["11", 0],
            "frame_rate": 8,
            "loop_count": 0,
            "filename_prefix": "wan_distorch_multigpu",
            "format": "video/h264-mp4",
            "save_output": True,
            "pingpong": False
        }
    }
}

print("=" * 60)
print("🚀 Wan2.2 I2V - DisTorch MultiGPU v3")
print("=" * 60)
print("📊 GPU Layout:")
print("   cuda:0 (3060): Compute + VAE + T5 + CLIP")
print("   cuda:1 (5060 Ti): BlockSwap storage (40 blocks)")
print()
print("🎬 480x832, 41 frames, 30 steps, LowNoise")
print("=" * 60)

try:
    response = requests.post("http://localhost:8188/prompt", json={"prompt": workflow})
    result = response.json()
    if "prompt_id" in result:
        print(f"✅ Queued! Prompt ID: {result['prompt_id']}")
    else:
        errors = result.get('node_errors', {})
        for node_id, node_err in list(errors.items())[:2]:
            print(f"❌ Node {node_id}: {node_err.get('class_type')}")
            for e in node_err.get('errors', [])[:2]:
                print(f"   - {e.get('details')}: {e.get('message')}")
except Exception as e:
    print(f"❌ Failed: {e}")
