#!/usr/bin/env python3
"""
Test ComfyUI Multi-GPU Wan2.2 I2V workflow
Uses WanVideoWrapper MultiGPU nodes to distribute model across:
- cuda:0 (5060 Ti 16GB): DiT models (compute)
- cuda:1 (3060 12GB): T5 + VAE (support)
"""

import json
import urllib.request
import urllib.error
import random
import time
import sys
from pathlib import Path

COMFYUI_URL = "http://localhost:8188"

# Multi-GPU Wan2.2 I2V workflow
MULTIGPU_WORKFLOW = {
    "1": {
        "class_type": "LoadWanVideoT5TextEncoderMultiGPU",
        "inputs": {
            "model_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "precision": "bf16",
            "device": "cuda:0",  # T5 on 3060
            "quantization": "disabled"
        }
    },
    "2": {
        "class_type": "WanVideoVAELoaderMultiGPU",
        "inputs": {
            "model_name": "Wan2.1_VAE.safetensors",
            "load_device": "cuda:0",  # VAE on 3060
            "precision": "bf16"
        }
    },
    "3": {
        "class_type": "WanVideoModelLoaderMultiGPU",
        "inputs": {
            "model": "Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf",
            "base_precision": "bf16",
            "quantization": "disabled",
            "load_device": "offload_device",
            "compute_device": "cuda:1",  # DiT on 5060 Ti
            "attention_mode": "sdpa"
        }
    },
    "4": {
        "class_type": "LoadImage",
        "inputs": {
            "image": "woman_portrait.png"  # Use the image we prepared earlier
        }
    },
    "5": {
        "class_type": "WanVideoTextEncodeMultiGPU",
        "inputs": {
            "wan_t5_model": ["1", 0],
            "prompt": "A woman with long blonde hair turns slowly to face the camera, gentle smile, soft indoor lighting, photorealistic, cinematic",
            "negative_prompt": "blurry, static, worst quality, low quality, deformed",
            "force_offload": False
        }
    },
    "6": {
        "class_type": "WanVideoImageToVideoEncodeMultiGPU",
        "inputs": {
            "width": 480,
            "height": 832,
            "num_frames": 21,
            "force_offload": False,
            "noise_aug_strength": 0.0,
            "start_latent_strength": 1.0,
            "end_latent_strength": 1.0,
            "start_image": ["4", 0],
            "vae": ["2", 0],
            "load_device": ["2", 1]
        }
    },
    "7": {
        "class_type": "WanVideoSamplerMultiGPU",
        "inputs": {
            "model": ["3", 0],
            "compute_device": ["3", 1],
            "image_embeds": ["6", 0],
            "positive": ["5", 0],
            "negative": ["5", 1],
            "steps": 20,
            "cfg": 5.0,
            "shift": 8.0,
            "seed": random.randint(0, 2**32),
            "force_offload": True,
            "scheduler": "unipc"
        }
    },
    "8": {
        "class_type": "WanVideoDecodeMultiGPU",
        "inputs": {
            "vae": ["2", 0],
            "load_device": ["2", 1],
            "samples": ["7", 0],
            "enable_vae_tiling": True,
            "tile_x": 272,
            "tile_y": 272,
            "tile_stride_x": 144,
            "tile_stride_y": 128
        }
    },
    "9": {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": ["8", 0],
            "frame_rate": 8,
            "loop_count": 0,
            "filename_prefix": "wan_multigpu",
            "format": "video/h264-mp4",
            "pingpong": False,
            "save_output": True
        }
    }
}


def queue_prompt(workflow):
    """Queue a workflow for execution"""
    prompt = {"prompt": workflow, "client_id": "multigpu_test"}
    data = json.dumps(prompt).encode('utf-8')

    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=data,
        headers={'Content-Type': 'application/json'}
    )

    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        print(f"❌ HTTP Error {e.code}: {error_body}")
        return None


def check_queue():
    """Check queue status"""
    try:
        with urllib.request.urlopen(f"{COMFYUI_URL}/queue") as response:
            return json.loads(response.read())
    except:
        return None


def main():
    print("=" * 60)
    print("🚀 ComfyUI Multi-GPU Wan2.2 I2V Test")
    print("=" * 60)
    print()
    print("GPU Distribution:")
    print("  cuda:0 (3060 12GB): T5 Encoder + VAE + Block Swap")
    print("  cuda:1 (5060 Ti 16GB): DiT Model (Q4_K_M GGUF)")
    print()

    # Check if input image exists
    input_image = Path("/home/flip/oelala/ComfyUI/input/woman_portrait.png")
    if not input_image.exists():
        print(f"⚠️  Input image not found: {input_image}")
        print("   Creating test image...")
        # Create a simple test image
        import subprocess
        subprocess.run([
            "convert", "-size", "480x832", "xc:#336699",
            "-font", "DejaVu-Sans", "-pointsize", "30",
            "-fill", "white", "-gravity", "center",
            "-annotate", "0", "Test Image",
            str(input_image)
        ], check=False)

    print(f"📸 Input: {input_image.name}")
    print("🎬 Output: wan_multigpu_*.mp4")
    print()

    # Queue the workflow
    print("⏳ Queueing workflow...")
    result = queue_prompt(MULTIGPU_WORKFLOW)

    if result:
        prompt_id = result.get('prompt_id', 'unknown')
        print(f"✅ Queued! Prompt ID: {prompt_id}")
        print()
        print("📊 Monitor progress at: http://localhost:8188")
        print("   Or check GPU usage: nvidia-smi -l 2")

        # Wait and check queue
        print()
        print("⏳ Waiting for completion...")

        while True:
            time.sleep(5)
            queue = check_queue()
            if queue:
                running = len(queue.get('queue_running', []))
                pending = len(queue.get('queue_pending', []))
                if running == 0 and pending == 0:
                    print("✅ Done! Check output folder for video.")
                    break
                print(f"   Running: {running}, Pending: {pending}")
            else:
                print("   Checking...")
    else:
        print("❌ Failed to queue workflow")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
