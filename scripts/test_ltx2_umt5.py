#!/usr/bin/env python3
"""Test LTX-2 with UMT5 text encoder.

⚠️ KNOWN ISSUE: LTX-2 requires ~32GB VRAM with UMT5, but we only have 28GB.
This test will likely OOM. See docs/COMFYUI_INVENTORY.md for details.

Options to make this work:
1. Download ltx-2-19b-dev-fp4.safetensors (20GB instead of 27GB)
2. Use diffusers with aggressive memory management
3. Wait for smaller quantized models

Usage:
    python scripts/test_ltx2_umt5.py
"""
import json
import time
import requests
import uuid

COMFYUI_URL = "http://localhost:8188"

# LTX-2 T2V workflow with UMT5 (smaller text encoder)
WORKFLOW = {
    # 1. Load LTX-2 checkpoint with native text encoder (UMT5)
    "checkpoint": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "ltx-2-19b-distilled-fp8.safetensors"
        }
    },
    # 2. Load UMT5 text encoder with native loader
    "text_encoder": {
        "class_type": "LTXAVTextEncoderLoader",
        "inputs": {
            "text_encoder": "umt5_xxl_fp8_e4m3fn.safetensors",
            "ckpt_name": "ltx-2-19b-distilled-fp8.safetensors",
            "device": "default"
        }
    },
    # 3. Positive prompt
    "positive_prompt": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "A fluffy orange cat walking on a sunny beach with gentle waves in the background, 4k, cinematic",
            "clip": ["text_encoder", 0]
        }
    },
    # 4. Negative prompt
    "negative_prompt": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "blurry, low quality, distorted, deformed",
            "clip": ["text_encoder", 0]
        }
    },
    # 5. Apply STG to model (required for LTX-2)
    "apply_stg": {
        "class_type": "LTXVApplySTG",
        "inputs": {
            "model": ["checkpoint", 0],
            "block_indices": "14, 19"
        }
    },
    # 6. Create empty latent video (small for testing: 512x320, 17 frames)
    "empty_latent": {
        "class_type": "EmptyLTXVLatentVideo",
        "inputs": {
            "width": 512,
            "height": 320,
            "length": 17,  # Minimal frames for fast test
            "batch_size": 1
        }
    },
    # 7. LTX Conditioning (adds frame rate)
    "conditioning": {
        "class_type": "LTXVConditioning",
        "inputs": {
            "positive": ["positive_prompt", 0],
            "negative": ["negative_prompt", 0],
            "frame_rate": 24.0
        }
    },
    # 8. Sample with KSampler
    "sampler": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["apply_stg", 0],
            "seed": 42,
            "steps": 8,  # Low for distilled model
            "cfg": 3.5,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": ["conditioning", 0],
            "negative": ["conditioning", 1],
            "latent_image": ["empty_latent", 0],
            "denoise": 1.0
        }
    },
    # 9. VAE Decode
    "vae_decode": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["sampler", 0],
            "vae": ["checkpoint", 2]
        }
    },
    # 10. Save video
    "save_video": {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": ["vae_decode", 0],
            "frame_rate": 24,
            "loop_count": 0,
            "filename_prefix": "ltx2_umt5_test",
            "format": "video/h264-mp4",
            "pingpong": False,
            "save_output": True
        }
    }
}


def queue_workflow(prompt: dict) -> tuple[str, bool]:
    """Queue a workflow and return (prompt_id, success)."""
    payload = {"prompt": prompt}
    try:
        resp = requests.post(f"{COMFYUI_URL}/prompt", json=payload, timeout=10)
        data = resp.json()
        if "error" in data:
            print(f"❌ Workflow error: {data['error']}")
            if "node_errors" in data:
                for node_id, errors in data["node_errors"].items():
                    print(f"   Node {node_id}: {errors}")
            return "", False
        return data.get("prompt_id", ""), True
    except Exception as e:
        print(f"❌ Failed to queue: {e}")
        return "", False


def wait_for_completion(prompt_id: str, timeout: int = 300) -> bool:
    """Poll for workflow completion."""
    print(f"⏳ Waiting for workflow {prompt_id[:8]}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=5)
            history = resp.json()
            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("completed", False):
                    print("✅ Workflow completed!")
                    return True
                if status.get("status_str") == "error":
                    print(f"❌ Workflow failed: {status}")
                    return False
        except Exception as e:
            print(f"⚠️ Poll error: {e}")
        time.sleep(2)
    print("❌ Timeout waiting for completion")
    return False


def check_gpu_usage():
    """Print current GPU memory usage."""
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    print("🔧 GPU Usage:")
    for line in result.stdout.strip().split("\n"):
        print(f"   {line}")


def main():
    print("🚀 Testing LTX-2 with Gemma 3 12B Text Encoder")
    print("=" * 50)
    
    # Check ComfyUI is running
    try:
        resp = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        print(f"✅ ComfyUI is running")
    except:
        print("❌ ComfyUI is not running!")
        return
    
    check_gpu_usage()
    print()
    
    # Queue workflow
    print("📤 Queueing LTX-2 T2V workflow...")
    prompt_id, success = queue_workflow(WORKFLOW)
    if not success:
        return
    
    print(f"✅ Queued: {prompt_id}")
    
    # Wait for completion
    if wait_for_completion(prompt_id, timeout=600):
        print()
        check_gpu_usage()
        print()
        print("🎬 Video should be saved in ComfyUI output folder!")
    else:
        print("Check ComfyUI logs: journalctl -u comfyui -n 100")


if __name__ == "__main__":
    main()
