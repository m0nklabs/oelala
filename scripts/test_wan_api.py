#!/usr/bin/env python3
"""Test Wan2.2 I2V GGUF via ComfyUI API"""
import json
import requests
import time
import sys

COMFYUI_URL = "http://localhost:8188"

# Workflow using wan_config.json settings:
# - prompt: "A woman with long blonde hair turns slowly..."
# - 480x832 portrait, 21 frames, 20 steps
workflow = {
    "1": {
        "class_type": "UnetLoaderGGUF",
        "inputs": {"unet_name": "Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf"}
    },
    "3": {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"}
    },
    "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "A woman with long blonde hair turns slowly to face the camera, gentle smile, soft indoor lighting, photorealistic, 8k", "clip": ["3", 0]}
    },
    "5": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "low quality, blurry, distorted, anime, cartoon", "clip": ["3", 0]}
    },
    "6": {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "Wan2.1_VAE.safetensors"}
    },
    "7": {
        "class_type": "LoadImage",
        "inputs": {"image": "woman_portrait.png"}
    },
    "8": {
        "class_type": "WanImageToVideo",
        "inputs": {
            "positive": ["4", 0],
            "negative": ["5", 0],
            "vae": ["6", 0],
            "start_image": ["7", 0],
            "width": 480,
            "height": 832,
            "length": 21,
            "batch_size": 1
        }
    },
    "9": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "seed": 42,
            "steps": 20,
            "cfg": 5.0,
            "sampler_name": "euler_ancestral",
            "scheduler": "normal",
            "positive": ["8", 0],
            "negative": ["8", 1],
            "latent_image": ["8", 2],
            "denoise": 1.0
        }
    },
    "10": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["9", 0], "vae": ["6", 0]}
    },
    "11": {
        "class_type": "SaveImage",
        "inputs": {"images": ["10", 0], "filename_prefix": "wan_woman"}
    }
}

def queue_prompt(prompt):
    """Queue a prompt for execution"""
    p = {"prompt": prompt}
    resp = requests.post(f"{COMFYUI_URL}/prompt", json=p)
    return resp.json()

def get_history(prompt_id):
    """Get execution history"""
    resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
    return resp.json()

def main():
    print("🎬 Testing Wan2.2 I2V GGUF via ComfyUI API...")
    print("📦 Queueing workflow...")

    try:
        result = queue_prompt(workflow)
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            if "node_errors" in result:
                for node, err in result.get("node_errors", {}).items():
                    print(f"   Node {node}: {err}")
            return 1

        prompt_id = result.get("prompt_id")
        print(f"✅ Queued! Prompt ID: {prompt_id}")
        print("⏳ Waiting for execution (this may take a few minutes)...")

        # Poll for completion
        start_time = time.time()
        while True:
            history = get_history(prompt_id)
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                if outputs:
                    print(f"\n✅ Done in {time.time() - start_time:.1f}s!")
                    print(f"📁 Output: {outputs}")
                    return 0
                status = history[prompt_id].get("status", {})
                if status.get("status_str") == "error":
                    print(f"\n❌ Execution failed: {status}")
                    return 1

            elapsed = time.time() - start_time
            if elapsed > 600:  # 10 min timeout
                print("\n⏰ Timeout after 10 minutes")
                return 1

            # Show progress
            sys.stdout.write(f"\r⏳ Running... {elapsed:.0f}s")
            sys.stdout.flush()
            time.sleep(5)

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
