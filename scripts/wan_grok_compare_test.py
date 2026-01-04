#!/usr/bin/env python3
"""WAN 2.2 test met Grok video specs: 464x688 portrait"""

import json
import urllib.request
import time
import random
import subprocess
from pathlib import Path

COMFY_API = "http://127.0.0.1:8188"
WORKFLOW_PATH = Path("/home/flip/oelala/ComfyUI/user/default/workflows/WAN2.2-I2V-DisTorch2-NEW_api.json")

# Grok specs: 464×688 @ 6sec 24fps = 144 frames
# WAN uses 16fps, 6sec = 96 frames (but needs 4n+1, so 97)
WIDTH = 464
HEIGHT = 688
FRAMES = 97  # ~6 sec @ 16fps

def get_gpu_memory():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        mems = [int(x.strip()) for x in result.stdout.strip().split("\n")]
        return tuple(mems) if len(mems) >= 2 else (0, 0)
    except:
        return (0, 0)

def update_workflow(width, height, frames):
    with open(WORKFLOW_PATH) as f:
        workflow = json.load(f)
    
    for nid, node in workflow.items():
        if node.get("class_type") == "WanImageToVideo":
            node["inputs"]["width"] = width
            node["inputs"]["height"] = height
            node["inputs"]["length"] = frames
        
        # Keep CPU offload for proper dual-GPU distribution
        if "expert_mode_allocations" in node.get("inputs", {}):
            node["inputs"]["expert_mode_allocations"] = "cuda:0,12gb;cuda:1,16gb;cpu,*"
        
        if node.get("class_type") == "KSamplerAdvanced":
            node["inputs"]["noise_seed"] = random.randint(0, 2**32 - 1)
        
        if node.get("class_type") == "VHS_VideoCombine":
            node["inputs"]["filename_prefix"] = f"grok_compare_{width}x{height}_{frames}f"
    
    return workflow

def queue_prompt(workflow):
    data = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(
        f"{COMFY_API}/prompt",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        result = json.loads(resp.read())
        return result.get("prompt_id")

if __name__ == "__main__":
    print(f"🎬 Grok Compare Test: {WIDTH}×{HEIGHT} @ {FRAMES} frames ({FRAMES/16:.1f}s @ 16fps)")
    print(f"📊 Grok specs: 464×688 @ 144 frames (6s @ 24fps) = 46M pixels")
    print(f"📊 Our test: {WIDTH * HEIGHT * FRAMES / 1_000_000:.1f}M pixels")
    print(f"📊 GPU allocation: cuda:0,12gb;cuda:1,16gb;cpu,*")
    
    gpu0, gpu1 = get_gpu_memory()
    print(f"📊 Starting VRAM: {gpu0}+{gpu1}={gpu0+gpu1}MB")
    
    workflow = update_workflow(WIDTH, HEIGHT, FRAMES)
    prompt_id = queue_prompt(workflow)
    print(f"✅ Queued: {prompt_id}")
    print(f"⏳ Monitor: watch -n5 'nvidia-smi --query-gpu=memory.used --format=csv'")
