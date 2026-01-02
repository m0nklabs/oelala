#!/usr/bin/env python3
"""
Quick Edge-Finding Test
========================
Tests the boundaries of each parameter to find hardware limits fast.
"""

import json
import time
import random
import urllib.request
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

# Force unbuffered output for nohup
os.environ['PYTHONUNBUFFERED'] = '1'

COMFY_API = "http://127.0.0.1:8188"
WORKFLOW_PATH = Path("/home/flip/oelala/ComfyUI/user/default/workflows/WAN2.2-I2V-DisTorch2-NEW_api.json")
RESULTS_FILE = Path("/home/flip/oelala/benchmark_results/edge_tests.json")
RESULTS_FILE.parent.mkdir(exist_ok=True)

# Max GPU allocation
GPU_ALLOCATION = "cuda:0,12gb;cuda:1,16gb;cpu,*"

# Known working configs (from earlier tests)
KNOWN_WORKING = [
    (720, 400, 101),    # ~4 min
    (1024, 576, 101),   # ~8.5 min
]

# Edge tests - push each parameter to find limits
EDGE_TESTS = [
    # === RESOLUTION EDGES (at 101 frames) ===
    # Push resolution UP from known working
    (1280, 720, 101, "HD 16:9 - resolution edge"),
    (1440, 810, 101, "1440p-ish - above HD"),
    
    # === FRAME EDGES (at 720x400 safe resolution) ===
    # Push frames UP
    (720, 400, 145, "145 frames (~6s video)"),
    (720, 400, 193, "193 frames (~8s video)"),
    (720, 400, 241, "241 frames (~10s video)"),
    
    # === ASPECT RATIO EDGES (at similar pixel count ~288k) ===
    # Portrait
    (576, 1024, 101, "9:16 portrait"),
    (720, 1280, 101, "HD portrait - edge"),
    
    # Square
    (720, 720, 101, "1:1 square"),
    
    # Ultrawide
    (1280, 544, 101, "21:9 ultrawide"),
    
    # === COMBINED EDGES ===
    # Can we do HD at fewer frames?
    (1280, 720, 65, "HD @ 65 frames"),
    (1280, 720, 49, "HD @ 49 frames"),
    
    # Can we do more frames at lower res?
    (576, 320, 193, "Low res @ 193 frames"),
]


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
        
        if "expert_mode_allocations" in node.get("inputs", {}):
            node["inputs"]["expert_mode_allocations"] = GPU_ALLOCATION
        
        if node.get("class_type") == "KSamplerAdvanced":
            node["inputs"]["noise_seed"] = random.randint(0, 2**32 - 1)
    
    return workflow


def queue_prompt(workflow):
    try:
        data = json.dumps({"prompt": workflow}).encode()
        req = urllib.request.Request(
            f"{COMFY_API}/prompt",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            return result.get("prompt_id")
    except Exception as e:
        print(f"  ❌ Queue failed: {e}")
        return None


def wait_for_completion(prompt_id, timeout=1800):
    start_time = time.time()
    peak_gpu0, peak_gpu1 = 0, 0
    last_report = 0
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            return "timeout", elapsed, peak_gpu0, peak_gpu1, "Timed out"
        
        gpu0, gpu1 = get_gpu_memory()
        peak_gpu0 = max(peak_gpu0, gpu0)
        peak_gpu1 = max(peak_gpu1, gpu1)
        
        if int(elapsed) >= last_report + 30:
            last_report = int(elapsed)
            print(f"    ⏳ {int(elapsed)}s - VRAM: {gpu0}+{gpu1}={gpu0+gpu1}MB")
        
        try:
            with urllib.request.urlopen(f"{COMFY_API}/queue", timeout=5) as resp:
                queue = json.loads(resp.read())
            
            running = queue.get("queue_running", [])
            pending = queue.get("queue_pending", [])
            is_active = any(p[1] == prompt_id for p in running + pending)
            
            if not is_active:
                with urllib.request.urlopen(f"{COMFY_API}/history/{prompt_id}", timeout=5) as resp:
                    history = json.loads(resp.read())
                
                if prompt_id in history:
                    status = history[prompt_id].get("status", {})
                    if status.get("completed", False):
                        return "success", elapsed, peak_gpu0, peak_gpu1, None
                    elif status.get("status_str") == "error":
                        msgs = str(status.get("messages", []))
                        if "out of memory" in msgs.lower() or "oom" in msgs.lower():
                            return "oom", elapsed, peak_gpu0, peak_gpu1, msgs[:200]
                        return "error", elapsed, peak_gpu0, peak_gpu1, msgs[:200]
        except:
            pass
        
        time.sleep(2)


def run_edge_tests():
    print(f"\n{'='*70}")
    print(f"🎯 EDGE-FINDING BENCHMARK - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   Goal: Find hardware limits for resolution, frames, aspect ratios")
    print(f"   GPU Allocation: {GPU_ALLOCATION}")
    print(f"{'='*70}")
    
    results = []
    oom_configs = []  # Track where OOM happened
    
    for width, height, frames, description in EDGE_TESTS:
        pixels = width * height
        
        # Skip if we already know a smaller config OOMed
        should_skip = False
        for oom_w, oom_h, oom_f in oom_configs:
            oom_pixels = oom_w * oom_h
            # Skip if more pixels at same frames, or more frames at same pixels
            if (pixels >= oom_pixels and frames >= oom_f):
                should_skip = True
                print(f"\n⏭️ Skipping {width}x{height}@{frames}f - beyond OOM limit")
                break
        
        if should_skip:
            results.append({
                "width": width, "height": height, "frames": frames,
                "description": description, "status": "skipped",
                "reason": "Beyond known OOM limit"
            })
            continue
        
        print(f"\n{'='*70}")
        print(f"🎬 {description}")
        print(f"   Config: {width}x{height} @ {frames} frames ({pixels} pixels)")
        print(f"{'='*70}")
        
        workflow = update_workflow(width, height, frames)
        prompt_id = queue_prompt(workflow)
        
        if not prompt_id:
            results.append({
                "width": width, "height": height, "frames": frames,
                "description": description, "status": "error",
                "error": "Failed to queue"
            })
            continue
        
        status, elapsed, peak0, peak1, error = wait_for_completion(prompt_id)
        
        result = {
            "width": width,
            "height": height,
            "frames": frames,
            "pixels": pixels,
            "description": description,
            "status": status,
            "time_sec": round(elapsed, 1),
            "gpu0_peak_mb": peak0,
            "gpu1_peak_mb": peak1,
            "total_vram_mb": peak0 + peak1,
            "timestamp": datetime.now().isoformat(),
        }
        if error:
            result["error"] = error
        
        results.append(result)
        
        # Report
        if status == "success":
            print(f"\n  ✅ SUCCESS in {elapsed:.1f}s")
            print(f"     Peak VRAM: {peak0} + {peak1} = {peak0+peak1}MB")
        elif status == "oom":
            print(f"\n  💥 OOM after {elapsed:.1f}s")
            print(f"     Peak VRAM: {peak0} + {peak1} = {peak0+peak1}MB")
            oom_configs.append((width, height, frames))
        else:
            print(f"\n  ❌ {status}: {error}")
        
        # Save after each test
        save_results(results)
    
    # Final summary
    print_summary(results)
    return results


def save_results(results):
    data = {
        "gpu_allocation": GPU_ALLOCATION,
        "results": results,
        "last_updated": datetime.now().isoformat(),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def print_summary(results):
    print(f"\n{'='*70}")
    print(f"📊 EDGE TEST SUMMARY")
    print(f"{'='*70}")
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] in ["oom", "error"]]
    skipped = [r for r in results if r["status"] == "skipped"]
    
    print(f"\n  Total: {len(results)}")
    print(f"  ✅ Success: {len(successful)}")
    print(f"  💥 OOM/Error: {len(failed)}")
    print(f"  ⏭️ Skipped: {len(skipped)}")
    
    if successful:
        print(f"\n🏆 WORKING CONFIGURATIONS:")
        for r in sorted(successful, key=lambda x: x["pixels"] * x["frames"], reverse=True):
            print(f"   {r['width']}x{r['height']} @ {r['frames']}f = {r['time_sec']}s ({r['description']})")
        
        # Find limits
        max_res = max(successful, key=lambda x: x["pixels"])
        max_frames = max(successful, key=lambda x: x["frames"])
        
        print(f"\n📐 DISCOVERED LIMITS:")
        print(f"   Max Resolution: {max_res['width']}x{max_res['height']} ({max_res['pixels']} pixels)")
        print(f"   Max Frames: {max_frames['frames']} frames @ {max_frames['width']}x{max_frames['height']}")
    
    if failed:
        print(f"\n💥 OOM/FAILED AT:")
        for r in failed:
            print(f"   {r['width']}x{r['height']} @ {r['frames']}f ({r['description']})")
    
    print(f"\n💾 Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    run_edge_tests()
