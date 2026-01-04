#!/usr/bin/env python3
"""
RESOLUTION TEST BY ASPECT RATIO
================================
Tests max resolutions for each aspect ratio at 101 frames.
"""

import json
import time
import random
import urllib.request
import subprocess
import os
from pathlib import Path
from datetime import datetime

os.environ['PYTHONUNBUFFERED'] = '1'

COMFY_API = "http://127.0.0.1:8188"
WORKFLOW_PATH = Path("/home/flip/oelala/ComfyUI/user/default/workflows/WAN2.2-I2V-DisTorch2-NEW_api.json")
RESULTS_FILE = Path("/home/flip/oelala/benchmark_results/resolution_tests.json")
RESULTS_FILE.parent.mkdir(exist_ok=True)

GPU_ALLOCATION = "cuda:0,12gb;cuda:1,16gb;cpu,*"
FRAMES = 101  # Standard frame count for resolution tests

# Resolution tests grouped by aspect ratio
# Format: (width, height, aspect_ratio_name, description)
RESOLUTION_TESTS = [
    # === 16:9 LANDSCAPE (most common) ===
    (1280, 720, "16:9", "720p HD"),
    (1440, 810, "16:9", "900p"),
    (1600, 900, "16:9", "900p+"),
    (1920, 1080, "16:9", "1080p Full HD"),
    
    # === 9:16 PORTRAIT (TikTok/Reels) ===
    (720, 1280, "9:16", "720p Portrait"),
    (810, 1440, "9:16", "900p Portrait"),
    (1080, 1920, "9:16", "1080p Portrait"),
    
    # === 1:1 SQUARE (Instagram) ===
    (720, 720, "1:1", "720p Square"),
    (1024, 1024, "1:1", "1024 Square"),
    (1280, 1280, "1:1", "1280 Square"),
    
    # === 4:3 CLASSIC ===
    (1024, 768, "4:3", "XGA"),
    (1280, 960, "4:3", "SXGA-"),
    (1440, 1080, "4:3", "1440x1080"),
    
    # === 21:9 ULTRAWIDE (Cinematic) ===
    (1280, 544, "21:9", "Ultrawide 1280"),
    (1680, 720, "21:9", "Ultrawide 1680"),
    (2560, 1080, "21:9", "UWFHD"),
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


def wait_for_completion(prompt_id, timeout=2400):
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


def run_resolution_tests():
    print(f"\n{'='*70}")
    print(f"📐 RESOLUTION TEST BY ASPECT RATIO - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   Frames: {FRAMES} | GPU: {GPU_ALLOCATION}")
    print(f"{'='*70}")
    
    results = []
    oom_pixels = {}  # Track OOM per aspect ratio
    
    for width, height, aspect, description in RESOLUTION_TESTS:
        pixels = width * height
        
        # Skip if we already OOMed at lower pixels for this aspect ratio
        if aspect in oom_pixels and pixels >= oom_pixels[aspect]:
            print(f"\n⏭️ Skipping {width}x{height} ({aspect}) - beyond OOM limit")
            results.append({
                "width": width, "height": height, "aspect": aspect,
                "description": description, "status": "skipped",
                "reason": f"Beyond OOM limit for {aspect}"
            })
            continue
        
        print(f"\n{'='*70}")
        print(f"📐 {aspect} - {description}")
        print(f"   Config: {width}x{height} @ {FRAMES} frames ({pixels:,} pixels)")
        print(f"{'='*70}")
        
        workflow = update_workflow(width, height, FRAMES)
        prompt_id = queue_prompt(workflow)
        
        if not prompt_id:
            results.append({
                "width": width, "height": height, "aspect": aspect,
                "description": description, "status": "error",
                "error": "Failed to queue"
            })
            continue
        
        status, elapsed, peak0, peak1, error = wait_for_completion(prompt_id)
        
        result = {
            "width": width,
            "height": height,
            "pixels": pixels,
            "aspect": aspect,
            "frames": FRAMES,
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
        
        if status == "success":
            print(f"\n  ✅ SUCCESS in {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print(f"     Peak VRAM: {peak0} + {peak1} = {peak0+peak1}MB")
        elif status == "oom":
            print(f"\n  💥 OOM after {elapsed:.1f}s")
            print(f"     Peak VRAM: {peak0} + {peak1} = {peak0+peak1}MB")
            oom_pixels[aspect] = pixels
        else:
            print(f"\n  ❌ {status}: {error}")
        
        save_results(results)
    
    print_summary(results)
    return results


def save_results(results):
    # Group by aspect ratio
    by_aspect = {}
    for r in results:
        asp = r.get("aspect", "unknown")
        if asp not in by_aspect:
            by_aspect[asp] = []
        by_aspect[asp].append(r)
    
    data = {
        "gpu_allocation": GPU_ALLOCATION,
        "frames": FRAMES,
        "by_aspect_ratio": by_aspect,
        "all_results": results,
        "last_updated": datetime.now().isoformat(),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def print_summary(results):
    print(f"\n{'='*70}")
    print(f"📊 RESOLUTION TEST SUMMARY")
    print(f"{'='*70}")
    
    # Group by aspect ratio
    by_aspect = {}
    for r in results:
        asp = r.get("aspect", "unknown")
        if asp not in by_aspect:
            by_aspect[asp] = {"success": [], "oom": [], "other": []}
        if r["status"] == "success":
            by_aspect[asp]["success"].append(r)
        elif r["status"] == "oom":
            by_aspect[asp]["oom"].append(r)
        else:
            by_aspect[asp]["other"].append(r)
    
    print(f"\n🏆 MAX WORKING RESOLUTION PER ASPECT RATIO:")
    for asp, data in by_aspect.items():
        if data["success"]:
            best = max(data["success"], key=lambda x: x["pixels"])
            print(f"   {asp}: {best['width']}x{best['height']} ({best['pixels']:,} px) - {best['time_sec']:.0f}s")
        elif data["oom"]:
            oom = data["oom"][0]
            print(f"   {asp}: OOM at {oom['width']}x{oom['height']}")
        else:
            print(f"   {asp}: No successful tests")
    
    print(f"\n💾 Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    run_resolution_tests()
