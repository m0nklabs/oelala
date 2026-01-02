#!/usr/bin/env python3
"""
PORTRAIT RESOLUTION TEST - Find max portrait resolutions
=========================================================
Tests 9:16 portrait format at increasing resolutions.
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
RESULTS_FILE = Path("/home/flip/oelala/benchmark_results/portrait_test.json")
RESULTS_FILE.parent.mkdir(exist_ok=True)

GPU_ALLOCATION = "cuda:0,12gb;cuda:1,16gb;cpu,*"

# Portrait tests (9:16 aspect ratio) - sorted by pixel count
# Width must be divisible by 16, height by 16
PORTRAIT_TESTS = [
    # Standard portrait sizes - start low, go high
    (576, 1024, 81, "576x1024 (9:16) ~590k pixels"),      # Safe baseline
    (720, 1280, 81, "720x1280 (9:16) ~922k pixels"),      # HD portrait
    (768, 1344, 81, "768x1344 (9:17) ~1.03M pixels"),     # Above HD
    (864, 1536, 81, "864x1536 (9:16) ~1.33M pixels"),     # Higher
    (1080, 1920, 49, "1080x1920 (9:16) Full HD portrait @ 49f"),  # FHD with fewer frames
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


def run_portrait_test():
    print(f"\n{'='*70}")
    print(f"📱 PORTRAIT RESOLUTION TEST - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   Goal: Find max 9:16 portrait resolution")
    print(f"   GPU Allocation: {GPU_ALLOCATION}")
    print(f"{'='*70}")
    
    results = []
    max_working_pixels = 0
    max_working_res = None
    oom_at_pixels = None
    
    for width, height, frames, description in PORTRAIT_TESTS:
        pixels = width * height
        
        # Skip if we know a smaller resolution OOMed
        if oom_at_pixels and pixels >= oom_at_pixels:
            print(f"\n⏭️ Skipping {width}x{height} - beyond OOM limit")
            results.append({
                "width": width, "height": height, "frames": frames,
                "description": description, "status": "skipped"
            })
            continue
        
        print(f"\n{'='*70}")
        print(f"📱 {description}")
        print(f"   Config: {width}x{height} @ {frames}f ({pixels:,} pixels)")
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
            "aspect_ratio": "9:16",
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
            if pixels > max_working_pixels:
                max_working_pixels = pixels
                max_working_res = (width, height)
        elif status == "oom":
            print(f"\n  💥 OOM after {elapsed:.1f}s")
            print(f"     Peak VRAM: {peak0} + {peak1} = {peak0+peak1}MB")
            oom_at_pixels = pixels
        else:
            print(f"\n  ❌ {status}: {error}")
        
        save_results(results, max_working_res, max_working_pixels)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 PORTRAIT TEST SUMMARY")
    print(f"{'='*70}")
    if max_working_res:
        print(f"\n  🏆 Max working: {max_working_res[0]}x{max_working_res[1]} ({max_working_pixels:,} pixels)")
    
    successful = [r for r in results if r["status"] == "success"]
    for r in successful:
        print(f"     ✅ {r['width']}x{r['height']} @ {r['frames']}f = {r['time_sec']}s")
    
    print(f"\n💾 Results saved to: {RESULTS_FILE}")
    return results


def save_results(results, max_res, max_pixels):
    data = {
        "test_type": "portrait_9:16",
        "gpu_allocation": GPU_ALLOCATION,
        "max_working_resolution": f"{max_res[0]}x{max_res[1]}" if max_res else None,
        "max_working_pixels": max_pixels,
        "results": results,
        "last_updated": datetime.now().isoformat(),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    run_portrait_test()
