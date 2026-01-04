#!/usr/bin/env python3
"""
GPU-ONLY PORTRAIT TEST - No CPU offloading
===========================================
Tests portrait resolutions using ONLY GPU memory (no T5 on CPU).
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
RESULTS_FILE = Path("/home/flip/oelala/benchmark_results/gpu_only_test.json")
RESULTS_FILE.parent.mkdir(exist_ok=True)

# GPU-ONLY: No CPU fallback - all on GPU
GPU_ALLOCATION = "cuda:0,12gb;cuda:1,16gb"

# Start with known working portrait, see if GPU-only works
TESTS = [
    (576, 1024, 81, "576x1024 portrait GPU-only"),
    (720, 1280, 81, "720x1280 HD portrait GPU-only"),
    (720, 1280, 49, "720x1280 HD @ 49f GPU-only"),
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


def run_gpu_only_test():
    print(f"\n{'='*70}")
    print(f"🎮 GPU-ONLY TEST - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   Mode: NO CPU offloading - pure GPU")
    print(f"   GPU Allocation: {GPU_ALLOCATION}")
    print(f"{'='*70}")
    
    results = []
    
    for width, height, frames, description in TESTS:
        pixels = width * height
        
        print(f"\n{'='*70}")
        print(f"🎮 {description}")
        print(f"   Config: {width}x{height} @ {frames}f ({pixels:,} pixels)")
        print(f"   Allocation: {GPU_ALLOCATION}")
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
            "gpu_allocation": GPU_ALLOCATION,
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
            print(f"\n  💥 OOM after {elapsed:.1f}s - GPU-only not possible")
            print(f"     Peak VRAM: {peak0} + {peak1} = {peak0+peak1}MB")
            print(f"     💡 T5 encoder needs ~8GB, must use CPU offload for this config")
            break  # Stop testing higher configs
        else:
            print(f"\n  ❌ {status}: {error}")
        
        save_results(results)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 GPU-ONLY TEST SUMMARY")
    print(f"{'='*70}")
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "oom"]
    
    if successful:
        print(f"\n  ✅ Working GPU-only configs:")
        for r in successful:
            print(f"     {r['width']}x{r['height']} @ {r['frames']}f = {r['time_sec']}s")
    
    if failed:
        print(f"\n  💥 OOM without CPU offload:")
        for r in failed:
            print(f"     {r['width']}x{r['height']} @ {r['frames']}f")
        print(f"\n  💡 Recommendation: Use 'cuda:0,12gb;cuda:1,16gb;cpu,*' for these configs")
    
    print(f"\n💾 Results saved to: {RESULTS_FILE}")
    return results


def save_results(results):
    data = {
        "test_type": "gpu_only",
        "gpu_allocation": GPU_ALLOCATION,
        "note": "No CPU offloading - T5 encoder must fit in GPU memory",
        "results": results,
        "last_updated": datetime.now().isoformat(),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    run_gpu_only_test()
