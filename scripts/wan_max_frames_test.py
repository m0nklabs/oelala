#!/usr/bin/env python3
"""
MAX FRAMES TEST - Find the longest video possible at 16fps
==========================================================
Tests at low resolution to find maximum frame count before OOM.
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
RESULTS_FILE = Path("/home/flip/oelala/benchmark_results/max_frames.json")
RESULTS_FILE.parent.mkdir(exist_ok=True)

GPU_ALLOCATION = "cuda:0,12gb;cuda:1,16gb"

# Test at 720x400 (known working) - push frames to the max
# 16fps internal, so frames/16 = seconds
FRAME_TESTS = [
    # Start high, binary search down on OOM
    (720, 400, 241, "241 frames (~15s video)"),
    (720, 400, 289, "289 frames (~18s video)"),
    (720, 400, 337, "337 frames (~21s video)"),
    (720, 400, 385, "385 frames (~24s video)"),
    (720, 400, 433, "433 frames (~27s video)"),
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


def wait_for_completion(prompt_id, timeout=3600):
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


def run_max_frames_test():
    print(f"\n{'='*70}")
    print(f"🎬 MAX FRAMES TEST - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   Goal: Find longest video possible at 720x400 @ 16fps")
    print(f"   GPU Allocation: {GPU_ALLOCATION}")
    print(f"{'='*70}")
    
    results = []
    max_working_frames = 101  # Known working
    oom_at_frames = None
    
    for width, height, frames, description in FRAME_TESTS:
        # Skip if we already know this will OOM
        if oom_at_frames and frames >= oom_at_frames:
            print(f"\n⏭️ Skipping {frames} frames - beyond OOM limit ({oom_at_frames})")
            results.append({
                "width": width, "height": height, "frames": frames,
                "description": description, "status": "skipped",
                "reason": f"Beyond OOM limit of {oom_at_frames} frames"
            })
            continue
        
        video_sec = frames / 16
        print(f"\n{'='*70}")
        print(f"🎬 {description}")
        print(f"   Config: {width}x{height} @ {frames} frames = {video_sec:.1f}s video @ 16fps")
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
            "video_seconds": round(video_sec, 1),
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
            print(f"     Video: {video_sec:.1f}s @ 16fps")
            print(f"     Peak VRAM: {peak0} + {peak1} = {peak0+peak1}MB")
            max_working_frames = max(max_working_frames, frames)
        elif status == "oom":
            print(f"\n  💥 OOM after {elapsed:.1f}s")
            print(f"     Peak VRAM: {peak0} + {peak1} = {peak0+peak1}MB")
            oom_at_frames = frames
        else:
            print(f"\n  ❌ {status}: {error}")
        
        # Save after each test
        save_results(results, max_working_frames, oom_at_frames)
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"📊 MAX FRAMES SUMMARY")
    print(f"{'='*70}")
    print(f"\n  🏆 Maximum working: {max_working_frames} frames = {max_working_frames/16:.1f}s video")
    if oom_at_frames:
        print(f"  💥 OOM at: {oom_at_frames} frames = {oom_at_frames/16:.1f}s video")
    print(f"\n💾 Results saved to: {RESULTS_FILE}")
    
    return results


def save_results(results, max_frames, oom_at):
    data = {
        "gpu_allocation": GPU_ALLOCATION,
        "resolution": "720x400",
        "max_working_frames": max_frames,
        "max_video_seconds": round(max_frames / 16, 1),
        "oom_at_frames": oom_at,
        "results": results,
        "last_updated": datetime.now().isoformat(),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    run_max_frames_test()
