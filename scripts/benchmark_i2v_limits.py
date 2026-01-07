#!/usr/bin/env python3
"""
DisTorch2 I2V Benchmark Script

Tests the limits of video generation with different:
- CPU offload amounts
- Resolutions
- Frame counts (video length)

Current baseline: 576x1024, 81 frames, 1gb CPU offload (~6.5 min)
"""

import json
import time
import requests
import sys
from pathlib import Path
from datetime import datetime

COMFYUI_URL = "http://localhost:8188"
WORKFLOW_PATH = Path(__file__).parent.parent / "workflows/ImageToVideo/sfw_i2v_distorch2_api.json"
OUTPUT_DIR = Path(__file__).parent.parent / "ComfyUI/output"
RESULTS_FILE = Path(__file__).parent.parent / "data/benchmark_results/i2v_limits_benchmark.json"

# Test configurations: (width, height, frames, cpu_offload_gb, description)
BENCHMARK_CONFIGS = [
    # Baseline
    (576, 1024, 81, 1, "baseline_576x1024_81f_1gb"),
    
    # More CPU offload at baseline resolution
    (576, 1024, 81, 2, "576x1024_81f_2gb_cpu"),
    (576, 1024, 81, 4, "576x1024_81f_4gb_cpu"),
    
    # Higher resolution with more CPU offload
    (720, 1280, 81, 2, "720x1280_81f_2gb_cpu"),
    (720, 1280, 81, 4, "720x1280_81f_4gb_cpu"),
    (720, 1280, 81, 6, "720x1280_81f_6gb_cpu"),
    
    # Longer videos at baseline resolution
    (576, 1024, 121, 2, "576x1024_121f_2gb_cpu"),  # ~7.5 sec
    (576, 1024, 161, 4, "576x1024_161f_4gb_cpu"),  # ~10 sec
    (576, 1024, 241, 6, "576x1024_241f_6gb_cpu"),  # ~15 sec
    
    # Push the limits - 1080p
    (1080, 1920, 41, 6, "1080x1920_41f_6gb_cpu"),   # Short 1080p
    (1080, 1920, 81, 8, "1080x1920_81f_8gb_cpu"),   # Full 1080p
    
    # Ultra long at lower res
    (480, 848, 321, 4, "480x848_321f_4gb_cpu"),     # ~20 sec at 480p
    (480, 848, 481, 6, "480x848_481f_6gb_cpu"),     # ~30 sec at 480p
]

def build_allocation_string(cpu_gb: float) -> str:
    """Build DisTorch2 allocation string with given CPU offload."""
    # Total available: cuda:0 = 12GB, cuda:1 = 16GB
    # Reserve some for overhead
    cuda0_gb = 11  # RTX 3060
    cuda1_gb = 15  # RTX 5060 Ti
    return f"cuda:0,{cuda0_gb}gb;cuda:1,{cuda1_gb}gb;cpu,{cpu_gb}gb"


def load_workflow():
    """Load the base workflow."""
    with open(WORKFLOW_PATH) as f:
        return json.load(f)


def modify_workflow(workflow: dict, width: int, height: int, frames: int, cpu_gb: float) -> dict:
    """Modify workflow for benchmark configuration."""
    wf = json.loads(json.dumps(workflow))  # Deep copy
    
    allocation = build_allocation_string(cpu_gb)
    
    for node_id, node in wf.items():
        if not isinstance(node, dict):
            continue
            
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})
        
        # Update resolution in EmptyWanLatentVideo
        if class_type == "EmptyWanLatentVideo":
            inputs["width"] = width
            inputs["height"] = height
            inputs["length"] = frames
            
        # Update allocation in all DisTorch2 loaders
        if "DisTorch2" in class_type and "expert_mode_allocations" in inputs:
            inputs["expert_mode_allocations"] = allocation
            
        # Update output filename
        if class_type == "VHS_VideoCombine":
            inputs["filename_prefix"] = f"benchmark_{width}x{height}_{frames}f_{cpu_gb}gb"
    
    return wf


def queue_prompt(workflow: dict) -> str:
    """Queue workflow and return prompt_id."""
    resp = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow})
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def wait_for_completion(prompt_id: str, timeout: int = 1800) -> dict:
    """Wait for job completion, return timing info."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
        if resp.status_code == 200:
            history = resp.json()
            if prompt_id in history:
                return {
                    "success": True,
                    "duration": time.time() - start_time,
                    "history": history[prompt_id]
                }
        time.sleep(5)
    
    return {"success": False, "duration": timeout, "error": "timeout"}


def check_for_errors(prompt_id: str) -> str | None:
    """Check if there are any errors in the queue."""
    try:
        resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
        if resp.status_code == 200:
            history = resp.json()
            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("status_str") == "error":
                    return status.get("messages", [["Unknown error"]])[0][1] if status.get("messages") else "Unknown error"
    except:
        pass
    return None


def run_benchmark(config: tuple) -> dict:
    """Run a single benchmark configuration."""
    width, height, frames, cpu_gb, name = config
    
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {name}")
    print(f"   Resolution: {width}x{height}, Frames: {frames}, CPU: {cpu_gb}GB")
    print(f"{'='*60}")
    
    result = {
        "name": name,
        "width": width,
        "height": height,
        "frames": frames,
        "cpu_offload_gb": cpu_gb,
        "allocation": build_allocation_string(cpu_gb),
        "started_at": datetime.now().isoformat(),
    }
    
    try:
        workflow = load_workflow()
        modified_wf = modify_workflow(workflow, width, height, frames, cpu_gb)
        
        print("   📤 Queueing workflow...")
        prompt_id = queue_prompt(modified_wf)
        result["prompt_id"] = prompt_id
        
        print("   ⏳ Waiting for completion...")
        completion = wait_for_completion(prompt_id, timeout=1800)  # 30 min max
        
        result["success"] = completion["success"]
        result["duration_seconds"] = completion["duration"]
        result["duration_minutes"] = round(completion["duration"] / 60, 2)
        
        if completion["success"]:
            print(f"   ✅ Completed in {result['duration_minutes']} minutes")
            
            # Calculate pixels/second throughput
            total_pixels = width * height * frames
            result["total_pixels"] = total_pixels
            result["pixels_per_second"] = round(total_pixels / completion["duration"])
            result["megapixels_per_second"] = round(total_pixels / completion["duration"] / 1_000_000, 3)
        else:
            error = completion.get("error") or check_for_errors(prompt_id)
            result["error"] = error
            print(f"   ❌ Failed: {error}")
            
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        print(f"   ❌ Exception: {e}")
    
    result["finished_at"] = datetime.now().isoformat()
    return result


def main():
    print("="*70)
    print("  DisTorch2 I2V Limits Benchmark")
    print("="*70)
    print(f"\nConfigurations to test: {len(BENCHMARK_CONFIGS)}")
    print(f"Workflow: {WORKFLOW_PATH}")
    print(f"Results will be saved to: {RESULTS_FILE}")
    
    # Check ComfyUI is running
    try:
        resp = requests.get(f"{COMFYUI_URL}/system_stats")
        resp.raise_for_status()
        print("\n✅ ComfyUI is running")
    except:
        print("\n❌ ComfyUI not reachable at", COMFYUI_URL)
        sys.exit(1)
    
    # Ensure results directory exists
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing results if any
    all_results = []
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)
        print(f"📂 Loaded {len(all_results)} existing results")
    
    # Filter out already-tested configs
    tested_names = {r["name"] for r in all_results}
    configs_to_run = [c for c in BENCHMARK_CONFIGS if c[4] not in tested_names]
    
    print(f"🧪 Running {len(configs_to_run)} new configurations")
    
    for i, config in enumerate(configs_to_run):
        print(f"\n[{i+1}/{len(configs_to_run)}] ", end="")
        result = run_benchmark(config)
        all_results.append(result)
        
        # Save after each test
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2)
        
        # If OOM or error, we might want to skip higher configs
        if not result["success"] and "memory" in str(result.get("error", "")).lower():
            print("   ⚠️  OOM detected, skipping remaining higher configs in this category")
            # Continue anyway, some later configs might work with different tradeoffs
    
    # Print summary
    print("\n" + "="*70)
    print("  BENCHMARK SUMMARY")
    print("="*70)
    
    successful = [r for r in all_results if r.get("success")]
    failed = [r for r in all_results if not r.get("success")]
    
    print(f"\n✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        print("\n📊 Successful Configurations (sorted by throughput):")
        print("-" * 70)
        print(f"{'Name':<35} {'Res':<12} {'Frames':<8} {'Time':<8} {'MP/s':<8}")
        print("-" * 70)
        
        for r in sorted(successful, key=lambda x: x.get("megapixels_per_second", 0), reverse=True):
            res = f"{r['width']}x{r['height']}"
            print(f"{r['name']:<35} {res:<12} {r['frames']:<8} {r['duration_minutes']:<8.1f} {r.get('megapixels_per_second', 0):<8.3f}")
    
    if failed:
        print("\n❌ Failed Configurations:")
        for r in failed:
            print(f"  - {r['name']}: {r.get('error', 'Unknown error')[:60]}")
    
    print(f"\n📁 Full results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
