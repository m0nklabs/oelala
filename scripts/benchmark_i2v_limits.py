#!/usr/bin/env python3
"""
DisTorch2 I2V Benchmark Script v3

Tests the limits of video generation with different:
- CPU offload amounts
- Resolutions  
- Frame counts (video length)
- Video duration (frames / fps)

Improvements in v3:
- Verifies output video actually exists
- Tracks actual video duration (not just frame count)
- Uses SFW image (beach_real.png)
- Better error detection
"""

import json
import time
import requests
import subprocess
import glob
from pathlib import Path
from datetime import datetime

COMFYUI_URL = "http://localhost:8188"
WORKFLOW_PATH = Path(__file__).parent.parent / "workflows/ImageToVideo/sfw_i2v_distorch2_api.json"
OUTPUT_DIR = Path(__file__).parent.parent / "ComfyUI/output"
RESULTS_FILE = Path(__file__).parent.parent / "data/benchmark_results/i2v_limits_benchmark.json"

# SFW image for benchmarks
SFW_IMAGE = "beach_real.png"

# Default FPS (can be varied in tests)
DEFAULT_FPS = 16

# Test configurations: (width, height, frames, cpu_offload_gb, fps, description)
# Video duration = frames / fps
BENCHMARK_CONFIGS = [
    # Baseline - 5 sec video @ 16fps
    (576, 1024, 81, 1, 16, "baseline_576x1024_5s"),
    
    # CPU offload comparison @ baseline
    (576, 1024, 81, 2, 16, "576x1024_5s_2gb"),
    (576, 1024, 81, 4, 16, "576x1024_5s_4gb"),
    
    # 720p tests - 5 sec
    (720, 1280, 81, 2, 16, "720x1280_5s_2gb"),
    (720, 1280, 81, 4, 16, "720x1280_5s_4gb"),
    (720, 1280, 81, 6, 16, "720x1280_5s_6gb"),
    
    # Longer videos at 576p
    (576, 1024, 121, 2, 16, "576x1024_7.5s_2gb"),   # 7.5 sec
    (576, 1024, 161, 4, 16, "576x1024_10s_4gb"),    # 10 sec
    (576, 1024, 241, 6, 16, "576x1024_15s_6gb"),    # 15 sec
    
    # 1080p tests
    (1080, 1920, 41, 6, 16, "1080x1920_2.5s_6gb"),  # 2.5 sec @ 1080p
    (1080, 1920, 81, 8, 16, "1080x1920_5s_8gb"),    # 5 sec @ 1080p
    
    # Ultra long at lower res
    (480, 848, 321, 4, 16, "480x848_20s_4gb"),      # 20 sec at 480p
    (480, 848, 481, 6, 16, "480x848_30s_6gb"),      # 30 sec at 480p
]


def build_allocation_string(cpu_gb: float) -> str:
    """Build DisTorch2 allocation string with given CPU offload."""
    cuda0_gb = 11  # RTX 3060
    cuda1_gb = 15  # RTX 5060 Ti
    return f"cuda:0,{cuda0_gb}gb;cuda:1,{cuda1_gb}gb;cpu,{cpu_gb}gb"


def load_workflow():
    """Load the base workflow."""
    with open(WORKFLOW_PATH) as f:
        return json.load(f)


def modify_workflow(workflow: dict, width: int, height: int, frames: int, 
                    cpu_gb: float, fps: int, name: str) -> dict:
    """Modify workflow for benchmark configuration."""
    wf = json.loads(json.dumps(workflow))  # Deep copy
    
    allocation = build_allocation_string(cpu_gb)
    
    for node_id, node in wf.items():
        if not isinstance(node, dict):
            continue
            
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})
        
        # Update resolution in WanImageToVideo
        if class_type == "WanImageToVideo":
            inputs["width"] = width
            inputs["height"] = height
            inputs["length"] = frames
            
        # Update allocation in all DisTorch2 loaders
        if "DisTorch2" in class_type and "expert_mode_allocations" in inputs:
            inputs["expert_mode_allocations"] = allocation
            
        # Update output filename and fps
        if class_type == "VHS_VideoCombine":
            inputs["filename_prefix"] = f"bench_{name}"
            inputs["frame_rate"] = fps
            
        # Use SFW image
        if class_type == "LoadImage":
            inputs["image"] = SFW_IMAGE
    
    return wf


def queue_prompt(workflow: dict) -> str:
    """Queue workflow and return prompt_id."""
    resp = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow})
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def get_job_status(prompt_id: str) -> dict | None:
    """Get job status from history."""
    try:
        resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
        if resp.status_code == 200:
            history = resp.json()
            if prompt_id in history:
                return history[prompt_id]
    except:
        pass
    return None


def check_job_error(status: dict) -> str | None:
    """Check if job has error status."""
    if not status:
        return None
    job_status = status.get("status", {})
    if job_status.get("status_str") == "error":
        messages = job_status.get("messages", [])
        if messages:
            # Extract error message
            for msg in messages:
                if isinstance(msg, list) and len(msg) > 1:
                    error_text = str(msg[1])
                    if "CUDA out of memory" in error_text or "OutOfMemoryError" in error_text:
                        return f"OOM: {error_text[:200]}"
                    return error_text[:200]
        return "Unknown error"
    return None


def find_output_video(name: str) -> Path | None:
    """Find the output video file for this benchmark."""
    pattern = str(OUTPUT_DIR / f"bench_{name}_*.mp4")
    files = glob.glob(pattern)
    if files:
        # Return most recent
        return Path(max(files, key=lambda f: Path(f).stat().st_mtime))
    return None


def get_video_info(video_path: Path) -> dict | None:
    """Get video metadata using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames,r_frame_rate,duration",
            "-of", "json",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            stream = data.get("streams", [{}])[0]
            
            # Parse frame rate (can be "16/1" format)
            fps_str = stream.get("r_frame_rate", "16/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
                
            return {
                "width": int(stream.get("width", 0)),
                "height": int(stream.get("height", 0)),
                "nb_frames": int(stream.get("nb_frames", 0)),
                "fps": round(fps, 2),
                "duration": float(stream.get("duration", 0)),
                "file_size": video_path.stat().st_size,
            }
    except Exception as e:
        print(f"      ⚠️ ffprobe error: {e}")
    return None


def wait_for_completion(prompt_id: str, name: str, timeout: int = 2400) -> dict:
    """Wait for job completion with proper error detection."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        status = get_job_status(prompt_id)
        
        if status:
            # Check for error
            error = check_job_error(status)
            if error:
                return {
                    "success": False,
                    "duration": time.time() - start_time,
                    "error": error
                }
            
            # Check if completed (has outputs)
            outputs = status.get("outputs", {})
            if outputs:
                # Job completed - verify video exists
                time.sleep(2)  # Give file system time to flush
                video = find_output_video(name)
                if video:
                    video_info = get_video_info(video)
                    return {
                        "success": True,
                        "duration": time.time() - start_time,
                        "video_path": str(video),
                        "video_info": video_info
                    }
                else:
                    return {
                        "success": False,
                        "duration": time.time() - start_time,
                        "error": "Job completed but no output video found"
                    }
        
        time.sleep(5)
    
    return {"success": False, "duration": timeout, "error": "timeout"}


def run_benchmark(config: tuple) -> dict:
    """Run a single benchmark configuration."""
    width, height, frames, cpu_gb, fps, name = config
    video_duration = frames / fps
    
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {name}")
    print(f"   Resolution: {width}x{height}")
    print(f"   Frames: {frames} @ {fps}fps = {video_duration:.1f}s video")
    print(f"   CPU offload: {cpu_gb}GB")
    print(f"{'='*60}")
    
    result = {
        "name": name,
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "target_video_duration": video_duration,
        "cpu_offload_gb": cpu_gb,
        "allocation": build_allocation_string(cpu_gb),
        "started_at": datetime.now().isoformat(),
    }
    
    try:
        workflow = load_workflow()
        modified_wf = modify_workflow(workflow, width, height, frames, cpu_gb, fps, name)
        
        print("   📤 Queueing workflow...")
        prompt_id = queue_prompt(modified_wf)
        result["prompt_id"] = prompt_id
        
        print("   ⏳ Waiting for completion...")
        completion = wait_for_completion(prompt_id, name, timeout=2400)
        
        result["success"] = completion["success"]
        result["generation_seconds"] = round(completion["duration"], 2)
        result["generation_minutes"] = round(completion["duration"] / 60, 2)
        
        if completion["success"]:
            result["video_path"] = completion.get("video_path")
            video_info = completion.get("video_info", {})
            result["actual_video"] = video_info
            
            # Calculate throughput
            total_pixels = width * height * frames
            result["total_pixels"] = total_pixels
            result["pixels_per_second"] = round(total_pixels / completion["duration"])
            result["megapixels_per_second"] = round(total_pixels / completion["duration"] / 1_000_000, 3)
            
            # Verify video matches expected
            if video_info:
                actual_frames = video_info.get("nb_frames", 0)
                actual_w = video_info.get("width", 0)
                actual_h = video_info.get("height", 0)
                actual_duration = video_info.get("duration", 0)
                
                if actual_w != width or actual_h != height:
                    result["warning"] = f"Resolution mismatch: expected {width}x{height}, got {actual_w}x{actual_h}"
                if abs(actual_frames - frames) > 2:
                    result["warning"] = f"Frame count mismatch: expected {frames}, got {actual_frames}"
                    
                print(f"   ✅ Completed in {result['generation_minutes']:.1f} min")
                print(f"      📹 Video: {actual_w}x{actual_h}, {actual_frames}f, {actual_duration:.1f}s")
                print(f"      💾 Size: {video_info.get('file_size', 0) / 1024 / 1024:.1f}MB")
            else:
                print(f"   ✅ Completed in {result['generation_minutes']:.1f} min (no video info)")
        else:
            error = completion.get("error", "Unknown error")
            result["error"] = error
            if "oom" in error.lower() or "memory" in error.lower():
                print(f"   💥 OOM ERROR: {error[:80]}")
            else:
                print(f"   ❌ Failed: {error[:80]}")
            
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        print(f"   ❌ Exception: {e}")
    
    result["finished_at"] = datetime.now().isoformat()
    return result


def load_existing_results() -> list:
    """Load existing results if any."""
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE) as f:
                return json.load(f)
        except:
            pass
    return []


def save_results(results: list):
    """Save results to file."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def print_summary(results: list):
    """Print benchmark summary."""
    print("\n" + "="*70)
    print("  BENCHMARK SUMMARY")
    print("="*70)
    
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    oom = [r for r in failed if "oom" in r.get("error", "").lower() or "memory" in r.get("error", "").lower()]
    
    print(f"\n✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)} (OOM: {len(oom)})")
    
    if successful:
        print("\n📊 Successful Configurations:")
        print("-"*70)
        print(f"{'Name':<25} {'Resolution':<12} {'Video':<8} {'Gen Time':<10} {'MP/s':<8}")
        print("-"*70)
        
        for r in sorted(successful, key=lambda x: x.get("total_pixels", 0), reverse=True):
            video_dur = r.get("target_video_duration", 0)
            gen_min = r.get("generation_minutes", 0)
            mps = r.get("megapixels_per_second", 0)
            print(f"{r['name']:<25} {r['width']}x{r['height']:<5} {video_dur:>5.1f}s   {gen_min:>6.1f}min   {mps:>6.3f}")
    
    if failed:
        print("\n❌ Failed Configurations:")
        print("-"*70)
        for r in failed:
            error_short = r.get("error", "?")[:40]
            print(f"  {r['name']}: {error_short}")
    
    print(f"\n📁 Results: {RESULTS_FILE}")


def main():
    print("="*70)
    print("  DisTorch2 I2V Limits Benchmark v3")
    print("="*70)
    print(f"\n📋 Configurations: {len(BENCHMARK_CONFIGS)}")
    print(f"🖼️  SFW Image: {SFW_IMAGE}")
    print(f"📂 Output: {OUTPUT_DIR}")
    print()
    
    # Verify SFW image exists
    sfw_path = Path(__file__).parent.parent / "ComfyUI/input" / SFW_IMAGE
    if not sfw_path.exists():
        print(f"❌ SFW image not found: {sfw_path}")
        return
    
    # Load existing results to skip completed
    results = load_existing_results()
    completed_names = {r["name"] for r in results if r.get("success")}
    
    # Clear failed results to retry them
    results = [r for r in results if r.get("success")]
    
    for i, config in enumerate(BENCHMARK_CONFIGS, 1):
        name = config[5]  # name is 6th element now
        
        if name in completed_names:
            print(f"\n[{i}/{len(BENCHMARK_CONFIGS)}] ⏭️  Skipping {name} (already done)")
            continue
            
        print(f"\n[{i}/{len(BENCHMARK_CONFIGS)}]", end="")
        result = run_benchmark(config)
        results.append(result)
        save_results(results)  # Save after each test
    
    print_summary(results)


if __name__ == "__main__":
    main()
