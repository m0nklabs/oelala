#!/usr/bin/env python3
"""
DisTorch2 I2V Benchmark - 15 Second Video Focus

Doel: Vind de configuratie die 15 seconden video mogelijk maakt.

Aanpak:
- 15 sec @ 16fps = 241 frames
- Test verschillende resoluties (lager = minder VRAM)
- Test hogere CPU offload (tot 16GB)
- Test lagere frame rates (minder frames voor zelfde duratie)
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
RESULTS_FILE = Path(__file__).parent.parent / "data/benchmark_results/i2v_15sec_benchmark.json"

SFW_IMAGE = "beach_real.png"

# Test configurations for 15 second videos
# Format: (width, height, frames, cpu_offload_gb, fps, description)
# Video duration = frames / fps

BENCHMARK_CONFIGS = [
    # === Strategy 1: Lower resolution @ 16fps (241 frames) ===
    (384, 680, 241, 8, 16, "384x680_15s_8gb"),      # Very low res - BASELINE
    (432, 768, 241, 8, 16, "432x768_15s_8gb"),      # Low res
    (480, 848, 241, 8, 16, "480x848_15s_8gb"),      # 480p
    (480, 848, 241, 12, 16, "480x848_15s_12gb"),    # 480p more CPU
    (512, 912, 241, 10, 16, "512x912_15s_10gb"),    # Between 480p and 576p
    
    # === Strategy 2: Lower FPS for higher quality ===
    # 15 sec @ 12fps = 181 frames
    (576, 1024, 181, 6, 12, "576x1024_15s_12fps_6gb"),
    (576, 1024, 181, 8, 12, "576x1024_15s_12fps_8gb"),
    (720, 1280, 181, 8, 12, "720x1280_15s_12fps_8gb"),
    
    # 15 sec @ 10fps = 151 frames (should work - 161f worked before)
    (576, 1024, 151, 4, 10, "576x1024_15s_10fps_4gb"),
    (576, 1024, 151, 6, 10, "576x1024_15s_10fps_6gb"),
    (720, 1280, 151, 6, 10, "720x1280_15s_10fps_6gb"),
    (720, 1280, 151, 8, 10, "720x1280_15s_10fps_8gb"),
    
    # 15 sec @ 8fps = 121 frames (definitely works)
    (720, 1280, 121, 4, 8, "720x1280_15s_8fps_4gb"),
    (720, 1280, 121, 6, 8, "720x1280_15s_8fps_6gb"),
    (1080, 1920, 121, 8, 8, "1080x1920_15s_8fps_8gb"),
    (1080, 1920, 121, 12, 8, "1080x1920_15s_8fps_12gb"),
    
    # === Strategy 3: Push CPU offload higher ===
    (576, 1024, 241, 10, 16, "576x1024_15s_10gb"),
    (576, 1024, 241, 12, 16, "576x1024_15s_12gb"),
    (576, 1024, 241, 16, 16, "576x1024_15s_16gb"),
    
    # === Interesting edge cases ===
    # 20 sec @ 8fps = 161 frames (should work at 720p)
    (720, 1280, 161, 6, 8, "720x1280_20s_8fps_6gb"),
    # 30 sec @ 8fps = 241 frames at lower res
    (480, 848, 241, 8, 8, "480x848_30s_8fps_8gb"),
]


def build_allocation_string(cpu_gb: float) -> str:
    """Build DisTorch2 allocation string with given CPU offload."""
    cuda0_gb = 11  # RTX 3060
    cuda1_gb = 15  # RTX 5060 Ti
    return f"cuda:0,{cuda0_gb}gb;cuda:1,{cuda1_gb}gb;cpu,{cpu_gb}gb"


def load_workflow():
    with open(WORKFLOW_PATH) as f:
        return json.load(f)


def modify_workflow(workflow: dict, width: int, height: int, frames: int, 
                    cpu_gb: float, fps: int, name: str) -> dict:
    wf = json.loads(json.dumps(workflow))
    allocation = build_allocation_string(cpu_gb)
    
    for node_id, node in wf.items():
        if not isinstance(node, dict):
            continue
            
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})
        
        if class_type == "WanImageToVideo":
            inputs["width"] = width
            inputs["height"] = height
            inputs["length"] = frames
            
        if "DisTorch2" in class_type and "expert_mode_allocations" in inputs:
            inputs["expert_mode_allocations"] = allocation
            
        if class_type == "VHS_VideoCombine":
            inputs["filename_prefix"] = f"bench15s_{name}"
            inputs["frame_rate"] = fps
            
        if class_type == "LoadImage":
            inputs["image"] = SFW_IMAGE
    
    return wf


def queue_prompt(workflow: dict) -> str:
    resp = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow})
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def get_job_status(prompt_id: str) -> dict | None:
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
    if not status:
        return None
    job_status = status.get("status", {})
    if job_status.get("status_str") == "error":
        messages = job_status.get("messages", [])
        if messages:
            for msg in messages:
                if isinstance(msg, list) and len(msg) > 1:
                    error_text = str(msg[1])
                    if "CUDA out of memory" in error_text or "OutOfMemoryError" in error_text:
                        return f"OOM: {error_text[:200]}"
                    return error_text[:200]
        return "Unknown error"
    return None


def find_output_video(name: str) -> Path | None:
    pattern = str(OUTPUT_DIR / f"bench15s_{name}_*.mp4")
    files = glob.glob(pattern)
    if files:
        return Path(max(files, key=lambda f: Path(f).stat().st_mtime))
    return None


def get_video_info(video_path: Path) -> dict | None:
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


def wait_for_completion(prompt_id: str, name: str, timeout: int = 3600) -> dict:
    """Wait for job completion - longer timeout for bigger jobs."""
    start_time = time.time()
    last_video_check = None
    
    while time.time() - start_time < timeout:
        status = get_job_status(prompt_id)
        
        if status:
            error = check_job_error(status)
            if error:
                return {
                    "success": False,
                    "duration": time.time() - start_time,
                    "error": error
                }
            
            # Check for completion - look at status_str or outputs
            job_status = status.get("status", {})
            status_str = job_status.get("status_str", "")
            outputs = status.get("outputs", {})
            
            # Job is done if status is success OR we have outputs
            is_complete = status_str == "success" or bool(outputs)
            
            if is_complete:
                time.sleep(2)  # Give filesystem time
                video = find_output_video(name)
                if video:
                    # Check if this is a NEW video (not cached from before)
                    if last_video_check is None or video.stat().st_mtime > last_video_check:
                        video_info = get_video_info(video)
                        return {
                            "success": True,
                            "duration": time.time() - start_time,
                            "video_path": str(video),
                            "video_info": video_info
                        }
                    else:
                        # Video exists but wasn't generated now - job was cached
                        # This is still a success, just fast
                        video_info = get_video_info(video)
                        return {
                            "success": True,
                            "duration": time.time() - start_time,
                            "video_path": str(video),
                            "video_info": video_info,
                            "cached": True
                        }
                else:
                    return {
                        "success": False,
                        "duration": time.time() - start_time,
                        "error": "Job completed but no output video found"
                    }
        
        # Track when we last checked for videos
        if last_video_check is None:
            last_video_check = time.time()
        
        time.sleep(5)
    
    return {"success": False, "duration": timeout, "error": "timeout"}


def run_benchmark(config: tuple) -> dict:
    width, height, frames, cpu_gb, fps, name = config
    video_duration = frames / fps
    
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {name}")
    print(f"   Resolution: {width}x{height}")
    print(f"   Frames: {frames} @ {fps}fps = {video_duration:.1f}s video")
    print(f"   CPU offload: {cpu_gb}GB")
    print(f"   Allocation: {build_allocation_string(cpu_gb)}")
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
        completion = wait_for_completion(prompt_id, name, timeout=3600)
        
        result["success"] = completion["success"]
        result["generation_seconds"] = round(completion["duration"], 2)
        result["generation_minutes"] = round(completion["duration"] / 60, 2)
        
        if completion["success"]:
            result["video_path"] = completion.get("video_path")
            video_info = completion.get("video_info", {})
            result["actual_video"] = video_info
            
            total_pixels = width * height * frames
            result["total_pixels"] = total_pixels
            result["pixels_per_second"] = round(total_pixels / completion["duration"])
            result["megapixels_per_second"] = round(total_pixels / completion["duration"] / 1_000_000, 3)
            
            if video_info:
                actual_duration = video_info.get("duration", 0)
                print(f"   ✅ Completed in {result['generation_minutes']:.1f} min")
                print(f"      📹 Video: {video_info.get('width')}x{video_info.get('height')}, {video_info.get('nb_frames')}f @ {video_info.get('fps')}fps")
                print(f"      ⏱️  Duration: {actual_duration:.1f}s")
                print(f"      💾 Size: {video_info.get('file_size', 0) / 1024 / 1024:.1f}MB")
        else:
            error = completion.get("error", "Unknown error")
            result["error"] = error
            if "oom" in error.lower() or "memory" in error.lower():
                print(f"   💥 OOM ERROR")
            else:
                print(f"   ❌ Failed: {error[:60]}")
            
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        print(f"   ❌ Exception: {e}")
    
    result["finished_at"] = datetime.now().isoformat()
    return result


def load_existing_results() -> list:
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE) as f:
                return json.load(f)
        except:
            pass
    return []


def save_results(results: list):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def print_summary(results: list):
    print("\n" + "="*70)
    print("  15 SECOND VIDEO BENCHMARK SUMMARY")
    print("="*70)
    
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    print(f"\n✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        print(f"\n🎉 WORKING 15-SEC CONFIGURATIONS:")
        print("-"*70)
        print(f"{'Name':<30} {'Resolution':<12} {'FPS':<6} {'Gen Time':<10}")
        print("-"*70)
        
        for r in sorted(successful, key=lambda x: (-x.get("width", 0), -x.get("fps", 0))):
            gen_min = r.get("generation_minutes", 0)
            fps = r.get("fps", 16)
            print(f"{r['name']:<30} {r['width']}x{r['height']:<5} {fps:<6} {gen_min:>6.1f}min")
        
        # Best option
        best = max(successful, key=lambda x: (x.get("width", 0) * x.get("height", 0), x.get("fps", 0)))
        print(f"\n⭐ BEST OPTION: {best['name']}")
        print(f"   {best['width']}x{best['height']} @ {best['fps']}fps = {best.get('target_video_duration', 15):.1f}s")
        print(f"   Generation time: {best.get('generation_minutes', 0):.1f} min")
    
    if failed:
        print(f"\n❌ Failed Configurations:")
        for r in failed:
            print(f"  {r['name']}: {r.get('error', '?')[:30]}")
    
    print(f"\n📁 Results: {RESULTS_FILE}")


def main():
    print("="*70)
    print("  15 SECOND VIDEO BENCHMARK")
    print("="*70)
    print(f"\n🎯 Goal: Find settings for 15-second videos")
    print(f"📋 Configurations: {len(BENCHMARK_CONFIGS)}")
    print(f"🖼️  Image: {SFW_IMAGE}")
    print()
    print("Strategies:")
    print("  1. Lower resolution (384p-480p) @ 16fps")
    print("  2. Lower FPS (8-12fps) @ higher resolution")
    print("  3. Higher CPU offload (8-16GB)")
    print()
    
    sfw_path = Path(__file__).parent.parent / "ComfyUI/input" / SFW_IMAGE
    if not sfw_path.exists():
        print(f"❌ SFW image not found: {sfw_path}")
        return
    
    results = load_existing_results()
    completed_names = {r["name"] for r in results if r.get("success")}
    results = [r for r in results if r.get("success")]
    
    for i, config in enumerate(BENCHMARK_CONFIGS, 1):
        name = config[5]
        
        if name in completed_names:
            print(f"\n[{i}/{len(BENCHMARK_CONFIGS)}] ⏭️  Skipping {name} (already done)")
            continue
            
        print(f"\n[{i}/{len(BENCHMARK_CONFIGS)}]", end="")
        result = run_benchmark(config)
        results.append(result)
        save_results(results)
    
    print_summary(results)


if __name__ == "__main__":
    main()
