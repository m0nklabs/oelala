#!/usr/bin/env python3
"""
WAN 2.2 Video Generation Test Suite
====================================
Systematically tests resolution, aspect ratio, frame count, and FPS limits.
Results are saved to JSON for documentation and frontend integration.

Usage:
    python wan_test_suite.py                    # Run full test suite
    python wan_test_suite.py --quick            # Quick test (subset)
    python wan_test_suite.py --resume           # Resume from last position
    python wan_test_suite.py --report           # Show results only
"""

import json
import time
import random
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

# Configuration
COMFY_API = "http://127.0.0.1:8188"
WORKFLOW_PATH = Path("/home/flip/oelala/ComfyUI/user/default/workflows/WAN2.2-I2V-DisTorch2-NEW_api.json")
RESULTS_PATH = Path("/home/flip/oelala/benchmark_results.json")
OUTPUT_DIR = Path("/home/flip/oelala/ComfyUI/output")

# GPU allocation (max GPU, T5 on CPU)
GPU_ALLOCATION = "cuda:0,12gb;cuda:1,16gb;cpu,*"

# Test configurations - ORDERED SMALL TO LARGE (stop at OOM)
ASPECT_RATIOS = {
    "16:9": [(720, 400), (848, 480), (1024, 576), (1280, 720)],
    "9:16": [(400, 720), (480, 848), (576, 1024), (720, 1280)],
    "4:3": [(640, 480), (768, 576), (960, 720)],
    "3:4": [(480, 640), (576, 768), (720, 960)],
    "1:1": [(480, 480), (576, 576), (720, 720)],
    "21:9": [(1024, 432), (1280, 544)],
    "9:21": [(432, 1024), (544, 1280)],
}

# Frame counts to test (WAN uses 4n+1 formula)
FRAME_COUNTS = [33, 49, 65, 81, 101, 121, 145, 169, 193, 241]

# FPS values (for output, not generation)
FPS_VALUES = [24, 30, 60]


@dataclass
class TestResult:
    """Single test result"""
    width: int
    height: int
    aspect_ratio: str
    frames: int
    duration_sec: float
    status: str  # "success", "oom", "error", "timeout"
    generation_time_sec: float
    gpu0_peak_mb: int
    gpu1_peak_mb: int
    error_message: Optional[str] = None
    output_file: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class TestSuiteConfig:
    """Test suite configuration"""
    model_name: str = "wan2.2_i2v_480p_14B_bf16.gguf"
    gpu_allocation: str = GPU_ALLOCATION
    sampler: str = "euler"
    scheduler: str = "beta"
    steps: int = 25
    cfg: float = 3.0
    shift: float = 8.0
    loras: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.loras is None:
            self.loras = []


class WanTestSuite:
    def __init__(self, config: TestSuiteConfig = None):
        self.config = config or TestSuiteConfig()
        self.results: List[TestResult] = []
        self.limits: Dict[str, Any] = {}  # Track discovered limits
        self.load_results()
    
    def load_results(self):
        """Load existing results from file"""
        if RESULTS_PATH.exists():
            try:
                with open(RESULTS_PATH) as f:
                    data = json.load(f)
                    self.results = [TestResult(**r) for r in data.get("results", [])]
                    self.limits = data.get("limits", {})
                    print(f"📂 Loaded {len(self.results)} existing results")
                    if self.limits:
                        print(f"📊 Known limits: {self.limits}")
            except Exception as e:
                print(f"⚠️ Could not load results: {e}")
    
    def save_results(self):
        """Save results to file"""
        data = {
            "config": asdict(self.config),
            "results": [asdict(r) for r in self.results],
            "limits": self.limits,
            "last_updated": datetime.now().isoformat(),
            "summary": self.get_summary()
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved {len(self.results)} results to {RESULTS_PATH}")
    
    def update_limit(self, param: str, value: Any, frames: int = 101):
        """Track discovered limits - don't test beyond these"""
        key = f"{param}@{frames}f"
        if key not in self.limits or value < self.limits[key]:
            self.limits[key] = value
            print(f"  📉 New limit discovered: {key} = {value}")
    
    def is_beyond_limit(self, width: int, height: int, frames: int) -> bool:
        """Check if test would exceed known limits"""
        pixels = width * height
        
        # Check resolution limit at this frame count
        res_key = f"max_pixels@{frames}f"
        if res_key in self.limits and pixels >= self.limits[res_key]:
            return True
        
        # Check frame limit at similar resolutions (within 20%)
        for key, limit_frames in self.limits.items():
            if key.startswith("max_frames@"):
                limit_pixels = int(key.split("@")[1].replace("px", ""))
                if abs(pixels - limit_pixels) / limit_pixels < 0.2:
                    if frames >= limit_frames:
                        return True
        
        return False
    
    def get_gpu_memory(self) -> tuple:
        """Get current GPU memory usage"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            mems = [int(x.strip()) for x in result.stdout.strip().split("\n")]
            return tuple(mems) if len(mems) >= 2 else (0, 0)
        except:
            return (0, 0)
    
    def update_workflow(self, width: int, height: int, frames: int) -> dict:
        """Update workflow with test parameters"""
        with open(WORKFLOW_PATH) as f:
            workflow = json.load(f)
        
        # Update WanImageToVideo node
        for nid, node in workflow.items():
            if node.get("class_type") == "WanImageToVideo":
                node["inputs"]["width"] = width
                node["inputs"]["height"] = height
                node["inputs"]["length"] = frames
            
            # Update GPU allocation
            if "expert_mode_allocations" in node.get("inputs", {}):
                node["inputs"]["expert_mode_allocations"] = self.config.gpu_allocation
            
            # Fresh seed
            if node.get("class_type") == "KSamplerAdvanced":
                node["inputs"]["noise_seed"] = random.randint(0, 2**32 - 1)
                node["inputs"]["steps"] = self.config.steps
                node["inputs"]["cfg"] = self.config.cfg
        
        return workflow
    
    def queue_prompt(self, workflow: dict) -> Optional[str]:
        """Queue workflow and return prompt_id"""
        import urllib.request
        import urllib.error
        
        data = json.dumps({"prompt": workflow}).encode()
        req = urllib.request.Request(
            f"{COMFY_API}/prompt",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                return result.get("prompt_id")
        except Exception as e:
            print(f"❌ Queue failed: {e}")
            return None
    
    def wait_for_completion(self, prompt_id: str, timeout: int = 1800) -> tuple:
        """Wait for prompt to complete, return (status, duration, peak_vram)"""
        import urllib.request
        
        start_time = time.time()
        peak_gpu0 = 0
        peak_gpu1 = 0
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                return ("timeout", elapsed, peak_gpu0, peak_gpu1, "Generation timed out")
            
            # Track peak VRAM
            gpu0, gpu1 = self.get_gpu_memory()
            peak_gpu0 = max(peak_gpu0, gpu0)
            peak_gpu1 = max(peak_gpu1, gpu1)
            
            # Check queue status
            try:
                with urllib.request.urlopen(f"{COMFY_API}/queue", timeout=5) as resp:
                    queue = json.loads(resp.read())
                    
                # Check if still running
                running = queue.get("queue_running", [])
                pending = queue.get("queue_pending", [])
                
                is_running = any(p[1] == prompt_id for p in running)
                is_pending = any(p[1] == prompt_id for p in pending)
                
                if not is_running and not is_pending:
                    # Check history for result
                    with urllib.request.urlopen(f"{COMFY_API}/history/{prompt_id}", timeout=5) as resp:
                        history = json.loads(resp.read())
                        
                    if prompt_id in history:
                        prompt_data = history[prompt_id]
                        if prompt_data.get("status", {}).get("completed", False):
                            return ("success", elapsed, peak_gpu0, peak_gpu1, None)
                        else:
                            # Check for errors
                            status = prompt_data.get("status", {})
                            if status.get("status_str") == "error":
                                msgs = status.get("messages", [])
                                error_msg = str(msgs) if msgs else "Unknown error"
                                if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
                                    return ("oom", elapsed, peak_gpu0, peak_gpu1, error_msg)
                                return ("error", elapsed, peak_gpu0, peak_gpu1, error_msg)
                    
            except Exception as e:
                pass  # Network hiccup, retry
            
            # Progress indicator
            if int(elapsed) % 30 == 0:
                print(f"  ⏳ {int(elapsed)}s - GPU0: {gpu0}MB, GPU1: {gpu1}MB")
            
            time.sleep(2)
    
    def run_single_test(self, width: int, height: int, frames: int, aspect_ratio: str) -> TestResult:
        """Run a single test configuration"""
        print(f"\n🎬 Testing {width}x{height} ({aspect_ratio}) @ {frames} frames...")
        
        # Check if already tested
        for r in self.results:
            if r.width == width and r.height == height and r.frames == frames:
                print(f"  ⏭️ Already tested, skipping")
                return r
        
        # Update and queue workflow
        workflow = self.update_workflow(width, height, frames)
        prompt_id = self.queue_prompt(workflow)
        
        if not prompt_id:
            result = TestResult(
                width=width, height=height, aspect_ratio=aspect_ratio,
                frames=frames, duration_sec=frames/24,
                status="error", generation_time_sec=0,
                gpu0_peak_mb=0, gpu1_peak_mb=0,
                error_message="Failed to queue prompt"
            )
            self.results.append(result)
            return result
        
        # Wait for completion
        status, elapsed, peak0, peak1, error = self.wait_for_completion(prompt_id)
        
        result = TestResult(
            width=width, height=height, aspect_ratio=aspect_ratio,
            frames=frames, duration_sec=frames/24,
            status=status, generation_time_sec=elapsed,
            gpu0_peak_mb=peak0, gpu1_peak_mb=peak1,
            error_message=error
        )
        
        self.results.append(result)
        self.save_results()
        
        # Report
        if status == "success":
            print(f"  ✅ SUCCESS in {elapsed:.1f}s - Peak VRAM: {peak0}MB + {peak1}MB")
        elif status == "oom":
            print(f"  💥 OOM after {elapsed:.1f}s - Peak VRAM: {peak0}MB + {peak1}MB")
        elif status == "timeout":
            print(f"  ⏰ TIMEOUT after {elapsed:.1f}s")
        else:
            print(f"  ❌ ERROR: {error}")
        
        return result
    
    def run_resolution_tests(self, quick: bool = False):
        """Test all resolutions at base frame count - SMALL TO LARGE, stop at OOM"""
        print("\n" + "="*60)
        print("📐 RESOLUTION TESTS (101 frames) - Small → Large")
        print("="*60)
        
        base_frames = 101
        max_working_pixels = 0  # Track largest successful resolution
        
        for aspect, resolutions in ASPECT_RATIOS.items():
            print(f"\n📏 Aspect Ratio: {aspect}")
            
            test_resolutions = resolutions[:2] if quick else resolutions
            
            for width, height in test_resolutions:
                pixels = width * height
                
                # Skip if beyond known limit
                if self.is_beyond_limit(width, height, base_frames):
                    print(f"  ⏭️ Skipping {width}x{height} - beyond known limit")
                    continue
                
                result = self.run_single_test(width, height, base_frames, aspect)
                
                if result.status == "success":
                    max_working_pixels = max(max_working_pixels, pixels)
                elif result.status == "oom":
                    # Record the limit - OOM at this pixel count
                    self.update_limit("max_pixels", pixels, base_frames)
                    print(f"  🛑 OOM at {width}x{height} ({pixels} pixels) - skipping larger resolutions")
                    break  # Stop this aspect ratio, continue with others (they might have different pixel counts)
    
    def run_frame_count_tests(self, quick: bool = False):
        """Test frame counts at a stable resolution - stop at OOM"""
        print("\n" + "="*60)
        print("🎞️ FRAME COUNT TESTS (720x400) - Low → High")
        print("="*60)
        
        # Use a resolution we know works
        width, height = 720, 400
        aspect = "16:9"
        pixels = width * height
        
        test_frames = FRAME_COUNTS[:4] if quick else FRAME_COUNTS
        
        for frames in test_frames:
            # Skip if beyond known limit
            if self.is_beyond_limit(width, height, frames):
                print(f"  ⏭️ Skipping {frames} frames - beyond known limit")
                continue
            
            result = self.run_single_test(width, height, frames, aspect)
            
            if result.status == "oom":
                # Record the limit
                self.update_limit(f"max_frames@{pixels}px", frames, frames)
                print(f"  🛑 OOM at {frames} frames - stopping frame tests")
                break
    
    def run_full_suite(self, quick: bool = False):
        """Run complete test suite"""
        print("\n" + "="*60)
        print(f"🚀 WAN 2.2 TEST SUITE - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"   Model: {self.config.model_name}")
        print(f"   GPU Allocation: {self.config.gpu_allocation}")
        print(f"   Mode: {'Quick' if quick else 'Full'}")
        print("="*60)
        
        self.run_resolution_tests(quick)
        self.run_frame_count_tests(quick)
        
        self.save_results()
        self.print_summary()
    
    def get_summary(self) -> dict:
        """Generate summary statistics"""
        if not self.results:
            return {}
        
        successful = [r for r in self.results if r.status == "success"]
        failed = [r for r in self.results if r.status != "success"]
        
        # Find limits
        max_resolution = max(successful, key=lambda r: r.width * r.height, default=None)
        max_frames = max(successful, key=lambda r: r.frames, default=None)
        
        return {
            "total_tests": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "max_resolution": f"{max_resolution.width}x{max_resolution.height}" if max_resolution else None,
            "max_frames": max_frames.frames if max_frames else None,
            "avg_generation_time": sum(r.generation_time_sec for r in successful) / len(successful) if successful else 0,
            "working_resolutions": [
                {"width": r.width, "height": r.height, "aspect": r.aspect_ratio, "time": r.generation_time_sec}
                for r in sorted(successful, key=lambda r: r.width * r.height, reverse=True)
            ]
        }
    
    def print_summary(self):
        """Print formatted summary"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        print(f"   Total Tests: {summary.get('total_tests', 0)}")
        print(f"   Successful: {summary.get('successful', 0)}")
        print(f"   Failed: {summary.get('failed', 0)}")
        print(f"   Max Resolution: {summary.get('max_resolution', 'N/A')}")
        print(f"   Max Frames: {summary.get('max_frames', 'N/A')}")
        print(f"   Avg Generation Time: {summary.get('avg_generation_time', 0):.1f}s")
        
        print("\n📋 Working Resolutions:")
        for r in summary.get("working_resolutions", [])[:10]:
            print(f"   {r['width']}x{r['height']} ({r['aspect']}) - {r['time']:.1f}s")
        
        print(f"\n💾 Full results saved to: {RESULTS_PATH}")
    
    def generate_markdown_report(self) -> str:
        """Generate markdown documentation"""
        summary = self.get_summary()
        
        md = f"""# WAN 2.2 Video Generation Limits

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Hardware Configuration
- GPU 0: RTX 3060 (12GB)
- GPU 1: RTX 5060 Ti (16GB)
- Allocation: `{self.config.gpu_allocation}`

## Summary
- Max Resolution: **{summary.get('max_resolution', 'N/A')}**
- Max Frames: **{summary.get('max_frames', 'N/A')}**
- Total Tests: {summary.get('total_tests', 0)}
- Success Rate: {summary.get('successful', 0)}/{summary.get('total_tests', 0)}

## Working Resolutions

| Resolution | Aspect | Frames | Time (s) | Status |
|------------|--------|--------|----------|--------|
"""
        for r in self.results:
            status_icon = "✅" if r.status == "success" else "❌" if r.status == "oom" else "⚠️"
            md += f"| {r.width}x{r.height} | {r.aspect_ratio} | {r.frames} | {r.generation_time_sec:.1f} | {status_icon} |\n"
        
        md += """
## Notes
- Frame counts use WAN's 4n+1 formula (33, 49, 65, 81, 101, 121, 145, 169, 193, 241)
- OOM = Out of Memory
- Times include VAE encode/decode
"""
        return md


def main():
    parser = argparse.ArgumentParser(description="WAN 2.2 Test Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick test subset")
    parser.add_argument("--resume", action="store_true", help="Resume from last position")
    parser.add_argument("--report", action="store_true", help="Show results only")
    parser.add_argument("--markdown", action="store_true", help="Generate markdown report")
    args = parser.parse_args()
    
    suite = WanTestSuite()
    
    if args.report:
        suite.print_summary()
    elif args.markdown:
        report = suite.generate_markdown_report()
        report_path = Path("/home/flip/oelala/docs/WAN_LIMITS.md")
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(report)
        print(f"📄 Report saved to {report_path}")
    else:
        suite.run_full_suite(quick=args.quick)


if __name__ == "__main__":
    main()
