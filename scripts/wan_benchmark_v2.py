#!/usr/bin/env python3
"""
WAN 2.2 Hardware Benchmark Suite v2
====================================
Finds hardware limits for video generation: resolution, frames, aspect ratios.
Outputs JSON for frontend configuration and generates visualization charts.

Usage:
    python wan_benchmark_v2.py                  # Full benchmark
    python wan_benchmark_v2.py --quick          # Quick test
    python wan_benchmark_v2.py --report         # Generate report only
    python wan_benchmark_v2.py --charts         # Generate charts only
"""

import json
import time
import random
import signal
import argparse
import subprocess
import urllib.request
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

COMFY_API = "http://127.0.0.1:8188"
WORKFLOW_PATH = Path("/home/flip/oelala/ComfyUI/user/default/workflows/WAN2.2-I2V-DisTorch2-NEW_api.json")
RESULTS_DIR = Path("/home/flip/oelala/benchmark_results")
RESULTS_DIR.mkdir(exist_ok=True)

# GPU Configuration - maximize GPU usage
GPU_ALLOCATION = "cuda:0,12gb;cuda:1,16gb;cpu,*"
GPU_SPECS = {
    "cuda:0": {"name": "RTX 3060", "vram_mb": 12288},
    "cuda:1": {"name": "RTX 5060 Ti", "vram_mb": 16311},
}
TOTAL_GPU_VRAM_MB = sum(g["vram_mb"] for g in GPU_SPECS.values())

# Test Matrix - ordered small to large
RESOLUTIONS = [
    # 16:9 landscape
    (720, 400), (848, 480), (1024, 576), (1280, 720), (1920, 1080),
    # 9:16 portrait  
    (400, 720), (480, 848), (576, 1024), (720, 1280),
    # 4:3
    (640, 480), (800, 600), (960, 720),
    # 1:1 square
    (480, 480), (576, 576), (720, 720), (1024, 1024),
    # 21:9 ultrawide
    (840, 360), (1024, 432), (1280, 544),
]

# Frame counts (WAN uses 4n+1 formula) - ordered low to high
FRAME_COUNTS = [17, 33, 49, 65, 81, 101, 121, 145, 169, 193, 241]

# FPS for output encoding (not generation speed)
OUTPUT_FPS = [24, 30, 60]

# Time constraints for frontend (seconds)
TIME_CONSTRAINTS = {
    "instant": 30,      # < 30s
    "fast": 120,        # < 2 min
    "normal": 300,      # < 5 min
    "slow": 600,        # < 10 min
    "very_slow": 1800,  # < 30 min
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    width: int
    height: int
    frames: int
    pixels: int = field(init=False)
    aspect_ratio: str = field(init=False)
    
    status: str = "pending"  # pending, running, success, oom, error, skipped
    generation_time_sec: float = 0.0
    gpu0_peak_mb: int = 0
    gpu1_peak_mb: int = 0
    total_vram_mb: int = field(init=False)
    
    # Computed metrics
    pixels_per_sec: float = 0.0
    frames_per_sec: float = 0.0
    time_category: str = ""
    
    error_message: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        self.pixels = self.width * self.height
        self.aspect_ratio = self._calc_aspect_ratio()
        self.total_vram_mb = self.gpu0_peak_mb + self.gpu1_peak_mb
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def _calc_aspect_ratio(self) -> str:
        """Calculate aspect ratio string"""
        from math import gcd
        g = gcd(self.width, self.height)
        w, h = self.width // g, self.height // g
        # Normalize common ratios
        ratios = {(16, 9): "16:9", (9, 16): "9:16", (4, 3): "4:3", (3, 4): "3:4",
                  (1, 1): "1:1", (21, 9): "21:9", (9, 21): "9:21"}
        return ratios.get((w, h), f"{w}:{h}")
    
    def compute_metrics(self):
        """Compute performance metrics after completion"""
        if self.status == "success" and self.generation_time_sec > 0:
            total_pixels = self.pixels * self.frames
            self.pixels_per_sec = total_pixels / self.generation_time_sec
            self.frames_per_sec = self.frames / self.generation_time_sec
            
            # Categorize by time
            for cat, max_time in TIME_CONSTRAINTS.items():
                if self.generation_time_sec <= max_time:
                    self.time_category = cat
                    break
            else:
                self.time_category = "extreme"


@dataclass 
class HardwareLimits:
    """Discovered hardware limits"""
    max_pixels_at_101f: int = 0
    max_frames_at_720x400: int = 0
    max_resolution: Tuple[int, int] = (0, 0)
    max_frames: int = 0
    
    # Per aspect ratio limits
    limits_by_aspect: Dict[str, Dict] = field(default_factory=dict)
    
    # OOM thresholds
    oom_pixel_threshold: int = 0
    oom_frame_threshold: int = 0
    
    def update_from_result(self, result: BenchmarkResult):
        """Update limits based on a test result"""
        if result.status == "success":
            if result.frames == 101 and result.pixels > self.max_pixels_at_101f:
                self.max_pixels_at_101f = result.pixels
                self.max_resolution = (result.width, result.height)
            
            if result.width == 720 and result.height == 400:
                if result.frames > self.max_frames_at_720x400:
                    self.max_frames_at_720x400 = result.frames
            
            if result.pixels > self.max_pixels_at_101f:
                self.max_pixels_at_101f = result.pixels
            if result.frames > self.max_frames:
                self.max_frames = result.frames
            
            # Per aspect ratio
            ar = result.aspect_ratio
            if ar not in self.limits_by_aspect:
                self.limits_by_aspect[ar] = {"max_res": (0, 0), "max_frames": 0}
            if result.pixels > self.limits_by_aspect[ar]["max_res"][0] * self.limits_by_aspect[ar]["max_res"][1]:
                self.limits_by_aspect[ar]["max_res"] = (result.width, result.height)
            if result.frames > self.limits_by_aspect[ar]["max_frames"]:
                self.limits_by_aspect[ar]["max_frames"] = result.frames
                
        elif result.status == "oom":
            if self.oom_pixel_threshold == 0 or result.pixels < self.oom_pixel_threshold:
                self.oom_pixel_threshold = result.pixels
            if self.oom_frame_threshold == 0 or result.frames < self.oom_frame_threshold:
                self.oom_frame_threshold = result.frames


# ============================================================================
# BENCHMARK ENGINE
# ============================================================================

class WanBenchmark:
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.limits = HardwareLimits()
        self.interrupted = False
        self.results_file = RESULTS_DIR / "benchmark_results.json"
        self.frontend_file = RESULTS_DIR / "frontend_config.json"
        
        # Setup interrupt handler
        signal.signal(signal.SIGINT, self._handle_interrupt)
        
        self._load_results()
    
    def _handle_interrupt(self, signum, frame):
        """Graceful shutdown on Ctrl+C"""
        print("\n\n⚠️ Interrupt received - saving results...")
        self.interrupted = True
        self._save_results()
        print("💾 Results saved. Exiting.")
    
    def _load_results(self):
        """Load existing results"""
        if self.results_file.exists():
            try:
                with open(self.results_file) as f:
                    data = json.load(f)
                self.results = [BenchmarkResult(**r) for r in data.get("results", [])]
                if "limits" in data:
                    self.limits = HardwareLimits(**data["limits"])
                print(f"📂 Loaded {len(self.results)} existing results")
            except Exception as e:
                print(f"⚠️ Could not load results: {e}")
    
    def _save_results(self):
        """Save results to JSON"""
        data = {
            "hardware": GPU_SPECS,
            "gpu_allocation": GPU_ALLOCATION,
            "results": [asdict(r) for r in self.results],
            "limits": asdict(self.limits),
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.results_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved {len(self.results)} results")
        
        # Also generate frontend config
        self._generate_frontend_config()
    
    def _generate_frontend_config(self):
        """Generate config for oelala frontend"""
        successful = [r for r in self.results if r.status == "success"]
        
        # Group by aspect ratio
        by_aspect = defaultdict(list)
        for r in successful:
            by_aspect[r.aspect_ratio].append(r)
        
        config = {
            "generated": datetime.now().isoformat(),
            "hardware": {
                "gpus": list(GPU_SPECS.values()),
                "total_vram_mb": TOTAL_GPU_VRAM_MB,
            },
            "limits": {
                "max_resolution": f"{self.limits.max_resolution[0]}x{self.limits.max_resolution[1]}",
                "max_pixels": self.limits.max_pixels_at_101f,
                "max_frames": self.limits.max_frames,
            },
            "aspect_ratios": {},
            "presets": [],
            "time_estimates": {},
        }
        
        # Per aspect ratio options
        for ar, results in by_aspect.items():
            resolutions = sorted(set((r.width, r.height) for r in results), 
                               key=lambda x: x[0]*x[1])
            max_frames = max(r.frames for r in results)
            
            config["aspect_ratios"][ar] = {
                "resolutions": [{"width": w, "height": h, "label": f"{w}x{h}"} 
                               for w, h in resolutions],
                "max_frames": max_frames,
                "frame_options": [f for f in FRAME_COUNTS if f <= max_frames],
            }
        
        # Time-based presets
        for time_cat, max_time in TIME_CONSTRAINTS.items():
            matching = [r for r in successful if r.generation_time_sec <= max_time]
            if matching:
                best = max(matching, key=lambda r: r.pixels * r.frames)
                config["presets"].append({
                    "name": time_cat,
                    "max_time_sec": max_time,
                    "recommended": {
                        "width": best.width,
                        "height": best.height,
                        "frames": best.frames,
                        "est_time_sec": best.generation_time_sec,
                    }
                })
        
        # Time estimation formula (linear regression style)
        if len(successful) >= 3:
            # Simple: time ≈ pixels * frames * factor
            factors = [r.generation_time_sec / (r.pixels * r.frames) 
                      for r in successful if r.pixels * r.frames > 0]
            avg_factor = sum(factors) / len(factors)
            config["time_estimates"] = {
                "formula": "time_sec = pixels * frames * factor",
                "factor": avg_factor,
                "unit": "seconds per pixel-frame",
            }
        
        with open(self.frontend_file, "w") as f:
            json.dump(config, f, indent=2)
        print(f"🎨 Frontend config saved to {self.frontend_file}")
    
    def _get_gpu_memory(self) -> Tuple[int, int]:
        """Get current GPU memory usage"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            mems = [int(x.strip()) for x in result.stdout.strip().split("\n")]
            return tuple(mems) if len(mems) >= 2 else (0, 0)
        except:
            return (0, 0)
    
    def _already_tested(self, width: int, height: int, frames: int) -> Optional[BenchmarkResult]:
        """Check if this config was already tested"""
        for r in self.results:
            if r.width == width and r.height == height and r.frames == frames:
                if r.status in ["success", "oom"]:
                    return r
        return None
    
    def _should_skip(self, width: int, height: int, frames: int) -> Tuple[bool, str]:
        """Determine if test should be skipped based on known limits"""
        pixels = width * height
        
        # Skip if we already know OOM threshold
        if self.limits.oom_pixel_threshold > 0:
            if pixels >= self.limits.oom_pixel_threshold and frames >= 101:
                return True, f"Beyond OOM threshold ({self.limits.oom_pixel_threshold} pixels)"
        
        if self.limits.oom_frame_threshold > 0:
            if frames >= self.limits.oom_frame_threshold and pixels >= 720*400:
                return True, f"Beyond OOM threshold ({self.limits.oom_frame_threshold} frames)"
        
        return False, ""
    
    def _update_workflow(self, width: int, height: int, frames: int) -> dict:
        """Update workflow with test parameters"""
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
    
    def _queue_prompt(self, workflow: dict) -> Optional[str]:
        """Queue workflow and return prompt_id"""
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
    
    def _wait_for_completion(self, prompt_id: str, timeout: int = 1800) -> Tuple[str, float, int, int, str]:
        """Wait for prompt completion, return (status, time, gpu0, gpu1, error)"""
        start_time = time.time()
        peak_gpu0 = 0
        peak_gpu1 = 0
        last_progress = 0
        
        while not self.interrupted:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                return ("timeout", elapsed, peak_gpu0, peak_gpu1, "Generation timed out")
            
            # Track peak VRAM
            gpu0, gpu1 = self._get_gpu_memory()
            peak_gpu0 = max(peak_gpu0, gpu0)
            peak_gpu1 = max(peak_gpu1, gpu1)
            
            # Progress indicator
            if int(elapsed) >= last_progress + 30:
                last_progress = int(elapsed)
                print(f"  ⏳ {int(elapsed)}s - VRAM: {gpu0}+{gpu1}={gpu0+gpu1}MB")
            
            try:
                with urllib.request.urlopen(f"{COMFY_API}/queue", timeout=5) as resp:
                    queue = json.loads(resp.read())
                
                running = queue.get("queue_running", [])
                pending = queue.get("queue_pending", [])
                
                is_active = any(p[1] == prompt_id for p in running + pending)
                
                if not is_active:
                    # Check history
                    with urllib.request.urlopen(f"{COMFY_API}/history/{prompt_id}", timeout=5) as resp:
                        history = json.loads(resp.read())
                    
                    if prompt_id in history:
                        status = history[prompt_id].get("status", {})
                        if status.get("completed", False):
                            return ("success", elapsed, peak_gpu0, peak_gpu1, None)
                        elif status.get("status_str") == "error":
                            msgs = str(status.get("messages", []))
                            if "out of memory" in msgs.lower() or "oom" in msgs.lower():
                                return ("oom", elapsed, peak_gpu0, peak_gpu1, msgs)
                            return ("error", elapsed, peak_gpu0, peak_gpu1, msgs)
                    else:
                        # Prompt finished but not in history yet
                        time.sleep(1)
                        continue
                        
            except Exception:
                pass
            
            time.sleep(2)
        
        return ("interrupted", time.time() - start_time, peak_gpu0, peak_gpu1, "User interrupted")
    
    def run_single_test(self, width: int, height: int, frames: int) -> BenchmarkResult:
        """Run a single benchmark test"""
        # Check if already tested
        existing = self._already_tested(width, height, frames)
        if existing:
            print(f"  ⏭️ Already tested: {existing.status}")
            return existing
        
        # Check if should skip
        should_skip, reason = self._should_skip(width, height, frames)
        if should_skip:
            print(f"  ⏭️ Skipping: {reason}")
            result = BenchmarkResult(width=width, height=height, frames=frames)
            result.status = "skipped"
            result.error_message = reason
            return result
        
        # Create result object
        result = BenchmarkResult(width=width, height=height, frames=frames)
        result.status = "running"
        
        # Update and queue workflow
        workflow = self._update_workflow(width, height, frames)
        prompt_id = self._queue_prompt(workflow)
        
        if not prompt_id:
            result.status = "error"
            result.error_message = "Failed to queue"
            self.results.append(result)
            return result
        
        # Wait for completion
        status, elapsed, peak0, peak1, error = self._wait_for_completion(prompt_id)
        
        result.status = status
        result.generation_time_sec = elapsed
        result.gpu0_peak_mb = peak0
        result.gpu1_peak_mb = peak1
        result.total_vram_mb = peak0 + peak1
        result.error_message = error
        result.compute_metrics()
        
        # Update limits
        self.limits.update_from_result(result)
        
        # Store and save
        self.results.append(result)
        self._save_results()
        
        # Report
        if status == "success":
            print(f"  ✅ SUCCESS {elapsed:.1f}s - {result.pixels_per_sec/1e6:.2f}Mpix/s - VRAM: {peak0}+{peak1}MB")
        elif status == "oom":
            print(f"  💥 OOM after {elapsed:.1f}s - VRAM: {peak0}+{peak1}MB")
        else:
            print(f"  ❌ {status}: {error}")
        
        return result
    
    def run_resolution_sweep(self, frames: int = 101, quick: bool = False):
        """Test all resolutions at fixed frame count"""
        print(f"\n{'='*60}")
        print(f"📐 RESOLUTION SWEEP @ {frames} frames")
        print(f"{'='*60}")
        
        # Sort by pixel count (small to large)
        sorted_res = sorted(RESOLUTIONS, key=lambda r: r[0]*r[1])
        if quick:
            sorted_res = sorted_res[:6]
        
        for width, height in sorted_res:
            if self.interrupted:
                break
            
            pixels = width * height
            ar = BenchmarkResult(width=width, height=height, frames=frames).aspect_ratio
            print(f"\n🎬 {width}x{height} ({ar}, {pixels} pixels)...")
            
            result = self.run_single_test(width, height, frames)
            
            # Stop if OOM (no point testing larger)
            if result.status == "oom":
                print(f"\n🛑 OOM limit reached at {pixels} pixels - stopping resolution sweep")
                break
    
    def run_frame_sweep(self, width: int = 720, height: int = 400, quick: bool = False):
        """Test all frame counts at fixed resolution"""
        print(f"\n{'='*60}")
        print(f"🎞️ FRAME SWEEP @ {width}x{height}")
        print(f"{'='*60}")
        
        test_frames = FRAME_COUNTS[:5] if quick else FRAME_COUNTS
        
        for frames in test_frames:
            if self.interrupted:
                break
            
            print(f"\n🎬 {frames} frames ({frames/24:.1f}s @ 24fps)...")
            
            result = self.run_single_test(width, height, frames)
            
            if result.status == "oom":
                print(f"\n🛑 OOM limit reached at {frames} frames - stopping frame sweep")
                break
    
    def run_full_benchmark(self, quick: bool = False):
        """Run complete benchmark suite"""
        print(f"\n{'='*60}")
        print(f"🚀 WAN 2.2 HARDWARE BENCHMARK v2")
        print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"   GPUs: {', '.join(g['name'] for g in GPU_SPECS.values())}")
        print(f"   Total VRAM: {TOTAL_GPU_VRAM_MB}MB")
        print(f"   Mode: {'Quick' if quick else 'Full'}")
        print(f"{'='*60}")
        
        # Phase 1: Resolution sweep at 101 frames
        self.run_resolution_sweep(frames=101, quick=quick)
        
        if not self.interrupted:
            # Phase 2: Frame sweep at safe resolution
            self.run_frame_sweep(width=720, height=400, quick=quick)
        
        if not self.interrupted:
            # Phase 3: Test max resolution at lower frame counts
            if self.limits.max_resolution != (0, 0):
                w, h = self.limits.max_resolution
                print(f"\n{'='*60}")
                print(f"🎯 TESTING MAX RES {w}x{h} AT VARIOUS FRAME COUNTS")
                print(f"{'='*60}")
                
                for frames in [33, 49, 65, 81]:
                    if self.interrupted:
                        break
                    print(f"\n🎬 {w}x{h} @ {frames} frames...")
                    self.run_single_test(w, h, frames)
        
        # Final report
        self._save_results()
        self.print_summary()
    
    def print_summary(self):
        """Print benchmark summary"""
        successful = [r for r in self.results if r.status == "success"]
        failed = [r for r in self.results if r.status in ["oom", "error"]]
        
        print(f"\n{'='*60}")
        print(f"📊 BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"   Total tests: {len(self.results)}")
        print(f"   Successful: {len(successful)}")
        print(f"   Failed (OOM/Error): {len(failed)}")
        
        if self.limits.max_resolution != (0, 0):
            w, h = self.limits.max_resolution
            print(f"\n   🏆 Max Resolution: {w}x{h} ({w*h} pixels)")
        print(f"   🏆 Max Frames: {self.limits.max_frames}")
        
        print(f"\n   📁 Results: {self.results_file}")
        print(f"   🎨 Frontend config: {self.frontend_file}")
        
        # Top performers by category
        if successful:
            print(f"\n🏅 TOP PERFORMERS:")
            
            # Fastest
            fastest = min(successful, key=lambda r: r.generation_time_sec)
            print(f"   ⚡ Fastest: {fastest.width}x{fastest.height}@{fastest.frames}f = {fastest.generation_time_sec:.1f}s")
            
            # Highest quality (most pixels*frames)
            best = max(successful, key=lambda r: r.pixels * r.frames)
            print(f"   👑 Best Quality: {best.width}x{best.height}@{best.frames}f = {best.generation_time_sec:.1f}s")
            
            # Best efficiency (pixels*frames per second)
            efficient = max(successful, key=lambda r: r.pixels_per_sec)
            print(f"   📈 Most Efficient: {efficient.width}x{efficient.height} = {efficient.pixels_per_sec/1e6:.2f}Mpix/s")
    
    def generate_charts(self):
        """Generate visualization charts"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("❌ matplotlib not installed. Run: pip install matplotlib")
            return
        
        successful = [r for r in self.results if r.status == "success"]
        if not successful:
            print("❌ No successful results to chart")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"WAN 2.2 Hardware Benchmark - {GPU_SPECS['cuda:0']['name']} + {GPU_SPECS['cuda:1']['name']}", 
                     fontsize=14, fontweight='bold')
        
        # Chart 1: Resolution vs Time
        ax1 = axes[0, 0]
        pixels = [r.pixels for r in successful]
        times = [r.generation_time_sec for r in successful]
        colors = ['green' if r.time_category in ['instant', 'fast'] else 'orange' if r.time_category == 'normal' else 'red' 
                  for r in successful]
        ax1.scatter(pixels, times, c=colors, alpha=0.7, s=50)
        ax1.set_xlabel('Pixels (Width × Height)')
        ax1.set_ylabel('Generation Time (seconds)')
        ax1.set_title('Resolution vs Generation Time')
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Frames vs Time at 720x400
        ax2 = axes[0, 1]
        frame_results = [r for r in successful if r.width == 720 and r.height == 400]
        if frame_results:
            frames = [r.frames for r in frame_results]
            times = [r.generation_time_sec for r in frame_results]
            ax2.bar(frames, times, color='steelblue', alpha=0.7)
            ax2.set_xlabel('Frame Count')
            ax2.set_ylabel('Generation Time (seconds)')
            ax2.set_title('Frame Count vs Time (720×400)')
            ax2.grid(True, alpha=0.3, axis='y')
        
        # Chart 3: VRAM Usage
        ax3 = axes[1, 0]
        vram = [r.total_vram_mb for r in successful]
        pixels = [r.pixels for r in successful]
        ax3.scatter(pixels, vram, alpha=0.7, s=50, c='purple')
        ax3.axhline(y=TOTAL_GPU_VRAM_MB, color='red', linestyle='--', label=f'Total VRAM ({TOTAL_GPU_VRAM_MB}MB)')
        ax3.set_xlabel('Pixels')
        ax3.set_ylabel('Peak VRAM (MB)')
        ax3.set_title('VRAM Usage by Resolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: Aspect Ratio Performance
        ax4 = axes[1, 1]
        by_aspect = defaultdict(list)
        for r in successful:
            by_aspect[r.aspect_ratio].append(r.pixels_per_sec / 1e6)
        
        aspects = list(by_aspect.keys())
        avg_perf = [sum(v)/len(v) for v in by_aspect.values()]
        bars = ax4.bar(aspects, avg_perf, color='teal', alpha=0.7)
        ax4.set_xlabel('Aspect Ratio')
        ax4.set_ylabel('Avg Performance (Mpix/sec)')
        ax4.set_title('Performance by Aspect Ratio')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        chart_path = RESULTS_DIR / "benchmark_charts.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        print(f"📊 Charts saved to {chart_path}")
        plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="WAN 2.2 Hardware Benchmark v2")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--report", action="store_true", help="Show report only")
    parser.add_argument("--charts", action="store_true", help="Generate charts only")
    parser.add_argument("--resolution", type=str, help="Test specific resolution (WxH)")
    parser.add_argument("--frames", type=int, help="Test specific frame count")
    args = parser.parse_args()
    
    bench = WanBenchmark()
    
    if args.report:
        bench.print_summary()
    elif args.charts:
        bench.generate_charts()
    elif args.resolution and args.frames:
        w, h = map(int, args.resolution.split('x'))
        print(f"🎬 Single test: {w}x{h}@{args.frames}f")
        bench.run_single_test(w, h, args.frames)
        bench.print_summary()
    else:
        bench.run_full_benchmark(quick=args.quick)
        bench.generate_charts()


if __name__ == "__main__":
    main()
