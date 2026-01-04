#!/usr/bin/env python3
"""
WAN 2.2 I2V Benchmark Script
Test video length and resolution limits
"""

import json
import sys
import time
import requests
import subprocess
from pathlib import Path

WORKFLOW_PATH = Path.home() / "oelala/ComfyUI/user/default/workflows/WAN2.2-I2V-DisTorch2-NEW_api.json"
COMFY_URL = "http://127.0.0.1:8188"

# Test configurations: (width, height, frames, description)
TESTS = [
    # Baseline
    (720, 400, 101, "Baseline 720x400 101f"),
    
    # Length tests (720x400)
    (720, 400, 145, "720x400 145f (~6s)"),
    (720, 400, 193, "720x400 193f (~8s)"),
    (720, 400, 241, "720x400 241f (~10s)"),
    
    # Resolution tests (101 frames)
    (832, 480, 101, "832x480 101f (16:9-ish)"),
    (960, 544, 101, "960x544 101f"),
    (1024, 576, 101, "1024x576 101f (16:9)"),
    (1280, 720, 101, "1280x720 101f (HD)"),
    
    # Aspect ratio tests
    (480, 832, 101, "480x832 101f (9:16 portrait)"),
    (576, 1024, 101, "576x1024 101f (9:16 portrait)"),
    (1024, 1024, 81, "1024x1024 81f (1:1 square)"),
    
    # Push limits
    (1280, 720, 145, "1280x720 145f (HD 6s)"),
    (1024, 576, 193, "1024x576 193f (8s)"),
]

def update_workflow(width: int, height: int, frames: int):
    """Update workflow with new settings."""
    with open(WORKFLOW_PATH) as f:
        data = json.load(f)
    
    for nid, node in data.items():
        if node.get('class_type') == 'WanImageToVideo':
            node['inputs']['width'] = width
            node['inputs']['height'] = height
            node['inputs']['length'] = frames
            
        # Fresh seed each run
        if node.get('class_type') == 'KSamplerAdvanced':
            import random
            node['inputs']['noise_seed'] = random.randint(0, 2**32-1)
    
    with open(WORKFLOW_PATH, 'w') as f:
        json.dump(data, f, indent=2)

def get_gpu_stats():
    """Get current GPU memory usage."""
    result = subprocess.run([
        'nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'
    ], capture_output=True, text=True)
    
    stats = []
    for line in result.stdout.strip().split('\n'):
        parts = [x.strip() for x in line.split(',')]
        stats.append({
            'index': int(parts[0]),
            'used_mb': int(parts[1]),
            'total_mb': int(parts[2]),
            'util': int(parts[3])
        })
    return stats

def queue_workflow():
    """Queue the workflow and return prompt_id."""
    with open(WORKFLOW_PATH) as f:
        workflow = json.load(f)
    
    response = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": workflow}
    )
    
    if response.status_code != 200:
        return None, response.text
    
    return response.json().get('prompt_id'), None

def wait_for_completion(prompt_id, timeout=600):
    """Wait for workflow to complete, tracking peak VRAM."""
    start = time.time()
    peak_vram = [0, 0]  # GPU0, GPU1
    samples = 0
    
    while time.time() - start < timeout:
        # Check queue status
        try:
            queue = requests.get(f"{COMFY_URL}/queue").json()
            running = queue.get('queue_running', [])
            pending = queue.get('queue_pending', [])
            
            is_running = any(p[1] == prompt_id for p in running)
            is_pending = any(p[1] == prompt_id for p in pending)
            
            if not is_running and not is_pending:
                # Check if it completed or errored
                history = requests.get(f"{COMFY_URL}/history/{prompt_id}").json()
                if prompt_id in history:
                    status = history[prompt_id].get('status', {})
                    if status.get('status_str') == 'error':
                        return False, peak_vram, time.time() - start, status.get('messages', [])
                    return True, peak_vram, time.time() - start, None
                    
        except Exception as e:
            pass
        
        # Track VRAM
        stats = get_gpu_stats()
        for gpu in stats:
            idx = gpu['index']
            if idx < 2:
                peak_vram[idx] = max(peak_vram[idx], gpu['used_mb'])
        
        samples += 1
        time.sleep(0.5)
    
    return False, peak_vram, timeout, "Timeout"

def run_test(width, height, frames, desc):
    """Run a single test configuration."""
    print(f"\n{'='*60}")
    print(f"🧪 TEST: {desc}")
    print(f"   Resolution: {width}x{height} ({width*height/1e6:.2f}MP)")
    print(f"   Frames: {frames} (~{frames/24:.1f}s @ 24fps)")
    print(f"{'='*60}")
    
    # Update workflow
    update_workflow(width, height, frames)
    
    # Queue
    prompt_id, error = queue_workflow()
    if error:
        print(f"❌ Failed to queue: {error}")
        return None
    
    print(f"📤 Queued: {prompt_id}")
    
    # Wait and track
    success, peak_vram, duration, error = wait_for_completion(prompt_id)
    
    result = {
        'desc': desc,
        'width': width,
        'height': height,
        'frames': frames,
        'megapixels': width * height / 1e6,
        'success': success,
        'duration': duration,
        'peak_gpu0_mb': peak_vram[0],
        'peak_gpu1_mb': peak_vram[1],
        'error': str(error) if error else None
    }
    
    if success:
        print(f"✅ SUCCESS in {duration:.1f}s")
        print(f"   GPU0 peak: {peak_vram[0]} MB")
        print(f"   GPU1 peak: {peak_vram[1]} MB")
    else:
        print(f"❌ FAILED after {duration:.1f}s")
        print(f"   Error: {error}")
        print(f"   GPU0 peak: {peak_vram[0]} MB")
        print(f"   GPU1 peak: {peak_vram[1]} MB")
    
    return result

def main():
    if len(sys.argv) > 1:
        # Single test mode: width height frames
        if len(sys.argv) >= 4:
            w, h, f = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
            desc = f"Manual {w}x{h} {f}f"
            run_test(w, h, f, desc)
        else:
            print("Usage: wan_benchmark.py [width height frames]")
            print("       wan_benchmark.py  (runs full benchmark)")
        return
    
    # Full benchmark
    print("\n" + "="*60)
    print("🚀 WAN 2.2 I2V BENCHMARK")
    print("="*60)
    print(f"GPUs: RTX 3060 (12GB) + RTX 5060 Ti (16GB)")
    print(f"Model: WAN 2.2 I2V 14B GGUF Q6_K")
    print(f"Allocation: cuda:0=8GB, cuda:1=12GB, cpu=rest")
    print("="*60)
    
    results = []
    for width, height, frames, desc in TESTS:
        result = run_test(width, height, frames, desc)
        if result:
            results.append(result)
        
        # Cool down between tests
        print("⏳ Cooldown 10s...")
        time.sleep(10)
    
    # Summary
    print("\n" + "="*60)
    print("📊 BENCHMARK SUMMARY")
    print("="*60)
    print(f"{'Config':<30} {'Status':<8} {'Time':<10} {'GPU0':<10} {'GPU1':<10}")
    print("-"*60)
    
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"{r['desc']:<30} {status:<8} {r['duration']:.1f}s      {r['peak_gpu0_mb']:<10} {r['peak_gpu1_mb']:<10}")
    
    # Save results
    with open('/home/flip/oelala/wan_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to wan_benchmark_results.json")

if __name__ == "__main__":
    main()
