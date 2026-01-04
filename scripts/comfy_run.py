#!/usr/bin/env python3
"""
ComfyUI Workflow Runner with Resource Monitoring
Runs workflow via API and monitors GPU/CPU resources
"""

import json
import time
import subprocess
import threading
import urllib.request
import urllib.error
import sys
from datetime import datetime
from pathlib import Path


class ResourceMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.running = False
        self.data = []
        self.thread = None
        
    def _get_gpu_stats(self):
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    gpus.append({
                        'index': int(parts[0]),
                        'mem_used_mb': int(parts[1]),
                        'mem_total_mb': int(parts[2]),
                        'util_pct': int(parts[3])
                    })
            return gpus
        except:
            return []
    
    def _get_cpu_ram(self):
        try:
            with open('/proc/meminfo') as f:
                mem = {}
                for line in f:
                    parts = line.split()
                    if parts[0] in ['MemTotal:', 'MemAvailable:']:
                        mem[parts[0][:-1]] = int(parts[1]) // 1024
            
            ram_used = mem.get('MemTotal', 0) - mem.get('MemAvailable', 0)
            ram_total = mem.get('MemTotal', 0)
            
            load = subprocess.run(['cat', '/proc/loadavg'], capture_output=True, text=True)
            cpu_load = float(load.stdout.split()[0])
            
            return {'ram_used_mb': ram_used, 'ram_total_mb': ram_total, 'cpu_load': cpu_load}
        except:
            return {'ram_used_mb': 0, 'ram_total_mb': 0, 'cpu_load': 0}
    
    def _monitor_loop(self):
        while self.running:
            timestamp = time.time()
            gpus = self._get_gpu_stats()
            cpu_ram = self._get_cpu_ram()
            
            self.data.append({
                'time': timestamp,
                'gpus': gpus,
                **cpu_ram
            })
            
            # Print live stats
            if gpus:
                gpu0 = gpus[0]['mem_used_mb'] if gpus else 0
                gpu1 = gpus[1]['mem_used_mb'] if len(gpus) > 1 else 0
                print(f"\r  GPU0: {gpu0:,}MB | GPU1: {gpu1:,}MB | RAM: {cpu_ram['ram_used_mb']:,}MB", end='', flush=True)
            
            time.sleep(self.interval)
    
    def start(self):
        self.running = True
        self.data = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        print()  # Newline after live stats
    
    def get_summary(self):
        if not self.data:
            return "No data collected"
        
        gpu0_peak = max((d['gpus'][0]['mem_used_mb'] for d in self.data if d['gpus']), default=0)
        gpu1_peak = max((d['gpus'][1]['mem_used_mb'] for d in self.data if len(d['gpus']) > 1), default=0)
        ram_peak = max(d['ram_used_mb'] for d in self.data)
        
        gpu0_total = self.data[0]['gpus'][0]['mem_total_mb'] if self.data[0]['gpus'] else 16384
        gpu1_total = self.data[0]['gpus'][1]['mem_total_mb'] if len(self.data[0]['gpus']) > 1 else 12288
        ram_total = self.data[0]['ram_total_mb'] or 1
        
        duration = self.data[-1]['time'] - self.data[0]['time']
        
        return f"""
════════════════════════════════════════════════════
📊 RESOURCE USAGE SUMMARY
════════════════════════════════════════════════════
Duration: {duration:.1f} seconds
Samples: {len(self.data)}

🎮 GPU 0 (cuda:0):
   Peak VRAM: {gpu0_peak:,} MB / {gpu0_total:,} MB ({100*gpu0_peak/gpu0_total:.1f}%)
   
🎮 GPU 1 (cuda:1):
   Peak VRAM: {gpu1_peak:,} MB / {gpu1_total:,} MB ({100*gpu1_peak/gpu1_total:.1f}%)

💾 System RAM:
   Peak: {ram_peak:,} MB / {ram_total:,} MB ({100*ram_peak/ram_total:.1f}%)
════════════════════════════════════════════════════
"""

    def export_csv(self, filename):
        with open(filename, 'w') as f:
            f.write("timestamp,gpu0_mem_mb,gpu0_util,gpu1_mem_mb,gpu1_util,ram_mb,cpu_load\n")
            for d in self.data:
                gpu0 = d['gpus'][0] if d['gpus'] else {'mem_used_mb': 0, 'util_pct': 0}
                gpu1 = d['gpus'][1] if len(d['gpus']) > 1 else {'mem_used_mb': 0, 'util_pct': 0}
                f.write(f"{d['time']},{gpu0['mem_used_mb']},{gpu0['util_pct']},{gpu1['mem_used_mb']},{gpu1['util_pct']},{d['ram_used_mb']},{d['cpu_load']}\n")
        return filename


def queue_prompt(api_workflow_path):
    """Queue a workflow via ComfyUI API"""
    with open(api_workflow_path) as f:
        prompt = json.load(f)
    
    data = json.dumps({"prompt": prompt}).encode('utf-8')
    req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=data)
    req.add_header("Content-Type", "application/json")
    
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read().decode())
        return result.get('prompt_id')
    except urllib.error.HTTPError as e:
        error = e.read().decode()
        print(f"❌ API Error: {error}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def get_queue_status():
    try:
        req = urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=5)
        return json.loads(req.read().decode())
    except:
        return None


def wait_for_completion(prompt_id, timeout=600):
    """Wait until specific prompt completes"""
    start = time.time()
    
    while time.time() - start < timeout:
        status = get_queue_status()
        if status:
            running = status.get('queue_running', [])
            pending = status.get('queue_pending', [])
            
            # Check if our prompt is still running/pending
            our_running = any(p[1] == prompt_id for p in running)
            our_pending = any(p[1] == prompt_id for p in pending)
            
            if not our_running and not our_pending:
                return True
        
        time.sleep(0.5)
    
    return False


def run_workflow(api_workflow_path):
    """Run workflow with monitoring"""
    print("\n🚀 ComfyUI Workflow Runner")
    print("═" * 50)
    print(f"📄 Workflow: {Path(api_workflow_path).name}")
    
    # Queue the workflow
    print("\n📤 Queueing workflow...")
    prompt_id = queue_prompt(api_workflow_path)
    
    if not prompt_id:
        print("❌ Failed to queue workflow")
        return None
    
    print(f"✅ Queued with ID: {prompt_id}")
    
    # Start monitoring
    monitor = ResourceMonitor(interval=0.5)
    print("\n📊 Starting resource monitoring...")
    monitor.start()
    
    # Wait for completion
    print("⏳ Running workflow...")
    try:
        success = wait_for_completion(prompt_id)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted!")
        success = False
    
    monitor.stop()
    
    if success:
        print("✅ Workflow completed!")
    else:
        print("⚠️ Workflow may have failed or timed out")
    
    # Show summary
    print(monitor.get_summary())
    
    # Export data
    csv_file = f"/home/flip/oelala/comfy_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    monitor.export_csv(csv_file)
    print(f"📁 Data exported: {csv_file}")
    
    return monitor.data


def main():
    if len(sys.argv) < 2:
        print("Usage: python comfy_run.py <workflow_api.json>")
        print("\nRuns ComfyUI workflow and monitors resources.")
        print("Use comfy_convert.py first to convert workflow to API format.")
        sys.exit(1)
    
    run_workflow(sys.argv[1])


if __name__ == "__main__":
    main()
