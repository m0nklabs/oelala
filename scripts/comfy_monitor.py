#!/usr/bin/env python3
"""
ComfyUI Workflow Runner with Resource Monitoring
Monitors GPU VRAM, CPU, RAM during workflow execution
"""

import json
import time
import subprocess
import threading
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

class ResourceMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.running = False
        self.data = []
        self.thread = None
        
    def _get_gpu_stats(self):
        """Get GPU memory usage via nvidia-smi"""
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
        except Exception as e:
            return []
    
    def _get_cpu_ram(self):
        """Get CPU and RAM usage"""
        try:
            # RAM
            with open('/proc/meminfo') as f:
                mem = {}
                for line in f:
                    parts = line.split()
                    if parts[0] in ['MemTotal:', 'MemAvailable:']:
                        mem[parts[0][:-1]] = int(parts[1]) // 1024  # KB to MB
            
            ram_used = mem.get('MemTotal', 0) - mem.get('MemAvailable', 0)
            ram_total = mem.get('MemTotal', 0)
            
            # CPU load
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
            time.sleep(self.interval)
    
    def start(self):
        self.running = True
        self.data = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()
        print("📊 Resource monitoring started...")
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        print("📊 Resource monitoring stopped.")
    
    def get_summary(self):
        if not self.data:
            return "No data collected"
        
        # Calculate peaks
        gpu0_peak = max((d['gpus'][0]['mem_used_mb'] for d in self.data if d['gpus']), default=0)
        gpu1_peak = max((d['gpus'][1]['mem_used_mb'] for d in self.data if len(d['gpus']) > 1), default=0)
        ram_peak = max(d['ram_used_mb'] for d in self.data)
        
        gpu0_total = self.data[0]['gpus'][0]['mem_total_mb'] if self.data[0]['gpus'] else 0
        gpu1_total = self.data[0]['gpus'][1]['mem_total_mb'] if len(self.data[0]['gpus']) > 1 else 0
        ram_total = self.data[0]['ram_total_mb']
        
        return f"""
═══════════════════════════════════════════════════
📊 RESOURCE USAGE SUMMARY
═══════════════════════════════════════════════════
Duration: {self.data[-1]['time'] - self.data[0]['time']:.1f} seconds
Samples: {len(self.data)}

🎮 GPU 0 (RTX 5060 Ti):
   Peak VRAM: {gpu0_peak:,} MB / {gpu0_total:,} MB ({100*gpu0_peak/gpu0_total:.1f}%)
   
🎮 GPU 1 (RTX 3060):
   Peak VRAM: {gpu1_peak:,} MB / {gpu1_total:,} MB ({100*gpu1_peak/gpu1_total:.1f}% if gpu1_total else 0)

💾 System RAM:
   Peak: {ram_peak:,} MB / {ram_total:,} MB ({100*ram_peak/ram_total:.1f}%)
═══════════════════════════════════════════════════
"""

    def export_csv(self, filename):
        with open(filename, 'w') as f:
            f.write("timestamp,gpu0_mem_mb,gpu0_util,gpu1_mem_mb,gpu1_util,ram_mb,cpu_load\n")
            for d in self.data:
                gpu0 = d['gpus'][0] if d['gpus'] else {'mem_used_mb': 0, 'util_pct': 0}
                gpu1 = d['gpus'][1] if len(d['gpus']) > 1 else {'mem_used_mb': 0, 'util_pct': 0}
                f.write(f"{d['time']},{gpu0['mem_used_mb']},{gpu0['util_pct']},{gpu1['mem_used_mb']},{gpu1['util_pct']},{d['ram_used_mb']},{d['cpu_load']}\n")
        print(f"📁 Data exported to {filename}")


def get_queue_status():
    """Check ComfyUI queue status"""
    try:
        req = urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=5)
        return json.loads(req.read().decode())
    except:
        return None

def wait_for_completion(timeout=600):
    """Wait until queue is empty"""
    start = time.time()
    print("⏳ Waiting for workflow to complete...")
    
    while time.time() - start < timeout:
        status = get_queue_status()
        if status:
            running = len(status.get('queue_running', []))
            pending = len(status.get('queue_pending', []))
            if running == 0 and pending == 0:
                print("✅ Workflow completed!")
                return True
            print(f"   Running: {running}, Pending: {pending}", end='\r')
        time.sleep(1)
    
    print("⚠️ Timeout waiting for completion")
    return False


def run_with_monitoring():
    """Monitor current ComfyUI execution"""
    print("\n🚀 ComfyUI Resource Monitor")
    print("=" * 50)
    
    # Check if ComfyUI is running
    status = get_queue_status()
    if not status:
        print("❌ ComfyUI not reachable at http://127.0.0.1:8188")
        print("   Start ComfyUI first, then queue a workflow in the UI")
        return
    
    running = len(status.get('queue_running', []))
    pending = len(status.get('queue_pending', []))
    
    print(f"📡 ComfyUI connected - Queue: {running} running, {pending} pending")
    
    if running == 0 and pending == 0:
        print("\n⏸️  No workflow running. Queue a workflow in ComfyUI UI,")
        print("   then run this script again, or press Enter after queueing...")
        input()
    
    # Start monitoring
    monitor = ResourceMonitor(interval=0.5)
    monitor.start()
    
    try:
        wait_for_completion()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    monitor.stop()
    
    # Show summary
    print(monitor.get_summary())
    
    # Export data
    csv_file = f"/home/flip/oelala/comfy_resources_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    monitor.export_csv(csv_file)
    
    return monitor.data


if __name__ == "__main__":
    run_with_monitoring()
