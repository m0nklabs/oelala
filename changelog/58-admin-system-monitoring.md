### Added

- **Admin System Monitoring Tab** (Issue #58)
  - GPU status display: VRAM usage, utilization, temperature per GPU
  - Service health monitoring: ComfyUI, oelala-storage status
  - ComfyUI queue viewer: running and pending jobs
  - System logs viewer: journalctl output for backend/comfyui/storage/frontend
  - Disk usage display for root, home, and SSD partitions
  - Auto-refresh every 10 seconds

- **Backend Admin Endpoints**
  - `GET /api/admin/system/gpu` - nvidia-smi GPU stats
  - `GET /api/admin/system/queue` - ComfyUI queue status
  - `GET /api/admin/system/health` - Service health + disk usage
  - `GET /api/admin/system/logs` - Recent systemd logs per service
