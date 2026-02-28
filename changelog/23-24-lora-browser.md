### Added
- LoRA browser API at `/api/loras` with search, filter (category/tag/base_model/noise), sort, and pagination
- `lora_scanner.py` — LoRA cache with TTL, scans both SSD and ComfyUI directories, extracts safetensors metadata
- Visual LoRA browser component (`LoraBrowser.jsx`) with grid layout, search/filter panel, tag pills
- LoRA Browser tool in dashboard sidebar under Advanced
- Fixed existing `/loras` endpoint to scan `/mnt/ssd/loras/` (was only scanning empty ComfyUI dir)
