### Added

- **LoRA on-demand download for Cloud Max (RunPod)**: When a user enables a local LoRA for a cloud job, the backend generates HMAC-signed download URLs and passes them to the RunPod worker. The handler downloads missing LoRAs before executing the workflow, caching them on the Network Volume for subsequent jobs.
- `GET /loras/download/{filename}` endpoint for serving LoRA files to RunPod workers (HMAC-SHA256 token authentication)
- LoRA download support in all 5 cloud-capable endpoints: Cloud Max, Wan2.2, BlockSwap Q8, DisTorch2 Q8, Ultra Q8

### Changed

- `_submit_to_runpod()` now accepts optional `lora_downloads` parameter
- RunPod handler downloads LoRAs on demand with progress updates and Network Volume caching
- Frontend QueueIndicator: cloud jobs display ☁️ icon, indigo accent, cloud-aware cancel
- Frontend ProgressTracker: indigo progress bar, Cloud Max badge, RunPod progress messages
- Docker image `ghcr.io/m0nklabs/oelala-comfyui-worker:latest` updated (`sha256:6fc20867`)
