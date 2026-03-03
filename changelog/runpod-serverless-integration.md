### Added

- **RunPod Serverless integration** — Cloud GPU offloading for video generation
  - New `runpod_client.py` backend module with async RunPod API client
  - RunPod worker Docker image (`deploy/runpod/Dockerfile`) with ComfyUI + essential custom nodes
  - RunPod handler (`deploy/runpod/handler.py`) bridging RunPod API to ComfyUI
  - 5 new RunPod API endpoints: `/runpod/status`, `/runpod/health`, `/runpod/job/{id}`, `/runpod/cancel/{id}`, `/runpod/submit`
  - `compute_target` parameter on all 5 async generation endpoints (wan22, blockswap-q8, distorch2-q8, ultra-q8, ltx2-i2v)
  - Cloud/Local toggle in ImageToVideoTool and TextToVideoTool frontends
  - RunPod deployment documentation (`deploy/runpod/README.md`)
- Health endpoint now reports `runpod_available` status
