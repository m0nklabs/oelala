### Fixed

- Fixed the RunPod serverless worker CUDA preflight in `deploy/runpod/handler.py` to support PyTorch builds that expose GPU memory as `total_memory` instead of `total_mem`.
- Verified the backend continues to target the working primary RunPod endpoint `x2x496ymkidl3m` after service restart.
- Added health-aware RunPod endpoint failover so throttled or unhealthy endpoints are skipped for new submissions when `RUNPOD_ENDPOINT_IDS` is configured.
- Updated the RunPod worker to link models from cached-model mounts or Hugging Face cache paths before falling back to live `hf_hub_download()` calls inside paid worker runtime.
- Added a disk-capacity preflight for live Hugging Face downloads so undersized ephemeral workers fail fast with a clear remediation message instead of crashing mid-download.
- Documented the recommended hybrid RunPod model strategy: cached model for the Wan 2.2 Hugging Face repo, small volume for the 2.1 VAE, CLIP Vision, and LoRA cache.
