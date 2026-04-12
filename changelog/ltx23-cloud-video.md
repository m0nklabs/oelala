### Added
- LTX-2.3 22B cloud video generation (replaces LTX-2.0 19B for cloud)
  - New `deploy/runpod-ltx23/` Docker infrastructure (Dockerfile, handler.py, deploy.sh)
  - Separate RunPod endpoint/template for LTX-2.3 (80GB+ GPU requirement)
  - `build_cloud_ltx23_t2v_workflow()` — T2V with 8-step distilled sigma schedule
  - `build_cloud_ltx23_i2v_workflow()` — I2V with LTXVImgToVideoConditionOnly conditioning
  - Programmatic workflow builders (no JSON templates needed)
- `RUNPOD_LTX23_ENDPOINT_ID` environment variable for routing LTX jobs to dedicated endpoint

### Changed
- Cloud LTX I2V/T2V now routes to LTX-2.3 22B endpoint (was LTX-2.0 19B)
- Frontend labels updated: "LTX-2 19B" → "LTX-2.3 22B" in TextToVideoTool and ImageToVideoTool
- T2V_GENERATION_MODES `ltx2` entry updated to reflect LTX-2.3 specs
- LTX-2.3 uses `CheckpointLoaderSimple` + `LTXAVTextEncoderLoader` (Gemma 3 12B fp8)
- Cloud GPU requirement: 80GB+ (AMPERE_80, HOPPER_141, BLACKWELL_96, BLACKWELL_180)

### Technical Details
- LTX-2.3 22B distilled checkpoint: 46.1 GB (bf16) from Lightricks/LTX-2.3
- Gemma 3 12B text encoder: 13.2 GB (fp8_scaled) from Comfy-Org/ltx-2
- Total VRAM: ~59.3 GB → requires A100/H100/B200/GB200
- Distilled sigma schedule: 1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0
- CFG=1.0 (distilled model), euler_ancestral sampler, VAEDecodeTiled
