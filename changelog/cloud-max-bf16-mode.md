### Added

- **Cloud Max generation mode** — bf16 full precision on RunPod cloud GPUs (A6000/A40 48GB VRAM)
  - `build_cloud_max_i2v_workflow()` — I2V with dual-pass high/low LoRA support, native UNETLoader
  - `build_cloud_max_t2v_workflow()` — T2V variant without image conditioning
  - `/generate-cloud-max-async` endpoint — cloud-only, supports both I2V and T2V via `mode` parameter
  - Frontend: Cloud Max option in ImageToVideoTool (top of model list) with auto-locked cloud compute
  - Frontend: Cloud Max option in TextToVideoTool with auto-set defaults (25 steps, CFG 3.0)
  - Model download script: `deploy/runpod/download_cloud_max_models.sh` for Network Volume setup
- **Cloud Max defaults**: dpmpp_2m sampler, beta scheduler, 25 steps, CFG 3.0, shift 8.0, high_noise_steps 12
- **Models used**: wan2.1_i2v_720p_14B_bf16 (32.8GB), wan2.1_t2v_14B_bf16 (28.6GB), umt5_xxl_fp16 (11.4GB), wan_2.1_vae (254MB), clip_vision_h (1.26GB)
- **Cost**: 2x credits multiplier for premium quality, ~$1.22/hr on A6000/A40 tier
