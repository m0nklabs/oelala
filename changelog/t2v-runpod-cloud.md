### Added
- T2V RunPod cloud GPU support via `/generate-text` endpoint
  - New `compute_target` parameter: `local` (default) or `cloud`
  - Cloud routing uses `build_cloud_max_t2v_workflow()` (fp8 precision, single-GPU compatible)
  - 2x credit multiplier for cloud T2V (same as Cloud Max)
  - LoRA support via signed URL download list for cloud worker
  - Full cloud params: steps, cfg, seed, negative_prompt, shift, sampler, scheduler
- Frontend wan22 cloud toggle now functional (previously forced back to local)
- Dynamic model info badge shows cloud vs local GPU status for wan22
