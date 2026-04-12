### Added

- LoRA support for LTX-2.3 cloud workflows (T2V and I2V)
  - `build_cloud_ltx23_t2v_workflow()` and `build_cloud_ltx23_i2v_workflow()` now accept `lora_configs` parameter
  - Inserts `LoraLoaderModelOnly` nodes between checkpoint loader and CFGGuider
  - Supports chaining multiple LoRAs with individual strength controls
- `lora_configs` form parameter added to `/generate-ltx2-i2v-async` endpoint
- `_build_lora_download_list()` now supports `name` key for single-stage models (in addition to existing `high`/`low` keys for Wan2.2)
- LoRA download URLs are generated for LTX-2.3 cloud jobs, enabling RunPod workers to fetch LoRAs on demand
- `_filter_loras_by_model_compat()` validates LoRA architecture compatibility before sending to cloud workers (prevents Wan LoRAs on LTX jobs and vice versa)
- `_sanitize_lora_configs_for_single_stage()` converts Wan2.2 dual-stage `{high, low}` configs to single-stage `{name}` format for LTX endpoints
- Frontend LoRA dropdown filters by model category (`ltx` vs `wan`) and clears selections on model switch

### Fixed

- LTX-2.3 RunPod worker now reads `lora_downloads` key (URL-based) instead of `loras` (HuggingFace format)
- Wan2.2 LoRAs no longer accidentally sent to LTX-2.3 cloud workers (4-layer defense: frontend filter, frontend clear, backend sanitize, backend compatibility check)
