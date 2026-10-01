### Fixed
- `tests/gpu/test_integration.py` no longer fails on a healthy install. Both
  `TestModelsAvailable` tests asserted that local weights exist under
  `ComfyUI/models/unet` and `ComfyUI/models/checkpoints`, but ComfyUI loads models
  from the main tree *plus* every extra path in `extra_model_paths.yaml`, and on this
  host the real weights live at the configured extra root. The placeholder-only main
  directories therefore reported "no models" while the models were in fact present.
  A new `_configured_model_dirs()` helper mirrors ComfyUI's own path resolution
  (`utils/extra_config.py`), and the hardcoded `/home/flip/oelala/...` literals are
  gone in favour of the existing `LOCAL_COMFYUI_DIR` variable with a
  repo-relative fallback. The stale "unet model for Wan2.2" docstring went with it,
  since Wan 2.2 was retired in 2026-10.
