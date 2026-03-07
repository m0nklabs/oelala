### Added
- Per-generation artifact storage: workflow.json, manifest.json, input_image, and execution.log saved to user's storage bucket for every generation (local + cloud)
- New `gen_artifacts.py` module for uniform artifact handling
- ComfyUI history execution trace extracted as log for local jobs
- RunPod handler logs saved as execution.log for cloud jobs

### Changed
- `record_generation_complete()` now saves execution logs via `save_gen_logs()` to `users/{user_id}/generations/{prompt_id}/execution.log`
- Removed old `_upload_gen_log_to_user_storage()` in favour of unified artifact storage
- Handler heartbeat interval reduced from 60s to 30s for faster timeout detection
- Cloud gen upload: fixed `register_job_metadata` → `register_job`, fixed `extra_metadata` kwarg, fixed nested settings lookup

### Fixed
- Cloud generation timeout detection improved with 30s heartbeat
- Cloud upload failures due to incorrect function signatures in `comfyui_client.py`
