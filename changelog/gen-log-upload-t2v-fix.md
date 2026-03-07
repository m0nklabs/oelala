### Added
- Gen log upload to user storage bucket for every generation (success + failure)
  - New `_upload_gen_log_to_user_storage()` helper uploads logs to `generated/users/{user_id}/logs/{ts}_{status}_{prompt_id[:8]}.log`
  - `record_generation_start()` now stores `user_id` from `job_info` into `active_jobs`
  - `record_generation_complete()` extended with optional `log_text` param; auto-generates summary log when no log text provided
  - Cloud (RunPod) generation logs (from `output.get("logs")`) now uploaded per-user on both success and failure
  - T2V local path now registered in `active_jobs` with `user_id` for stats + log tracking

### Fixed
- `user_id` missing from `job_info` in local path for `wan22_i2v`, `blockswap_q8_i2v`, and `ltx2_i2v` endpoints — prevents log upload for local gens
- T2V cloud generation failure: `ensure_workflow_models()` incorrectly treating corrupt/empty placeholder model files as valid
  - `_is_model_present()` now checks file size (≥50MB minimum) in addition to file existence
  - `_link_model_into_comfyui()` now removes stale/corrupt files before symlinking
  - `download_requested_models()` now removes stale placeholder files before downloading replacements
