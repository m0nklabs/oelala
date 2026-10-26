### Added

- Regression tests for user-scoped storage of completed cloud jobs
  (`tests/test_cloud_output_media_visibility.py`). A finished RunPod job must upload to
  the authenticated user's storage and report `storage_path` plus a signed URL, must stay
  `running` when that upload fails so the poller retries, and must never reach
  `completed` without an owner. This keeps cloud output from disappearing from My Media
  behind a green job status.
