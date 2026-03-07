### Fixed

- **Cloud generation timeout bug**: `get_comfyui_queue` was firing a timeout based on local
  cache state (`_cloud_status="IN_QUEUE"`) without ever polling RunPod. Jobs running longer
  than `CLOUD_QUEUE_TIMEOUT_SECONDS` (default 300s) would be killed prematurely even though
  RunPod was still executing them. Timeout now only fires via `_handle_cloud_job_status`,
  which actually polls RunPod before deciding to cancel. Raised default timeout to 1800s.

- **Credit refund on cloud failure**: Added `_refund_cloud_job_credits()` helper that
  automatically refunds credits when a cloud job fails (FAILED / CANCELLED / TIMED_OUT
  RunPod status, or local queue timeout). Refund is idempotent via `_credit_refunded` flag.
  All 7 cloud job types now include `credits_required` in their job_info dict.

### Added

- `scripts/recover_cloud_job.py`: One-shot recovery tool to download a completed RunPod job
  by job ID, upload it to oelala-storage, and register the user_media record in Supabase.
  Useful when backend in-memory state is lost (restart) and the video was never saved.
