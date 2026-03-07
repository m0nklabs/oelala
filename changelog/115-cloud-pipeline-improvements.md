### Changed
- Cloud job completion now sends WebSocket notifications to the user (`job_complete` / `job_failed` events)
- MyMedia auto-refreshes when a cloud job completes (via `refreshToken` prop from QueueIndicator)
- Storage upload failures during cloud completion no longer mark job as completed — poller retries on next cycle
- `execution_time_s` falls back to elapsed time from `_start_time` when RunPod doesn't provide it
- Completed cloud jobs are cleaned up from `active_jobs` memory after 5 minutes
- QueueIndicator `watchJobs` effect now has proper cleanup to prevent stale async operations
