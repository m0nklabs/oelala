### Fixed

- **V2 local generation outputs 404 ("Failed to load image")**: `track_v2_local_job`
  dropped `backend_id` from the `active_jobs` entry, so completion polling
  (`_resolve_local_job_result`) fell back to the default ComfyUI client singleton
  instead of the backend-scoped client that holds the `register_job` metadata.
  The auto-upload was skipped ("No job metadata found"), the result URL fell back
  to `/comfyui/output/...`, and that proxy only serves from the `comfyui-local`
  MinIO bucket (which nothing populates) — every affected output returned 404.
  Preserving `backend_id` restores the auto-upload and signed media URLs.
