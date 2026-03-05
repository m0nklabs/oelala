### Changed

- **Storage migration (Phase 1-3)**: Migrated all primary content serving, writing, and deletion operations from local filesystem to oelala-storage API
  - Moved primary data directory from `media/` to external `/home/flip/oelala-storage-data/` (symlink preserved for backward compat)
  - All serving endpoints (`/comfyui/output/`, `/media/generated/`, `/files/`, `/videos/`, `/avatars/`) now proxy through oelala-storage API
  - ComfyUI output downloads (images + videos) automatically uploaded to `generated` bucket
  - Cloud Max output saved directly to storage instead of local disk
  - Avatar uploads go to `avatars` bucket via StorageClient
  - Delete and batch-download-zip endpoints use storage API
  - `list-videos` endpoint reads from storage buckets
  - Removed `StaticFiles` mounts for `/comfyui-output` and `/avatars`
  - StorageClient enhanced with `get_with_metadata()` and `stream()` methods
  - `on_job_complete` / `on_job_complete_async` now fall back to storage when local file not found

### Added

- `docs/STORAGE_MIGRATION_PLAN.md` — comprehensive 5-phase migration plan
- `_storage_proxy_response()` helper in app.py for consistent storage-backed content serving with CORS headers

### Fixed

- Cloud Max `saved_path` now correctly handled as storage key (not local path) in user media upload flow
