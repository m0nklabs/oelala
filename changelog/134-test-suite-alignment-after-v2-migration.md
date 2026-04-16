### Fixed
- Updated generation adapter tests to match current behavior (`wan22-local-t2i` now returns `completed` for synchronous local execution).
- Updated media signed URL tests for `public_url`-first behavior with presigned fallback.
- Updated WebSocket progress tests for structured `job_ownership` metadata.
- Updated GPU endpoint existence tests to allow `401` when auth is required.

### Changed
- Removed import-time `sys.modules["storage_client"]` monkeypatch from `tests/test_auto_upload.py` to prevent test-order contamination across the full suite.

### Verified
- Full backend suite: `473 passed, 9 skipped`.
- Frontend production build succeeds via `Frontend: build` task.
