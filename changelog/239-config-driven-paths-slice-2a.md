### Changed

- `yt-dlp` is resolved instead of hardcoded: `YT_DLP_PATH`, then `PATH`, then the binary next to
  the interpreter running the backend. The old fallback named one specific virtualenv, which a
  rebuild would rename.
- The MinIO data directory defaults to `<home>/minio-data` instead of a literal host path, and
  the lookup lives in one helper (`comfyui_client.minio_data_dir()`) so the guard that protects
  MinIO's own files reads it from a single place.
- `.env.example` documents `YT_DLP_PATH` and uses a placeholder for `MINIO_DATA_DIR` instead of a
  real path — rule 6 covers committed templates too.

### Added

- `tests/test_config_path_resolution.py` pins both resolution orders.
- The rule-6 ratchet now also scans committed configuration templates, so a real host path in
  `.env.example` fails the build instead of surviving review.
