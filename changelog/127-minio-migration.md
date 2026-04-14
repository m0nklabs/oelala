### Changed
- **storage_client.py**: Rewritten to use MinIO Python SDK (`minio==7.2.20`) instead of httpx + custom oelala-storage HTTP API. Same public API surface — all callers (app.py, admin_api.py, gallery_api.py, comfyui_client.py, gen_artifacts.py, profile_api.py) continue to work unchanged.
- **media_service.py**: Signed URLs now use MinIO's native S3 presigned URLs (SigV4) instead of custom HMAC-SHA256 scheme. Upload and delete operations use MinIO SDK via storage_client instead of raw httpx. User quota is calculated from Supabase `user_media` table instead of custom `/buckets/{user_id}` endpoint.
- **admin_api.py**: Health check uses MinIO bucket-list instead of `localhost:7990/health`. Allowed systemd service list updated from `oelala-storage` to `minio`.
- **app.py**: Storage proxy error handling uses `S3Error` instead of `httpx.HTTPStatusError`. Removed `storage_nodes_api` router. Health endpoint uses MinIO SDK instead of httpx to `localhost:7990`.
- **config.js**: `STORAGE_BASE` dev URL updated from `localhost:7990` to `localhost:9000` (MinIO). Comments updated.
- **AdminStorageNodesTab.jsx**: Rewritten from heartbeat-based node listing (called deleted `/api/storage-nodes/`) to MinIO health status card using `/api/admin/system/health`.
- **AdminSystemTab.jsx**: Service list updated from `oelala-storage` to `minio`.
- Code comments across backend updated from "oelala-storage" to "MinIO" (gallery_api, profile_api, comfyui_client, app).

### Added
- `minio==7.2.20` dependency in `requirements.txt`
- `get_storage_client` alias in storage_client.py (fixes pre-existing import in gallery_api.py)
- `presigned_get()` and `presigned_put()` methods on StorageClient for native S3 presigned URLs
- Bucket name mapping: `generated` → `oelala-generated`, `comfyui-local` → `oelala-comfyui`, `avatars` → `oelala-avatars`, `users` → `oelala-users`
- New env vars: `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` (with fallback to old `STORAGE_URL`/`STORAGE_API_KEY`)
- Unit tests: `tests/test_storage_client.py` (33 tests), `tests/test_media_service.py` (11 tests)

### Removed
- **b2_client.py**: Deleted — Backblaze B2 dual-write/fallback removed. B2 interaction will be managed by MinIO ILM tiering.
- **storage_nodes_api.py**: Deleted — storage node heartbeat/registration no longer needed (MinIO handles cluster internally).
- Removed `MEDIA_SIGNING_SECRET` dependency (replaced by MinIO presigned URLs)
- Removed `STORAGE_NODE_API_KEY` (was `dev-secret-key-12345` default)
- Removed B2 env vars from `.env.example` (`B2_ENDPOINT`, `B2_KEY_ID`, `B2_APPLICATION_KEY`, `B2_BUCKET_NAME`)
