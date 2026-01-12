### Added

- **MediaService** (`src/backend/media_service.py`): Unified interface for oelala-storage + Supabase metadata sync
  - Upload files with automatic metadata tracking in Supabase `user_media` table
  - Generate signed URLs for temporary public access (HMAC-SHA256, 1-hour default)
  - List, get, delete user media
  - MediaRecord dataclass matching existing `006_user_media.sql` schema
- Storage integration config in `.env.example`: `STORAGE_URL`, `STORAGE_API_KEY`, `MEDIA_SIGNING_SECRET`
- **Backend Integration** (`src/backend/app.py`):
  - `get_media_service()` - Global lazy-initialized MediaService instance
  - `upload_generated_media()` - Helper for post-generation storage upload + Supabase sync
  - `get_signed_media_url()` - Helper for signed URL generation
- Cloudflare Tunnel + CDN setup guide (`docs/CLOUDFLARE_SETUP.md`)

### Changed

- oelala-storage: Added signed URL support in auth middleware
  - Query params: `?expires={unix_timestamp}&sig={hmac_sha256_hex}`
  - Valid signatures bypass auth, return anonymous user context
  - Invalid/expired signatures return 401
  - Config: `security.signing_secret` in `oelala-storage.yaml`
- Wan2.2 I2V endpoint (`/generate-wan22-comfyui`):
  - Now uploads generated videos to oelala-storage
  - Syncs metadata to Supabase
  - Returns `signed_url` (24h) and `storage_path` in response
  - Fallback to local `/files/` URL if storage upload fails
