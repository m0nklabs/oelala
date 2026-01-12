### Added

- **MediaService** (`src/backend/media_service.py`): Unified interface for oelala-storage + Supabase metadata sync
  - Upload files with automatic metadata tracking in Supabase `user_media` table
  - Generate signed URLs for temporary public access (HMAC-SHA256, 1-hour default)
  - List, get, delete user media
  - MediaRecord dataclass matching existing `006_user_media.sql` schema
- Storage integration config in `.env.example`: `STORAGE_URL`, `STORAGE_API_KEY`, `MEDIA_SIGNING_SECRET`

### Changed

- oelala-storage: Added signed URL support in auth middleware
  - Query params: `?expires={unix_timestamp}&sig={hmac_sha256_hex}`
  - Valid signatures bypass auth, return anonymous user context
  - Invalid/expired signatures return 401
  - Config: `security.signing_secret` in `oelala-storage.yaml`
