### Changed
- Metadata endpoint (`/comfyui-metadata/{filename}`) now fetches from oelala-storage when file not found locally, using temp file for ffprobe extraction
- Admin media listing discovers users from storage API instead of filesystem iteration
- Dead local path constants removed: `AVATARS_DIR`, `CLOUD_MAX_OUTPUT_DIR`, `THUMBNAIL_DIR`

### Added
- Unified storage proxy route `GET /storage/{bucket}/{key}` with bucket whitelist and path traversal protection
- Storage download fallback for metadata extraction (Phase 3b cleanup)

### Fixed
- Removed hardcoded `/home/flip/oelala/media/generated` and `/home/flip/oelala/media/users` filesystem dependencies

### Infrastructure
- Phase 5: Decentralized Cloudflare tunnels — each storage node has its own tunnel (no SPOF)
- Tunnel `oelala-main` (ai-kvm2): oelala.xyz, api.oelala.xyz, storage.oelala.xyz
- Tunnel `oelala-storage-node2` (ubuntu-oelalastorage2): storage2.oelala.xyz
- CORS configured on Go storage service with explicit origins
- Frontend `STORAGE_BASE` set to `https://storage.oelala.xyz`
- Storage migration Phase 1-5 fully complete, issue #110 closed
