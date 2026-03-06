# Storage Migration Plan: Final State

**Status**: ✅ COMPLETE (Phase 1-5 done, issue #110 closed)
**Created**: 2026-03-05
**Updated**: 2026-03-06
**Goal**: Keep oelala application state separate from media object storage, with storage nodes acting as the source of truth and local disk used only for temporary processing.

## Final Architecture

```
oelala-backend (Python/FastAPI)
  ├─ Temporary processing only: /tmp/oelala_uploads, /tmp/oelala_generated
  ├─ Reads/writes content via StorageClient
  ├─ Proxies selected media routes for compatibility
  └─ Uploads/cleans up local temp files after success

oelala-storage (Go)
  ├─ storage-main.oelala.xyz     → coordinator / primary node (:7990)
  ├─ storage-node-01.oelala.xyz  → additional local node (:7993)
  ├─ storage2.oelala.xyz         → remote node 2
  └─ Handles metadata, dedup, retention, signed/public URL behavior
```

## What Changed

### Phase 1: Primary storage data moved out of oelala

- Primary storage data lives under `/home/flip/oelala-storage-data/`
- The old media path stopped being the canonical storage location
- Any remaining compatibility symlink or fallback path is transitional, not authoritative

### Phase 2: Backend writes moved to StorageClient

Completed write-path migrations include:

- ComfyUI output upload to storage after generation
- Cloud generation output upload to storage
- Avatar/user media flows using storage buckets
- Job completion paths that now upload first and clean up local files afterward

### Phase 3: Backend reads moved to storage-backed routes

Completed read-path migrations include:

- Gallery/admin listing flows now use storage APIs instead of raw filesystem scans
- Proxy routes serve storage-backed media for compatibility
- Storage metadata is used for size/time information where available

### Phase 4: Frontend harmonized around storage-aware URLs

- `apiFetch()` standardized frontend auth/CORS behavior
- `getMediaUrl()` can resolve relative paths, signed URLs, and protected media access
- Media display no longer assumes everything is a plain local file path

### Phase 5: Multi-node delivery enabled

- Storage hostnames and tunnels are documented and active
- Each node owns its own Cloudflare tunnel and can operate independently
- Backend proxy routes remain in place as a fallback and compatibility layer

## Current Hostnames

| Role | Hostname | Notes |
|------|----------|-------|
| Primary/coordinator | `storage-main.oelala.xyz` | Main storage entrypoint |
| Local node 01 | `storage-node-01.oelala.xyz` | Extra local node on separate ports |
| Remote node 02 | `storage2.oelala.xyz` | Separate machine and tunnel |

## What Still Exists Locally

These are expected and not a rollback of the migration:

| Path / Constant | Purpose |
|-----------------|---------|
| `ComfyUI/output/` | ComfyUI's own generation output before backend post-processing |
| `/tmp/oelala_uploads` | Temporary upload staging |
| `/tmp/oelala_generated` | Temporary generated-file staging |
| Selected proxy routes in backend | Compatibility for existing frontend/media URLs |

## Important Notes

- Local permanent media directories are no longer the design target.
- Storage buckets, metadata, and storage-hosted URLs are the authoritative media layer.
- Cleanup after upload is now part of the intended behavior to keep the app repo free of persistent media buildup.
- Remaining fallback code exists to reduce breakage during rollout, not because the migration is unfinished.

## Follow-Up Work After Migration

- Remove more legacy fallback code once all consumers use the new storage hostnames cleanly
- Improve direct-public/private media serving strategy
- Expand node registration, heartbeat, and admin observability for the storage cluster

Each phase is independently reversible:
- Phase 1: Change config back to old path
- Phase 2-3: Feature flag `USE_STORAGE_CLIENT=true/false`, fall back to local disk
- Phase 4: Keep old URL routes as aliases
- Phase 5: Cloudflare config change only

## Success Criteria

- [x] Data directory moved out of oelala repo to `/home/flip/oelala-storage-data/`
- [x] All content accessible via `http://localhost:7990/{bucket}/{key}`
- [x] Content replicated to node 2 within 5 minutes of creation
- [x] All primary serving endpoints proxy through storage API
- [x] All write operations go through StorageClient
- [x] Gallery, admin, and user endpoints work via storage proxy
- [x] Dead path constants removed (AVATARS_DIR, CLOUD_MAX_OUTPUT_DIR, THUMBNAIL_DIR)
- [x] Unified `/storage/{bucket}/{key}` route added with bucket whitelist
- [x] Metadata endpoint uses storage fallback (temp file for ffprobe)
- [x] Frontend verified: `getMediaUrl()`, `PublishModal`, all 7 tools
- [x] Cloudflare tunnels for storage nodes (Phase 5)
- [x] Frontend STORAGE_BASE configured for `https://storage.oelala.xyz`
- [ ] Remove backend proxy endpoints (future cleanup — low priority)
- [x] StaticFiles mounts removed (except `/static` for frontend)
- [ ] Zero `Path("/home/flip/oelala/media/...")` constants in backend code (some remain as fallback refs)
- [x] Frontend URL harmonization (Phase 4)
- [x] Cloudflare multi-node serving (Phase 5) — decentralized, each node own tunnel
- [ ] Remove symlink at `/home/flip/oelala/media` and local path constants (future cleanup)
- [x] Frontend serves media from storage (or proxied storage)
