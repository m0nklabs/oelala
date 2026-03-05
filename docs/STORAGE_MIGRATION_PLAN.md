# Storage Migration Plan: Out of oelala, Into oelala-storage

**Status**: ✅ COMPLETE (Phase 1-5 done, issue #110 closed)  
**Created**: 2026-03-05  
**Updated**: 2026-03-05  
**Goal**: Eliminate ALL local content storage from the oelala directory. Every file lives on oelala-storage nodes, served and replicated independently.

## Current Architecture

```
oelala-backend (Python/FastAPI)
  ├─ Writes directly to local disk (/home/flip/oelala/media/*)
  ├─ Serves files via StaticFiles mounts & FileResponse
  └─ Some paths already use StorageClient (user media)

oelala-storage (Go)
  ├─ Primary: 192.168.1.35:7990 (data: /home/flip/oelala-storage-data/)  ← MOVED ✅
  ├─ Node 2:  192.168.1.62:7990 (data: ~/oelala-storage-node-data/)
  └─ Auto-sync between nodes every 5 minutes
```

**Problem**: oelala-storage's primary uses `/home/flip/oelala/media/` as its data directory. The backend ALSO reads/writes directly to this path. This creates two access patterns to the same files.

## Target Architecture

```
oelala-backend (Python/FastAPI)
  ├─ ALL storage via StorageClient → http://localhost:7990
  ├─ NO local disk writes for content
  └─ Serves files by proxying from storage service

oelala-storage
  ├─ Primary: 192.168.1.35:7990 (data: /home/flip/oelala-storage-data/)  ← MOVED
  ├─ Node 2:  192.168.1.62:7990 (data: ~/oelala-storage-node-data/)
  ├─ Node N:  (future Cloudflare-proxied nodes)
  └─ Auto-sync, content-addressed dedup

Frontend → Cloudflare → Any node (failover between nodes)
```

## Migration Phases

### Phase 1: Move primary data directory out of oelala (LOW RISK)

**What**: Change primary's `storage.path` from `/home/flip/oelala/media/` to `/home/flip/oelala-storage-data/`

**Steps**:
1. `mkdir /home/flip/oelala-storage-data`
2. `rsync -a /home/flip/oelala/media/ /home/flip/oelala-storage-data/`
3. Update `oelala-storage.yaml`: `storage.path: "/home/flip/oelala-storage-data/"`
4. Restart oelala-storage service
5. Verify health + file access via API
6. Keep `/home/flip/oelala/media/` as read-only fallback during transition

**Risk**: Low — only moves the storage service's data dir.

**Status**: ✅ COMPLETE — Data moved to `/home/flip/oelala-storage-data/`, symlink at old path for backward compat.

---

### Phase 2: Migrate backend writes to StorageClient (MEDIUM RISK) ✅ COMPLETE

Replace ALL local filesystem writes with `StorageClient.put()` calls.

#### 2a. ComfyUI output capture → Storage

**Current** (`app.py` ~L2713):
```python
output_path = COMFYUI_OUTPUT_DIR / output_filename
# File already exists in ComfyUI/output/ from generation
shutil.copy(output_path, OUTPUT_DIR / output_filename)
```

**Target**:
```python
# Read from ComfyUI output, upload to storage
comfyui_path = COMFYUI_OUTPUT_DIR / output_filename
storage = get_storage_client()
storage.put("generated", output_filename, comfyui_path)
```

**Files**: `app.py` (multiple endpoints that call `comfyui_client.get_output_*`)

#### 2b. Cloud Max output → Storage

**Current** (`app.py` ~L2291):
```python
save_path = CLOUD_MAX_OUTPUT_DIR / save_name
with open(save_path, "wb") as f:
    f.write(content)
```

**Target**:
```python
storage = get_storage_client()
storage.put("generated", f"cloud-max/{save_name}", content)
```

#### 2c. User uploads → Storage

**Current** (`app.py` ~L630-640):
```python
dest = UPLOAD_DIR / filename
dest.write_bytes(data)
```

**Target**: Already partially migrated via `media_service.py`. Complete the migration for remaining direct writes.

#### 2d. Avatar uploads → Storage

**Current** (`profile_api.py` ~L521):
```python
avatar_path = AVATARS_DIR / f"{user.id}.jpg"
img.save(avatar_path)
```

**Target**:
```python
storage = get_storage_client()
buf = io.BytesIO()
img.save(buf, format="JPEG")
storage.put("avatars", f"{user.id}.jpg", buf.getvalue())
```

---

### Phase 3: Migrate backend reads to StorageClient (MEDIUM RISK) ✅ COMPLETE

Replace ALL local filesystem reads and `FileResponse` with storage proxy endpoints.

#### 3a. Replace StaticFiles mounts with proxy endpoints

**Current** (`app.py` ~L651-659):
```python
app.mount("/comfyui-output", StaticFiles(directory=COMFYUI_OUTPUT_DIR))
app.mount("/avatars", StaticFiles(directory=AVATARS_DIR))
```

**Target**: Replace with dynamic endpoints that proxy from storage:
```python
@app.get("/media/{bucket}/{path:path}")
async def serve_media(bucket: str, path: str):
    storage = get_storage_client()
    data, content_type = storage.get_with_type(bucket, path)
    return Response(content=data, media_type=content_type)
```

#### 3b. Replace FileResponse endpoints ✅ COMPLETE

**Endpoints to convert**:
| Endpoint | Current Source | Target Bucket/Key |
|----------|----------------|-------------------|
| `GET /comfyui/output/{fn}` | `ComfyUI/output/{fn}` | `generated/{fn}` |
| `GET /media/generated/{fn}` | `media/generated/{fn}` | `generated/{fn}` |
| `GET /media/generated/cloud-max/{fn}` | `media/generated/cloud-max/{fn}` | `generated/cloud-max/{fn}` |

#### 3c. Gallery API filesystem scans → Storage list ✅ COMPLETE

**Current** (`gallery_api.py` ~L889-890):
```python
for f in Path("/home/flip/oelala/media/generated/").glob("*"):
    ...
for f in Path("/home/flip/oelala/ComfyUI/output").glob("*"):
    ...
```

**Target**:
```python
storage = get_storage_client()
objects = storage.list("generated")
```

---

### Phase 4: Frontend URL harmonization ✅ COMPLETE

All frontend media URLs reference specific backend paths that proxy to storage:
- `/comfyui/output/{fn}` → `comfyui-local` bucket
- `/media/generated/{fn}` → `generated` bucket
- `/media/generated/cloud-max/{fn}` → `generated/cloud-max/` in bucket
- `/user/media/{type}/{fn}` → `users/{uid}/{type}/{fn}` in bucket

**Unified route added**: `GET /storage/{bucket}/{key}` serves content from whitelisted
buckets (`generated`, `comfyui-local`, `avatars`) with path traversal protection.
Old backward-compatible routes remain functional.

**Frontend**: `getMediaUrl()` in `config.js` handles signed URLs and relative paths.
`STORAGE_BASE` ready for Phase 5 (Cloudflare direct serving).
`PublishModal.jsx` uses `apiFetch` and existing routes — no changes needed.

---

### Phase 5: Cloudflare multi-node serving ✅ COMPLETE

Both storage nodes have independent Cloudflare tunnels:

| Node | Tunnel | Hostname | Tunnel ID |
|------|--------|----------|-----------|
| Primary (192.168.1.35) | oelala-main | `storage.oelala.xyz` | `b34ce27b-e9b1-4926-b5fe-ebbaf42d506a` |
| Node 2 (192.168.1.62) | oelala-storage-node2 | `storage2.oelala.xyz` | `83d253c4-24eb-4643-b36f-174a2fc3f10b` |

**Architecture**: Fully decentralized — each node runs its own cloudflared instance,
its own tunnel, its own credentials. No single point of failure between nodes.

**CORS**: Go storage service configured with explicit AllowOrigins (oelala.xyz, localhost dev).

**Frontend**: `STORAGE_BASE` in `config.js` points to `https://storage.oelala.xyz`.

**Remaining**: Backend proxy endpoints (`/storage/{bucket}/{key}`) kept as fallback.
Can be removed when all frontend paths use `STORAGE_BASE` directly.

---

## Files Requiring Changes

### Backend (src/backend/)

| File | Changes | Status |
|------|---------|--------|
| `app.py` | Remove local path constants, replace writes/reads with StorageClient | ✅ Done |
| `gallery_api.py` | Replace Path.glob() scans with StorageClient.list() | ✅ Done |
| `admin_api.py` | Replace local file scanning with StorageClient.list() | ✅ Done |
| `profile_api.py` | Replace avatar disk I/O with StorageClient | ✅ Done |
| `comfyui_client.py` | Output download → StorageClient.put() | ✅ Done |
| `media_service.py` | Already uses StorageClient | ✅ Verified |
| `storage_client.py` | Added `get_with_metadata()`, `stream()` methods | ✅ Done |

### Frontend (src/frontend/)

| File | Changes | Status |
|------|---------|--------|
| `config.js` | `getMediaUrl()` handles signed URLs + relative paths | ✅ No change needed |
| Tool components (7 files) | Use `apiFetch` for all media fetching | ✅ Already migrated |
| `PublishModal.jsx` | Uses `apiFetch` + existing proxy routes | ✅ No change needed |

### Remaining Local Path Constants (Intentionally Kept)

| Constant | File | Reason |
|----------|------|--------|
| `UPLOAD_DIR` | `app.py` | Staging dir for ComfyUI uploads |
| `OUTPUT_DIR` | `app.py` | Local generated output dir |
| `COMFYUI_OUTPUT_DIR` | `app.py` | ComfyUI writes here directly |
| `MEDIA_GENERATED_DIR` | `admin_api.py` | Fallback when storage unavailable |
| `COMFYUI_OUTPUT_DIR` | `admin_api.py` | Fallback when storage unavailable |

### Workflows

| File | Changes | Priority |
|------|---------|----------|
| All workflow JSON files | Output paths are ComfyUI-internal, no change needed | N/A |

## ComfyUI Output Special Case

ComfyUI writes to `/home/flip/oelala/ComfyUI/output/` directly during generation. This cannot be changed — it's ComfyUI's internal behavior. 

**Solution**: After each generation completes, the backend copies output from `ComfyUI/output/` → `StorageClient.put("generated", ...)`. This is already how it works, just needs to target Storage API instead of local disk.

The `comfyui-local` symlink in media/ can be removed after migration.

## Rollback Strategy

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
