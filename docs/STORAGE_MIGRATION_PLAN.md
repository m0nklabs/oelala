# Storage Migration Plan: Out of oelala, Into oelala-storage

**Status**: Planning  
**Created**: 2026-03-05  
**Goal**: Eliminate ALL local content storage from the oelala directory. Every file lives on oelala-storage nodes, served and replicated independently.

## Current Architecture

```
oelala-backend (Python/FastAPI)
  ├─ Writes directly to local disk (/home/flip/oelala/media/*)
  ├─ Serves files via StaticFiles mounts & FileResponse
  └─ Some paths already use StorageClient (user media)

oelala-storage (Go)
  ├─ Primary: 192.168.1.35:7990 (data: /home/flip/oelala/media/)  ← SAME DIR!
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

**Risk**: Low — only moves the storage service's data dir. Backend still writes to old path until Phase 2.

---

### Phase 2: Migrate backend writes to StorageClient (MEDIUM RISK)

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

### Phase 3: Migrate backend reads to StorageClient (MEDIUM RISK)

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

#### 3b. Replace FileResponse endpoints

**Endpoints to convert**:
| Endpoint | Current Source | Target Bucket/Key |
|----------|----------------|-------------------|
| `GET /comfyui/output/{fn}` | `ComfyUI/output/{fn}` | `generated/{fn}` |
| `GET /media/generated/{fn}` | `media/generated/{fn}` | `generated/{fn}` |
| `GET /media/generated/cloud-max/{fn}` | `media/generated/cloud-max/{fn}` | `generated/cloud-max/{fn}` |

#### 3c. Gallery API filesystem scans → Storage list

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

### Phase 4: Frontend URL harmonization

All frontend media URLs currently reference specific backend paths:
- `/comfyui/output/{fn}` 
- `/media/generated/{fn}`
- `/media/generated/cloud-max/{fn}`
- `/user/media/{type}/{fn}`

**Target**: Single unified pattern:
```
/storage/{bucket}/{key}
```

Or keep backward-compatible routes that internally proxy to storage.

---

### Phase 5: Cloudflare multi-node serving

Once all content is served via oelala-storage API:

1. Set up Cloudflare tunnel for node 2 (e.g. `storage2.oelala.xyz`)
2. Configure Cloudflare load balancer or failover between nodes
3. Frontend can fetch directly from storage nodes (bypassing backend)
4. Backend becomes auth-only gateway, storage serves content directly

---

## Files Requiring Changes

### Backend (src/backend/)

| File | Changes | Priority |
|------|---------|----------|
| `app.py` | Remove local path constants, replace writes/reads with StorageClient | P0 |
| `gallery_api.py` | Replace Path.glob() scans with StorageClient.list() | P0 |
| `admin_api.py` | Replace local file scanning with StorageClient.list() | P1 |
| `profile_api.py` | Replace avatar disk I/O with StorageClient | P1 |
| `comfyui_client.py` | Output download → StorageClient.put() | P0 |
| `media_service.py` | Already uses StorageClient — verify completeness | P2 |
| `storage_client.py` | May need `get_with_content_type()` method for proxying | P1 |

### Frontend (src/frontend/)

| File | Changes | Priority |
|------|---------|----------|
| `api.ts` | Update media URL construction | P1 |
| Tool components (7 files) | Update image/video URL references | P1 |
| `PublishModal.jsx` | Update URL construction | P2 |

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

- [ ] Zero files in `/home/flip/oelala/media/` (except ComfyUI/output which is internal)
- [ ] Zero `Path("/home/flip/oelala/media/...")` in backend code
- [ ] All content accessible via `http://localhost:7990/{bucket}/{key}`
- [ ] Content replicated to node 2 within 5 minutes of creation
- [ ] Gallery, admin, and user endpoints work via storage proxy
- [ ] Frontend serves media from storage (or proxied storage)
