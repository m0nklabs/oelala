# MinIO Migration Plan

> Replacing custom oelala-storage (Go/Fiber/BadgerDB) with MinIO.
> Created: 2026-07-23

## Decision Summary

**Why**: oelala-storage is reinventing distributed object storage poorly. MinIO provides:
- Built-in admin console (monitoring, health, IAM)
- Native B2 remote tiering via ILM (replacing manual dual-write)
- Webhook notifications (replacing custom webhook code)
- Battle-tested S3 API (48k+ GitHub stars)
- Proper IAM / bucket policies for user isolation

**Trade-off**: ~2x memory footprint (~300MB vs ~155MB per node). Acceptable.

---

## Current State Inventory

### Data (~1.1 GB real data)
| Bucket | Size | Purpose | Migrate? |
|--------|------|---------|----------|
| `generated/` | 752 MB | AI-generated media (videos, images) | ✅ Yes |
| `comfyui-local/` | 234 MB | Direct ComfyUI output | ✅ Yes |
| `users/` | 111 MB | Per-user media (`users/{user_id}/...`) | ✅ Yes |
| `metadata/` | 204 KB | BadgerDB metadata (internal) | ❌ Skip |
| `dedup/` | 32 KB | BadgerDB dedup index (internal) | ❌ Skip |
| `apikeys/` | 32 KB | BadgerDB API keys (internal) | ❌ Skip |
| `mybucket/` | 20 KB | Test data | ❌ Skip |
| `test-*`, `secret/`, `blobs/` | <20 KB | Test/dev artifacts | ❌ Skip |

### Files That Import Storage (11 files)
| File | What it does | Migration effort |
|------|-------------|-----------------|
| `storage_client.py` | Core client (httpx → oelala-storage) | **Full rewrite** → minio SDK |
| `b2_client.py` | B2 dual-write via boto3 | **Delete** → MinIO ILM replaces |
| `app.py` | Main API, file serving, upload, delete | Update imports |
| `comfyui_client.py` | Auto-upload generated media | Update imports |
| `admin_api.py` | Admin media listing | Update imports |
| `gen_artifacts.py` | Workflow JSON/manifest storage | Update imports |
| `profile_api.py` | Avatar upload to `avatars` bucket | Update imports |
| `gallery_api.py` | Workflow extraction from user media | Update imports |
| `media_service.py` | Async media service (direct httpx URLs) | **Rewrite** URLs → presigned |
| `storage_nodes_api.py` | Heartbeat receiver, node listing | **Delete** → MinIO handles |
| Frontend (`api.ts`, tools) | `STORAGE_BASE` URL | Update URL |

### Environment Variables (in `.env`)
| Current Var | Current Value | After Migration |
|-------------|---------------|----------------|
| `STORAGE_URL` | `http://localhost:7990` | `http://localhost:9000` (MinIO API) |
| `STORAGE_API_KEY` | `10dd456a...` | MinIO access key |
| `B2_ENDPOINT` | `s3.eu-central-003.backblazeb2.com` | **Remove** (MinIO ILM handles) |
| `B2_KEY_ID` | B2 app key ID | Move to MinIO tier config |
| `B2_APPLICATION_KEY` | B2 app key | Move to MinIO tier config |
| `B2_BUCKET_NAME` | `oelala-media-eu` | Move to MinIO tier config |
| — (new) | — | `MINIO_ROOT_USER` |
| — (new) | — | `MINIO_ROOT_PASSWORD` |
| — (new) | — | `MINIO_ACCESS_KEY` (for oelala backend) |
| — (new) | — | `MINIO_SECRET_KEY` (for oelala backend) |

### Services to Replace
| Current (systemd) | Machine | After |
|-------------------|---------|-------|
| `oelala-storage.service` (port 7990) | ai-kvm2 | `minio.service` (port 9000 API, 9001 Console) |
| `oelala-storage.service` (port 7990) | 192.168.1.62 | `minio.service` (port 9000 API, 9001 Console) |
| `oelala-storage-autoupdate.timer` | ai-kvm2 | **Remove** (MinIO has its own update mechanism) |

### Cloudflare Tunnels
| Hostname | Current Target | After |
|----------|---------------|-------|
| `storage.oelala.xyz` | `http://localhost:7990` | `http://localhost:9000` |
| `storage2.oelala.xyz` | `http://localhost:7990` | `http://localhost:9000` |
| — (new) `console.oelala.xyz` | — | `http://localhost:9001` (MinIO Console) |

---

## Migration Phases

### Phase 1: MinIO Infrastructure (Node 1)
> Install MinIO, create buckets, verify it works locally.

- [ ] Install MinIO binary on ai-kvm2
- [ ] Create systemd service (`minio.service`)
- [ ] Configure data directory (`/home/flip/minio-data/`)
- [ ] Set root credentials (env vars)
- [ ] Start MinIO, verify Console at `:9001`
- [ ] Create buckets: `generated`, `comfyui-local`, `users`, `avatars`, `uploads`, `archive`, `temp`
- [ ] Create IAM user + access key for oelala-backend
- [ ] Set bucket policies (public-read for `generated`, private for `users`)

### Phase 2: Data Migration
> Copy all existing data from oelala-storage to MinIO.

- [ ] Install `mc` (MinIO CLI)
- [ ] Run `mc mirror` from oelala-storage-data filesystem to MinIO for each bucket:
  - `mc mirror /home/flip/oelala-storage-data/generated/ myminio/generated/`
  - `mc mirror /home/flip/oelala-storage-data/comfyui-local/ myminio/comfyui-local/`
  - `mc mirror /home/flip/oelala-storage-data/users/ myminio/users/`
- [ ] Verify object counts and sizes match
- [ ] Verify random samples are accessible via MinIO API

### Phase 3: B2 Remote Tier
> Configure Backblaze B2 as ILM tier in MinIO.

- [ ] Add B2 as remote tier: `mc ilm tier add s3 myminio B2_COLD ...`
- [ ] Create ILM rule: transition to B2 after 30 days (configurable)
- [ ] Verify tiering works with a test object
- [ ] Document lifecycle policy

### Phase 4: Backend Rewrite
> Replace oelala-storage httpx client with MinIO Python SDK.

- [ ] `pip install minio` in GPU venv
- [ ] Rewrite `storage_client.py`:
  - Replace `httpx.Client` with `minio.Minio` client
  - Keep same public API (put, get, delete, list, exists, head, stream, etc.)
  - Remove B2 dual-write logic (MinIO ILM handles this)
  - Use presigned URLs for external access instead of direct storage URLs
- [ ] Delete `b2_client.py` (no longer needed)
- [ ] Delete `storage_nodes_api.py` (no longer needed — MinIO handles cluster)
- [ ] Update `media_service.py`:
  - Replace direct httpx URL construction with presigned URL generation
  - Remove storage_url / storage_token config (use MinIO client)
- [ ] Update `app.py`: remove storage proxy endpoints if no longer needed
- [ ] Update env vars in `.env`: add MINIO_*, remove B2_*
- [ ] Run all tests, verify nothing breaks

### Phase 5: Cloudflare & DNS
> Switch tunnels from oelala-storage to MinIO.

- [ ] Update Cloudflare tunnel config: `storage.oelala.xyz` → `localhost:9000`
- [ ] Add `console.oelala.xyz` → `localhost:9001` (MinIO Console)
- [ ] Restart cloudflared
- [ ] Verify external access works
- [ ] Update frontend `STORAGE_BASE` URL if changed

### Phase 6: Webhook Notifications
> Set up MinIO bucket notifications to replace oelala-storage webhooks.

- [ ] Configure webhook target: `https://api.oelala.xyz/webhooks/storage`
- [ ] Add event subscriptions for `generated` bucket (PUT, DELETE)
- [ ] Test notification delivery
- [ ] Remove old webhook handling code from backend if applicable

### Phase 7: Node 2 (Site Replication)
> Set up MinIO on node 2 for redundancy.

- [ ] Install MinIO on 192.168.1.62
- [ ] Create systemd service
- [ ] Configure as site replication peer (NOT a distributed cluster — 2 independent sites)
- [ ] Set up site replication: `mc admin replicate add site1 site2`
- [ ] Update `storage2.oelala.xyz` tunnel → `localhost:9000`
- [ ] Verify replication works

### Phase 8: Cleanup
> Remove oelala-storage and all traces.

- [ ] Stop and disable `oelala-storage.service` on both nodes
- [ ] Stop and disable `oelala-storage-autoupdate.timer` on ai-kvm2
- [ ] Archive `/home/flip/oelala-storage-data/` (keep backup for 30 days)
- [ ] Archive oelala-storage GitHub repo (set to archived)
- [ ] Remove `/home/flip/oelala-storage/` directory
- [ ] Remove `oelala-storage-data/` from workspace
- [ ] Update all documentation:
  - `copilot-instructions.md` — remove oelala-storage references, add MinIO
  - `MARK1.md` — update resource inventory
  - `docs/CLOUDFLARE_SETUP.md` — update tunnel targets
  - `README.md` — update architecture section
  - `CHANGELOG.md` — via changelog fragment
  - Repo memory (`/memories/repo/oelala.md`)
- [ ] Create changelog fragment: `changelog/XXX-minio-migration.md`

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Data loss during migration | High | `mc mirror` + verify + keep oelala-storage running in parallel until confirmed |
| Downtime during switchover | Medium | Run both systems side-by-side, switch DNS atomically |
| Frontend URL breakage | Medium | Keep same URL pattern (`storage.oelala.xyz/{bucket}/{key}`), MinIO is S3-compatible |
| B2 ILM misconfiguration | Low | Test with dummy objects first, B2 data is a backup not primary |
| MinIO memory usage higher | Low | ~300MB vs ~155MB — acceptable on ai-kvm2 |
| Node 2 replication lag | Low | Site replication is eventually consistent, acceptable for our use case |

## Non-Goals (Explicitly Out of Scope)
- Multi-drive erasure coding (overkill for ~1GB data)
- MinIO operator / Kubernetes deployment
- Encryption at rest (Cloudflare tunnel provides transport encryption)
- Migrating existing B2 data back to MinIO (stays as cold archive)

---

## Estimated Effort Per Phase

| Phase | Effort | Dependencies |
|-------|--------|-------------|
| 1. Infrastructure | Small | None |
| 2. Data Migration | Small | Phase 1 |
| 3. B2 Remote Tier | Small | Phase 1 |
| 4. Backend Rewrite | **Medium** | Phase 1 |
| 5. Cloudflare & DNS | Small | Phase 1 |
| 6. Webhooks | Small | Phase 4 |
| 7. Node 2 | Small | Phase 1, 2 |
| 8. Cleanup | Small | All above |

Phase 4 (backend rewrite) is the largest piece of work — ~600 lines in `storage_client.py` to rewrite plus updates in `media_service.py` and removal of `b2_client.py` + `storage_nodes_api.py`.
