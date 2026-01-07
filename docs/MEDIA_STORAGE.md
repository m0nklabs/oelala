# Media Storage Architecture

> **Last Updated**: 2026-01-06
> **Related Project**: [oelala-storage](https://github.com/m0nklabs/oelala-storage) (separate repo)

## Overview

Storage is split into two components:
1. **oelala** - Main application (this repo) - handles generation, UI, business logic
2. **oelala-storage** - Standalone storage service (separate repo) - handles files, sync, caching

## Current Status ✅

### oelala-storage Service
- **Port**: 7990 (HTTP), 7991 (gRPC), 7992 (Metrics)
- **Status**: Running as systemd service
- **Storage Path**: `/home/flip/oelala/media/`

### User-Scoped Storage
Each user gets their own bucket (directory) based on Supabase user ID:
```
/home/flip/oelala/media/
├── {user-uuid-1}/           ← User 1's bucket
│   ├── videos/
│   ├── images/
│   └── audio/
├── {user-uuid-2}/           ← User 2's bucket
│   ├── videos/
│   └── images/
└── ...
```

Buckets are automatically created on first upload.

## API Endpoints

### Backend (FastAPI) - User Media
```
GET  /user/media              → List user's media files
GET  /user/media/{type}/{file} → Get specific file
POST /user/media/upload       → Upload file to user storage
DELETE /user/media/{type}/{file} → Delete file
```

All endpoints require JWT authentication (`Authorization: Bearer <token>`).

### oelala-storage (Go) - S3-compatible
```
PUT    /{bucket}/{key}        → Upload file
GET    /{bucket}/{key}        → Download file
DELETE /{bucket}/{key}        → Delete file
GET    /{bucket}              → List bucket contents
HEAD   /{bucket}/{key}        → Get file metadata
```

## Access Control

| User Type | Can See | Notes |
|-----------|---------|-------|
| Guest | Nothing | Empty media list |
| Logged In | Own media only | User-scoped bucket |
| Admin | Own + ComfyUI output | Whitelist: `mark.op.mobiel@gmail.com` |

## Development vs Production

### Development (Current)
- ComfyUI outputs to `/home/flip/oelala/ComfyUI/output/`
- Admin can view ComfyUI output in MyMedia
- Other users only see their user storage

### Production (Current)
- ✅ **Auto-upload implemented**: All generated content automatically uploads to user storage
- Video generation (I2V, T2V, Sequential) → `users/{user_id}/videos/`
- Image generation (SD1.5, Wan2.2 T2I) → `users/{user_id}/images/`
- ComfyUI output remains as fallback if upload fails
- Job tracking associates user_id with each generation

### Future Enhancements
- Storage quota tracking and enforcement
- Automatic cleanup of old ComfyUI output files
- Retention policies (auto-delete after X days)

## Auto-Upload Feature

### How It Works

1. **Job Submission**: When user queues a generation job, backend stores user_id with the job
2. **Job Completion**: After ComfyUI finishes, the generated file is downloaded
3. **Auto-Upload**: File is automatically uploaded to user's storage bucket
   - Videos: `users/{user_id}/videos/{timestamp}_{filename}`
   - Images: `users/{user_id}/images/{timestamp}_{filename}`
4. **Metadata Cleanup**: Job tracking data is cleared after successful upload

### Endpoints Updated
- `/generate-video` - I2V generation
- `/generate-text-video` - T2V generation
- `/generate-wan22-comfyui` - Main video generation with LoRAs
- `/generate-sd15` - SD 1.5 image generation
- `/generate-wan22-t2i` - Wan2.2 text-to-image

### Error Handling
- Failed uploads are logged but don't block user flow
- Original file remains in ComfyUI output as fallback
- Users can still access content via `/files/{filename}` endpoint

## Known Issues / TODO

| Issue | Status | Priority |
|-------|--------|----------|
| Auto-upload after generation | ✅ Complete | ~~Critical~~ |
| Storage quota tracking | ⏳ Todo | High (#33) |
| Retention policies | ⏳ Todo | Medium (#71) |
| Signed URL generation | ⏳ Todo | Low |
- Long-term maintainability for one-man team

**Philosophy**: Local-first, self-hosted nodes that sync P2P.

```
┌─────────────────────────────────────────────────────────────┐
│              oelala-storage node (Go binary)                │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │  BadgerDB     │  │  HTTP/gRPC    │  │  Sync Engine  │   │
│  │  (metadata)   │  │  (S3-compat)  │  │  (P2P/WAN)    │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │  Local FS     │  │  LRU Cache    │  │  WebSocket    │   │
│  │  (blobs)      │  │  (hot files)  │  │  (events)     │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Node A         │  │  Node B         │  │  Node C         │
│  (Home Server)  │◀─▶│  (Office PC)    │◀─▶│  (Cloud VPS)    │
│  Linux          │  │  Windows        │  │  Linux          │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

#### Integration with Oelala

```
┌─────────────────────────────────────────────────────────────┐
│                    oelala (FastAPI)                         │
│  - Generation logic, UI, auth, billing                      │
│  - Talks to oelala-storage via S3-compatible API            │
│  - Local Redis/SQLite cache for metadata                    │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ S3 API (GET/PUT/DELETE)
                           │ WebSocket (realtime events)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  oelala-storage node                        │
│  - Handles file storage, dedup, sync                        │
│  - Can run on same machine or separate                      │
│  - Scales horizontally with more nodes                      │
└─────────────────────────────────────────────────────────────┘
```

#### Storage Node Features

| Feature | Description | Priority |
|---------|-------------|----------|
| Go binary | Single executable, ~10-15MB | Critical |
| Windows + Linux | Native builds, no Wine/WSL | Critical |
| S3-compatible API | Standard interface for tools | High |
| BadgerDB metadata | Embedded KV store, no setup | High |
| Content addressing | SHA-256 hash for deduplication | High |
| LRU caching | Hot files in memory | High |
| gRPC node-to-node | Efficient binary protocol for sync | Medium |
| File chunking | Large file support, resumable | Medium |
| Sync engine | Background P2P sync | Medium |
| Conflict resolution | Last-write-wins or versioning | Medium |
| Encryption | Optional at-rest (ChaCha20-Poly1305) | Low |
| Compression | Optional LZ4 for bandwidth savings | Low |

#### Node Types

| Type | Description | Use Case |
|------|-------------|----------|
| **Primary** | Main node, write-enabled | Production server |
| **Replica** | Mirror, can be promoted | Failover/backup |
| **Edge** | Local cache, partial sync | User's desktop |
| **Archive** | Cold storage, async sync | Long-term backup |

#### Configuration Example

```yaml
# oelala-storage.yaml
node:
  id: "node_abc123"
  name: "Home Server"
  type: primary

storage:
  path: "/data/oelala"
  max_size_gb: 500
  cache_size_mb: 2048

api:
  http_port: 7999
  grpc_port: 7998
  enable_tls: true

sync:
  peers:
    - url: "https://node-b.example.com:7999"
      type: replica
    - url: "https://archive.example.com:7999"
      type: archive
  strategy: realtime  # realtime, scheduled, manual
  interval_minutes: 15

security:
  encryption_at_rest: true
  auth_tokens:
    - name: "oelala-main"
      token: "${OELALA_STORAGE_TOKEN}"
      permissions: ["read", "write", "delete"]
```

### Phase 4: Cloud Integration (Future)

Once local distributed storage is stable, add cloud backends:

```
┌─────────────────────────────────────────────────────────────┐
│                     Storage Abstraction Layer                │
├─────────────────────────────────────────────────────────────┤
│  Oelala Nodes  │  S3/MinIO  │  GCS  │  Azure Blob  │  IPFS │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   CDN Layer       │
                    │  (CloudFlare/etc) │
                    └───────────────────┘
```

**Cloud Features** (Phase 4+):
- S3-compatible API for existing tools
- CDN integration for delivery
- Geographic distribution
- Tiered storage (hot/warm/cold)

## File Naming Convention

### Current Format
```
{YYYYMMDD}_{HHMMSS}_{type}_{uuid}.{ext}
Example: 20251128_195244_text_video.mp4
```

### Proposed Format
```
{user_id}/{project_id}/{timestamp}_{type}_{short_hash}.{ext}
Example: usr_abc123/prj_xyz789/20260103_213045_t2v_f8a2c1.mp4
```

## API Endpoints

### Current
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload file to uploads/ |
| `/files/{filename}` | GET | Download from generated/ |
| `/comfyui/output/{filename}` | GET | Download from ComfyUI/output/ |
| `/media` | GET | List all media (both locations) |

### Proposed (Unified)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v2/media/upload` | POST | Upload file |
| `/api/v2/media/{media_id}` | GET | Get media by ID |
| `/api/v2/media/{media_id}` | DELETE | Delete media |
| `/api/v2/media/list` | GET | List user's media |
| `/api/v2/media/search` | GET | Search media by metadata |

## Migration Path

1. **Step 1**: Create unified `/media/` directory structure
2. **Step 2**: Symlink ComfyUI directories to unified location
3. **Step 3**: Update backend to use unified paths
4. **Step 4**: Update frontend to use new API
5. **Step 5**: Migrate existing files
6. **Step 6**: Remove old directory references

## Security Considerations

- [ ] File type validation (magic bytes, not just extension)
- [ ] Max file size limits per tier
- [ ] Virus/malware scanning for uploads
- [ ] Signed URLs for temporary access
- [ ] Rate limiting on uploads
- [ ] Content hash deduplication

## Related Documents

- [ROADMAP.md](./ROADMAP.md) - Full product roadmap
- [ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture
- [PROJECT_PLAN.md](./PROJECT_PLAN.md) - Development plan
