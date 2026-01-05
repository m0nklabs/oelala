# Media Storage Architecture

> **Last Updated**: 2026-01-03
> **Related Project**: [oelala-storage](https://github.com/m0nklabs/oelala-storage) (separate repo)

## Overview

Storage is split into two components:
1. **oelala** - Main application (this repo) - handles generation, UI, business logic
2. **oelala-storage** - Standalone storage service (separate repo) - handles files, sync, caching

## Current Storage Locations (Development)

| Directory | Purpose | Served Via | Persistence |
|-----------|---------|------------|-------------|
| `/home/flip/oelala/uploads/` | User uploads (images, videos, audio) | `/uploads/` endpoint | Session-based |
| `/home/flip/oelala/generated/` | Oelala backend generated files | `/files/` endpoint | Persistent |
| `/home/flip/oelala/ComfyUI/output/` | ComfyUI workflow outputs | `/comfyui/output/` endpoint | Persistent |
| `/home/flip/oelala/ComfyUI/input/` | ComfyUI workflow inputs | Internal use | Temporary |

### File Flow Diagram

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  User Upload    │────▶│  /uploads/       │────▶│  ComfyUI/input/ │
│  (Frontend)     │     │  (Oelala)        │     │  (Processing)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  My Media       │◀────│  /generated/     │◀────│  ComfyUI/output │
│  (Frontend)     │     │  (Oelala)        │     │  (Results)      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Known Issues

### Inconsistency Problem (Current State)

1. **Dual Output Locations**: Generated files end up in either:
   - `generated/` - When Oelala backend processes directly
   - `ComfyUI/output/` - When ComfyUI workflows generate output

2. **Developer Experience**: Testing in ComfyUI doesn't automatically show in Oelala "My Media"

3. **File Discovery**: Frontend needs to check multiple locations

### Temporary Solution (Development)

The backend currently serves both directories:
- `/files/{filename}` → `generated/`
- `/comfyui/output/{filename}` → `ComfyUI/output/`

## Proposed Architecture

### Phase 1: Unified Media Layer (Near-term)

```python
# Unified media directories
MEDIA_ROOT = Path("/home/flip/oelala/media")
MEDIA_UPLOADS = MEDIA_ROOT / "uploads"      # User uploads
MEDIA_GENERATED = MEDIA_ROOT / "generated"  # All generated content
MEDIA_TEMP = MEDIA_ROOT / "temp"            # Processing intermediates

# ComfyUI symlink integration
# ComfyUI/output -> ../media/generated (symlink)
# ComfyUI/input -> ../media/uploads (symlink)
```

**Benefits**:
- Single source of truth for all media
- ComfyUI and Oelala share the same directories
- Simplified file discovery in frontend

### Phase 2: User-Scoped Storage (Future)

```
/media/
├── public/                    # Shared/example content
├── users/
│   ├── {user_id}/
│   │   ├── uploads/           # User's uploaded files
│   │   ├── generated/         # User's generated content
│   │   ├── projects/          # Organized by project
│   │   └── metadata.json      # Usage stats, quotas
│   └── ...
└── temp/                      # Processing workspace
```

### Phase 3: oelala-storage (Separate Repository)

> **Repository**: [github.com/m0nklabs/oelala-storage](https://github.com/m0nklabs/oelala-storage)
> **Language**: Go
> **Platforms**: Windows, Linux (no macOS support planned)

**Why Go?**
- Single binary (~10-15MB), no runtime dependencies
- Excellent I/O performance and concurrency (goroutines)
- Native cross-compilation for Windows/Linux
- Battle-tested for storage systems (MinIO, rclone, Syncthing)
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
