# Media Storage Architecture

> **Last Updated**: 2026-01-03

## Current Storage Locations

### Development Environment

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

### Phase 3: Local-First Distributed Storage

**Philosophy**: Start local, scale to cloud later. Each node is a full participant.

```
┌─────────────────────────────────────────────────────────────┐
│                    Oelala Storage Node                       │
│                   (Windows / Linux)                          │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │  Local FS     │  │  Node API     │  │  Sync Engine  │   │
│  │  (SQLite DB)  │  │  (REST/gRPC)  │  │  (P2P/WAN)    │   │
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

#### Storage Node Features

| Feature | Description | Priority |
|---------|-------------|----------|
| Cross-platform | Python-based, works on Windows/Linux/Mac | High |
| Local-first | Full functionality without internet | High |
| SQLite metadata | Lightweight, embedded database | High |
| REST API | Standard HTTP API for node communication | High |
| File chunking | Large file support with resumable transfers | Medium |
| Content addressing | SHA-256 hash for deduplication | Medium |
| Sync engine | Background sync between nodes | Medium |
| Conflict resolution | Last-write-wins or manual merge | Medium |
| Encryption | Optional at-rest and in-transit encryption | Low |
| Compression | Optional LZ4/ZSTD compression | Low |

#### Node Types

| Type | Description | Use Case |
|------|-------------|----------|
| **Primary** | Main production node, always online | Server |
| **Replica** | Mirror of primary, read-only or failover | Backup |
| **Edge** | Local cache, partial sync | Desktop/Laptop |
| **Archive** | Cold storage, infrequent access | Long-term backup |

#### Sync Strategies

```python
# Example: Node configuration
node_config = {
    "node_id": "node_abc123",
    "node_type": "primary",
    "storage_path": "/home/flip/oelala/media",
    "max_storage_gb": 500,
    "sync_peers": [
        {"url": "http://192.168.1.100:7999", "type": "replica"},
        {"url": "https://vps.example.com:7999", "type": "archive"}
    ],
    "sync_strategy": "realtime",  # realtime, scheduled, manual
    "sync_interval_minutes": 15,
    "encryption_enabled": True,
    "compression": "lz4"
}
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
