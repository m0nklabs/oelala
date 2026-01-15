# Media Storage Architecture

> **Last Updated**: 2026-01-15
> **Related Project**: [oelala-storage](https://github.com/m0nklabs/oelala-storage) (separate repo)
> **Canonical Docs**: [oelala-storage/docs/VISION.md](../../oelala-storage/docs/VISION.md)

---

## 🎯 Key Concept: Separation of Concerns

**oelala-backend is the "brain", oelala-storage is the "dumb" storage.**

| Responsibility | oelala-backend | oelala-storage |
|---------------|----------------|----------------|
| User authentication | ✅ | ❌ |
| Access control (who sees what) | ✅ | ❌ |
| Retention policies | ✅ (sets `X-Expires-At`) | ❌ (just executes) |
| Tier/quota logic | ✅ | ❌ |
| Storing files | ❌ | ✅ |
| Deduplication | ❌ | ✅ |
| CDN/replication | ❌ | ✅ |

---

## Overview

Storage is split into two components:
1. **oelala** - Main application (this repo) - handles generation, UI, business logic
2. **oelala-storage** - Standalone storage service (separate repo) - handles files, sync, caching

### oelala-storage Architecture

oelala-storage is a **client/server/CDN** system:

```
┌────────────────────────────────────────────────────────────────┐
│  oelala-backend (this repo)                                    │
│                                                                │
│  • Determines who can see what                                 │
│  • Sets retention via X-Expires-At header                     │
│  • Manages user tiers/quotas                                  │
│                                                                │
│                        │                                       │
│                        ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  oelala-storage Coordinator (CDN entry point)            │ │
│  │                                                          │ │
│  │  • Routes requests to correct node                       │ │
│  │  • Manages replication                                   │ │
│  │  • Handles deduplication                                 │ │
│  │                                                          │ │
│  │               ┌─────────┬─────────┐                     │ │
│  │               ▼         ▼         ▼                     │ │
│  │            Node 1    Node 2    Node 3                   │ │
│  │            500GB     200GB     1TB                      │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

## Current Status ✅

### oelala-storage Service
- **Mode**: Standalone (coordinator + node combined)
- **Port**: 7990 (HTTP), 7991 (gRPC), 7992 (Metrics)
- **Status**: Running as systemd service
- **Storage Path**: `/home/flip/oelala/media/`

### User-Scoped Storage
Each user gets their own bucket (directory) based on Supabase user ID:
```
/home/flip/oelala/media/
├── users/
│   ├── {user-uuid-1}/           ← User 1's bucket
│   │   ├── videos/
│   │   ├── images/
│   │   └── audio/
│   ├── {user-uuid-2}/           ← User 2's bucket
│   │   ├── videos/
│   │   └── images/
│   └── ...
├── public/                       ← Shared/published content
└── temp/                         ← Temporary processing files
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
- ✅ **Auto-upload implemented**: Generated content automatically uploads to user storage for synchronous endpoints
- **Synchronous endpoints with auto-upload**:
  - Image generation (SD1.5, Wan2.2 T2I) → `users/{user_id}/images/`
  - These endpoints wait for ComfyUI completion before returning
- **Async endpoints (queued, no auto-upload yet)**:
  - Video generation (I2V, T2V, Sequential) - queued but return immediately
  - Auto-upload not implemented for async workflows (files remain in ComfyUI output)
- ComfyUI output remains as fallback if upload fails
- Job tracking associates user_id with each generation

### Future Enhancements
- **Background auto-upload for async endpoints**: Implement background tasks to upload after async job completion
- Storage quota tracking and enforcement
- Automatic cleanup of old ComfyUI output files
- Retention policies (auto-delete after X days)

## Auto-Upload Feature

### How It Works

1. **Job Submission**: When user queues a generation job, backend stores user_id with the job
2. **Job Completion**: After ComfyUI finishes, the generated file is downloaded
3. **Auto-Upload**: File is automatically uploaded to user's storage bucket
   - Images: `users/{user_id}/images/{timestamp}_{filename}`
   - Videos: `users/{user_id}/videos/{timestamp}_{filename}` (synchronous endpoints only)
4. **Metadata Cleanup**: Job tracking data is cleared after successful upload

### Endpoints with Auto-Upload

**✅ Synchronous (auto-upload working)**
- `/generate-sd15` - SD 1.5 image generation
- `/generate-wan22-t2i` - Wan2.2 text-to-image

**⏳ Async (queued, no auto-upload)**
- `/generate-video` - I2V generation (returns immediately after queueing)
- `/generate-text-video` - T2V generation (returns immediately after queueing)
- `/generate-wan22-comfyui` - Main video generation with LoRAs (returns immediately after queueing)

> **Note**: Async endpoints queue jobs and return immediately. Files remain in ComfyUI output directory.
> Background auto-upload for async endpoints will be implemented in a future update.

### Error Handling
- Failed uploads are logged but don't block user flow
- Original file remains in ComfyUI output as fallback
- Users can still access content via `/files/{filename}` endpoint

## Known Issues / TODO

| Issue | Status | Priority |
|-------|--------|----------|
| Auto-upload after generation | ⚠️ Sync only | Critical (#15) |
| Background auto-upload for async | ⏳ Todo | Critical (#15) |
| Storage quota tracking | ⏳ Todo | High (#33) |
| Retention policies via `X-Expires-At` | ⏳ Todo | Medium (#71) |
| ✅ Signed URL generation | ✅ Done | - |
| User bucket support in storage | ⏳ Todo | Critical (oelala-storage) |

---

## 📅 Retention & Cleanup

### oelala-backend Responsibility

oelala-backend determines retention, NOT storage:

```python
# When uploading to storage
headers = {
    "X-User-ID": user_id,
    "X-Expires-At": (datetime.now() + timedelta(days=180)).isoformat(),  # 6 months EU
}
storage_client.put(bucket, key, data, headers=headers)
```

### Retention Rules (Backend enforces)

| Content Type | Retention | Reason |
|--------------|-----------|--------|
| Generated media | 6 months | EU GDPR minimum |
| User uploads | 6 months | EU GDPR minimum |
| Deleted account media | 2 years + 6 months | Legal requirements |
| Published content | Until unpublished | User controls |

### What Storage Does

Storage just executes what backend tells it:
1. Receives `X-Expires-At` header on upload
2. Stores expiration in file metadata
3. GC job deletes expired files periodically
4. Backend can delete files early if needed

---

## 🔗 oelala-storage Architecture

See [oelala-storage/docs/VISION.md](../../oelala-storage/docs/VISION.md) for the canonical architecture.

**Current Mode**: Standalone (single node, coordinator + storage combined)

**Future Mode**: Client/Server/CDN with multiple storage nodes:

```
oelala-backend  ───────────────────→  Coordinator (CDN)
     │                                      │
     │  "Store this with 6 month expiry"    │
     │  "User X can access file Y"           │
     │  "Delete all files for user Z"        │
                                            │
                                     ┌──────┼──────┐
                                     ▼      ▼      ▼
                                  Node 1  Node 2  Node 3
```

---

## 📁 File Naming Convention

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
