# Oelala Architecture

> Last Updated: 2026-07-14

## Overview

Oelala is a hybrid AI media platform, not just a single video-generation app. The current codebase combines product UI, orchestration, storage, billing, community, admin, and AI-assist flows behind one surface.

At runtime it is composed of:

1. **Frontend product surface**: React/Vite dashboard organized by tool families
2. **Backend control plane**: FastAPI APIs for auth, credits, generation, media, gallery, moderation, and admin
3. **Execution engines**: local ComfyUI plus RunPod Cloud Max for burst workloads
4. **Storage layer**: MinIO (S3-compatible object storage) for buckets, retention, presigned URLs, and media persistence
5. **Platform dependencies**: Supabase, Stripe, Cloudflare, and selected LLM/media services

## Capability Map

### Creative Tool Families

| Family | Current Capabilities |
|--------|----------------------|
| Video | Image-to-Video, Text-to-Video, Text-to-Image-to-Video, Video-to-Video, Speech-to-Video |
| Image | Text-to-Image, Image-to-Image, Inpaint, Reframe, Face Swap, Upscale |
| Prompt / Analysis | Prompt Generator, Image-to-Text, Video-to-Text, image analysis, analyze-and-generate |
| Audio | Audio generation, Voice Cloning, Lip Sync |
| Advanced | Pipeline tool, Post-Processing, LoRA Browser, LoRA Training |
| Community | Gallery publishing, likes, moderation/reporting |
| Account | Profile, API keys, credits, storage quota |
| Admin | Admin panel, analytics, moderation, storage nodes, system tooling |

## Runtime Topology

```text
User Browser
    │
    ▼
Frontend (React/Vite, :5174)
    │
    ▼
Backend API (FastAPI, :7998)
    ├─ Auth / profiles / credits / API keys
    ├─ Gallery / moderation / admin APIs
    ├─ Media library / workflow metadata / storage proxy
    ├─ Local ComfyUI orchestration (:8188)
    ├─ RunPod Cloud Max orchestration
    ├─ LLM-assisted prompt / caption / analysis flows
    └─ WebSocket / queue / progress plumbing
          │
          ├─ Local execution → ComfyUI
          ├─ Cloud execution → RunPod worker
          ├─ Media persistence → MinIO (S3 API)
          ├─ Auth/data → Supabase
          └─ Payments/credits → Stripe

MinIO Object Storage
    ├─ S3 API (:9000)
    └─ Console (:9001)
```

## Frontend Layer

The frontend is a dashboard application organized around tool groups, account surfaces, and admin operations.

### Main Responsibilities

- tool configuration and submission
- authenticated media browsing across images, video, audio, and prompts
- queue/progress display
- gallery/community surfaces
- profile, API key, and credit flows
- admin visibility into moderation, analytics, storage nodes, and system state

### Important Frontend Conventions

- use `apiFetch()` for backend API calls
- avoid raw `fetch()` for normal backend communication
- media can be served via backend-relative routes, signed URLs, or storage-facing URLs
- creations picking is implemented as inline panels, not blocking modal overlays

## Backend Layer

The backend is the product control plane. Policy, orchestration, and user-facing integration logic live here.

### Major Subsystems

| Area | Main Modules / Routes |
|------|------------------------|
| Auth / profile | `auth.py`, `profile_api.py`, `/user/profile`, `/api/profile/*` |
| Credits / billing | `credits.py`, `credits_api.py`, `/api/credits/*`, `/api/stripe/*` |
| Gallery / community | `gallery_api.py`, `/api/gallery/*` |
| Moderation | `moderation_api.py`, `/api/report/*`, `/api/admin/moderation/*` |
| API keys | `api_keys_management.py`, `/api/keys/*`, `/api/v1/*` |
| Admin | `admin_api.py`, `/api/admin/*`, `/api/admin/metrics` |
| Storage integration | `storage_client.py`, `/storage/*` |
| LoRA surfaces | `lora_api.py`, `lora_scanner.py`, `/api/loras/*`, `/train-lora` |
| Face systems | `face_service.py`, `face_train_service.py`, `/face-swap*`, `/api/face-train*`, `/api/face-profiles*` |
| Generation core | `comfyui_client.py`, `runpod_client.py`, `workflow_loader.py`, `job_queue.py` |
| LLM/media intelligence | `guardian_client.py`, `llm_queue.py`, captioning, prompt generation, analysis endpoints |

### Important Backend Responsibilities

- Supabase-backed authentication and user state
- credit charging and entitlement checks
- tool endpoint validation and workflow assembly
- ComfyUI prompt submission, polling, queue inspection, and cancellation
- RunPod Cloud Max submission, queue-state persistence, timeout handling, and recovery behavior
- storage uploads via MinIO S3 API, presigned URLs, workflow metadata lookup, and unified media APIs
- admin and reporting endpoints for users, storage, moderation, and platform operations

### Design Rules

- business and entitlement logic belong in the backend, not storage
- storage access is backend-driven and application-authorized
- generation state should report honest queue/running/failed outcomes
- legacy compatibility endpoints may still exist, but product behavior should be described through the modular APIs above

## Execution Layer

Generation is split across local and cloud execution paths.

### Local Execution: ComfyUI

Current local characteristics:

- Wan 2.2 workflows for video generation
- Flux, SDXL, SD1.5, and related image-generation paths
- dedicated endpoints for I2I, T2I, T2V, V2V, upscale, interpolation, reframe, inpaint, and face workflows
- DisTorch2 multi-GPU distribution for large models
- local outputs land in `ComfyUI/output/` or temporary processing paths before storage upload

### Cloud Execution: RunPod Cloud Max

Current cloud characteristics:

- asynchronous cloud generation for Wan 2.2 workloads
- persisted cloud job state and queue age tracking
- worker-side LoRA download support
- timeout handling when workers never provision
- output persistence back into the same media/storage surface exposed to users

### LLM-Assisted Media Flows

Oelala also has non-generation AI flows layered into the backend:

- prompt generation
- motion prompt generation
- image and video captioning
- image analysis and analyze-and-generate flows
- Guardian/Ollama status visibility for local language-model integrations

## GPU Layout

| GPU | VRAM | CUDA | Role |
|-----|------|------|------|
| RTX 3060 | 12GB | cuda:0 | donor / weight-heavy device |
| RTX 5060 Ti | 16GB | cuda:1 | primary compute device |

Preferred DisTorch2 allocation:

```text
cuda:0,10gb;cuda:1,15gb;cpu,*
```

## Storage Layer

Persistent media storage is delegated to **MinIO**, an S3-compatible object storage service.

### Storage Principles

- MinIO buckets are the source of truth
- backend-local paths are temporary processing locations only
- retention is defined by backend metadata and lifecycle rules
- presigned URLs provide time-limited access; policy still belongs to the backend

### MinIO Configuration

| Service | Port | Purpose |
|---------|------|---------|
| MinIO S3 API | 9000 | S3-compatible object storage API |
| MinIO Console | 9001 | Web-based admin console |

## Platform Dependencies

| Dependency | Role |
|------------|------|
| Supabase | auth, profile data, app state |
| Stripe | credits and payments |
| Cloudflare | tunnels, DNS, proxy/cache behavior |
| RunPod | burst cloud GPU execution |
| MinIO | S3-compatible object storage, presigned URLs, bucket lifecycle |
| Guardian / local LLM stack | prompt and analysis support paths |

## Delivery and Edge

Cloudflare is used for public routing and tunnel-based connectivity.

### Public Hostnames

| Hostname | Target |
|----------|--------|
| `oelala.xyz` | frontend |
| `api.oelala.xyz` | backend |
| `storage.oelala.xyz` | storage primary |
| `storage2.oelala.xyz` | remote storage node |

### Important Edge Rules

- `Vary: Origin` matters for CORS correctness behind Cloudflare
- explicit origins are required when credentials are involved
- per-node tunnels are preferred over centralizing node ingress through one host

## Runtime Services

| Service | Port | systemd unit |
|---------|------|--------------|
| frontend | 5174 | `oelala-frontend.service` |
| backend | 7998 | `oelala-backend.service` |
| ComfyUI | 8188 | `comfyui.service` |
| MinIO S3 API | 9000 | `minio.service` |
| MinIO Console | 9001 | `minio.service` |

## Representative Flows

### Local Generation

1. User configures a tool in the frontend.
2. Backend validates auth, credits, and tool-specific settings.
3. Backend assembles a workflow and submits it to ComfyUI.
4. Progress is polled or streamed back to the frontend.
5. Output is uploaded to storage and indexed in the user media surface.
6. Temporary local files are cleaned up.

### Cloud Generation

1. User submits a cloud-capable generation.
2. Backend creates a cloud job record and sends work to RunPod.
3. Backend tracks queue age and status transitions instead of faking running state.
4. Completed output is persisted into the same media/storage system as local jobs.
5. Stuck jobs age out into failure when workers never provision.

### Community / Account Flow

1. User authenticates via Supabase-backed flows.
2. Credits and permissions are checked before generation or publishing actions.
3. Generated media can move into user libraries, gallery surfaces, moderation flows, and API-backed access paths.
4. Admin tools observe usage, storage nodes, moderation items, and system state through dedicated APIs.

## Canonical Supporting Docs

- `PROJECT_OVERVIEW.md`
- `ROADMAP.md`
- `INFRASTRUCTURE.md`
- `STORAGE_MIGRATION_PLAN.md`
- `CLOUDFLARE_SETUP.md`
- `FACE_SYSTEM.md`
- `COMFYUI_INVENTORY.md`
- `DISTORCH2_MULTI_GPU_SETTINGS.md`
