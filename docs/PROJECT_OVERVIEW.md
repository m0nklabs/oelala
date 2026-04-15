# Oelala Project Overview

> **Last Updated**: 2026-07-14
> **Version**: 0.11.x (Alpha)

---

## Vision

**Oelala** is a hybrid local-plus-cloud AI media platform for image, video, audio, and prompt workflows. It combines local multi-GPU ComfyUI execution, optional RunPod cloud bursting, user-scoped storage, credits, gallery publishing, and admin tooling behind a single product surface.

---

## Current Status

### ✅ Core Platform

| Category | Features |
|----------|----------|
| **Image** | Text-to-Image, Image-to-Image, Inpainting, Reframe, prompt generation, image captioning |
| **Video** | Image-to-Video, Text-to-Video, Video-to-Video, upscaling, frame interpolation |
| **Audio** | Text-to-Speech, voice cloning, lip sync, audio generation modes |
| **Face Workflows** | IP-Adapter FaceID, FaceDetailer, GFPGAN, face swap, face LoRA training queue integration |
| **User System** | Supabase auth, credits, Stripe, profile system, gallery publishing, likes, NSFW gating |
| **Storage** | MinIO-backed object storage (S3 API), storage proxy routes, presigned URL support, admin MinIO health dashboard |
| **Cloud Compute** | RunPod Cloud Max integration for Wan 2.2 T2V/I2V workloads with queue polling and persistence |
| **Operations** | systemd services, Cloudflare tunnels, WebSocket progress, admin storage cluster dashboard |

### 🔄 Active Work

- Cloud worker reliability and queue recovery around RunPod provisioning
- Storage cluster rollout beyond the primary node and remote node 2
- Retention/quota surfacing and additional media management polish
- Cleanup of remaining legacy local-path assumptions and fallback code

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                            OELALA PLATFORM                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌────────────────────────┐  │
│  │   React     │───▶│   FastAPI   │───▶│  ComfyUI + Workflows   │  │
│  │  Frontend   │    │   Backend   │    │        :8188           │  │
│  │    :5174    │    │    :7998    │    └────────────────────────┘  │
│  └─────────────┘    └─────────────┘                 │              │
│         │                  │                        │              │
│         │                  ├───────────────┐        │              │
│         ▼                  ▼               ▼        ▼              │
│  ┌─────────────┐    ┌──────────────┐  ┌────────┐  ┌─────────────┐  │
│  │  Supabase   │    │    MinIO    │  │RunPod  │  │ Cloudflare  │  │
│  │ Auth / DB   │    │ S3-compat  │  │Cloud   │  │ tunnels/DNS │  │
│  └─────────────┘    │ storage    │  │ Max    │  └─────────────┘  │
│                     │ :9000      │  └────────┘                    │
│                     └────────────┘                                │
└────────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | React, Vite, tool-based dashboard UI |
| Backend | FastAPI, Python, Supabase integration |
| AI Engine | ComfyUI, DisTorch2, Wan 2.2, Flux, SDXL |
| Cloud GPU | RunPod serverless endpoint + custom worker image |
| Storage | MinIO (S3-compatible object storage), presigned URLs, bucket lifecycle |
| Auth | Supabase Auth (Google/GitHub OAuth + JWT) |
| Payments | Stripe credits |
| Edge/Delivery | Cloudflare tunnels, cache and CORS controls |

---

## Directory Structure

```
/home/flip/oelala/
├── src/
│   ├── backend/          # FastAPI application, auth, queue, storage proxy
│   └── frontend/         # React/Vite dashboard and tools
├── ComfyUI/              # Local generation engine and models
├── workflows/            # API-format ComfyUI workflows
├── docs/                 # Product, infra, GPU, and migration docs
├── tests/                # Unit/integration/GPU tests
├── deploy/               # RunPod and service deployment assets
└── changelog/            # Changelog fragments
```

---

## Hardware Configuration

| GPU | VRAM | CUDA Device | Role |
|-----|------|-------------|------|
| RTX 3060 | 12GB | cuda:0 | Donor GPU for model weights |
| RTX 5060 Ti | 16GB | cuda:1 | Primary compute GPU |
| **Total** | **28GB** | | Local multi-GPU generation budget |

### Multi-GPU Setup (DisTorch2)
```
cuda:0,10gb;cuda:1,15gb;cpu,*
```

This ordering is intentional: putting the 3060 first keeps the 5060 Ti freer for activations.

---

## Services

| Service | Port | Runtime | Notes |
|---------|------|---------|-------|
| Frontend | 5174 | systemd | `oelala-frontend.service` |
| Backend API | 7998 | systemd | `oelala-backend.service` |
| ComfyUI | 8188 | systemd | `comfyui.service` |
| MinIO S3 API | 9000 | systemd | `minio.service` |
| MinIO Console | 9001 | systemd | `minio.service` |

---

## Storage Model

- Local permanent media directories are no longer the source of truth.
- Generated/uploaded content is pushed into **MinIO** buckets (S3 API) and served via storage proxy routes or presigned URLs.
- Temporary backend processing now uses `/tmp/oelala_uploads` and `/tmp/oelala_generated`, then unlinks files after successful upload.
- Storage nodes currently exposed in docs and config:
	- MinIO S3 API at `:9000`
	- MinIO Console at `:9001`

---

## Development Notes

- Use `apiFetch()` for frontend backend calls; raw `fetch()` is considered legacy debt.
- Use the canonical GPU environment at `/home/flip/venvs/gpu`.
- Run services via systemd for production-like behavior; do not start backend or ComfyUI manually outside approved workflows.
- Root `README.md` is intentionally sparse and should stay that way.

---

## Key Documentation

| Document | Description |
|----------|-------------|
| [ROADMAP.md](ROADMAP.md) | Product and infrastructure direction |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture and runtime flow |
| [INFRASTRUCTURE.md](INFRASTRUCTURE.md) | Canonical infrastructure inventory |
| [STORAGE_MIGRATION_PLAN.md](STORAGE_MIGRATION_PLAN.md) | Final state of the storage migration |
| [CLOUDFLARE_SETUP.md](CLOUDFLARE_SETUP.md) | Tunnel, CORS, and cache behavior |
| [FACE_SYSTEM.md](FACE_SYSTEM.md) | I2I/face processing architecture |
| [COMFYUI_INVENTORY.md](COMFYUI_INVENTORY.md) | Installed models and node inventory |
| [DISTORCH2_MULTI_GPU_SETTINGS.md](DISTORCH2_MULTI_GPU_SETTINGS.md) | Multi-GPU tuning guide |
| [TODO_LIST.md](TODO_LIST.md) | Active implementation backlog |

---

## Links

- **Repository**: [github.com/m0nklabs/oelala](https://github.com/m0nklabs/oelala)
- **Storage Service**: MinIO (S3-compatible, local service on port 9000)

---

*Maintained as an actively developed alpha platform.*
