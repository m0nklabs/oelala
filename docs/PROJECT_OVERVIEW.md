# Oelala Project Overview

> **Last Updated**: 2026-03-06
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
| **Storage** | Full migration to oelala-storage completed, storage proxy routes, signed/public URL support, admin node visibility |
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
│  │  Supabase   │    │ oelala-      │  │RunPod  │  │ Cloudflare  │  │
│  │ Auth / DB   │    │ storage      │  │Cloud   │  │ tunnels/DNS │  │
│  └─────────────┘    │ coordinator  │  │ Max    │  └─────────────┘  │
│                     │ :7990        │  └────────┘                    │
│                     └──────┬───────┘                                │
│                            │                                        │
│                    ┌───────▼────────┐                               │
│                    │ storage-node-01 │                               │
│                    │      :7993      │                               │
│                    └───────┬────────┘                               │
│                            │                                        │
│                    ┌───────▼────────┐                               │
│                    │  storage-node-02│                              │
│                    │  remote tunnel  │                              │
│                    └─────────────────┘                              │
└────────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | React, Vite, tool-based dashboard UI |
| Backend | FastAPI, Python, Supabase integration |
| AI Engine | ComfyUI, DisTorch2, Wan 2.2, Flux, SDXL |
| Cloud GPU | RunPod serverless endpoint + custom worker image |
| Storage | oelala-storage (Go/Fiber), signed URLs, dedup, GC |
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
| Storage coordinator | 7990 | systemd | `oelala-storage.service` |
| Storage node 01 | 7993 | systemd | `oelala-node-01.service` |

---

## Storage Model

- Local permanent media directories are no longer the source of truth.
- Generated/uploaded content is pushed into **oelala-storage** buckets and served via storage routes or Cloudflare-facing storage hostnames.
- Temporary backend processing now uses `/tmp/oelala_uploads` and `/tmp/oelala_generated`, then unlinks files after successful upload.
- Storage nodes currently exposed in docs and config:
	- `storage-main.oelala.xyz` → coordinator / primary node
	- `storage-node-01.oelala.xyz` → additional local node
	- `storage2.oelala.xyz` → remote node 2

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
| [STORAGE_MIGRATION_PLAN.md](STORAGE_MIGRATION_PLAN.md) | Final state of the storage migration |
| [CLOUDFLARE_SETUP.md](CLOUDFLARE_SETUP.md) | Tunnel, CORS, and cache behavior |
| [FACE_SYSTEM.md](FACE_SYSTEM.md) | I2I/face processing architecture |
| [COMFYUI_INVENTORY.md](COMFYUI_INVENTORY.md) | Installed models and node inventory |
| [DISTORCH2_MULTI_GPU_SETTINGS.md](DISTORCH2_MULTI_GPU_SETTINGS.md) | Multi-GPU tuning guide |
| [TODO_LIST.md](TODO_LIST.md) | Active implementation backlog |

---

## Links

- **Repository**: [github.com/m0nklabs/oelala](https://github.com/m0nklabs/oelala)
- **Storage Service**: [github.com/m0nklabs/oelala-storage](https://github.com/m0nklabs/oelala-storage)

---

*Maintained as an actively developed alpha platform.*
