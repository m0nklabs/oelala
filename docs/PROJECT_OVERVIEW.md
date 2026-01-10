# Oelala Project Overview

> **Last Updated**: January 2026
> **Version**: 0.4.x (Alpha)

---

## Vision

**Oelala** is an AI-powered media creation platform that enables creators to produce professional-quality image and video content using state-of-the-art generative AI models. The platform provides a unified, credits-based interface with age-gated mature content support.

---

## Current Status

### ✅ Core Features (Implemented)

| Category | Features |
|----------|----------|
| **Image Generation** | Text-to-Image (Flux, SDXL), Image Upscaling (RealESRGAN) |
| **Video Generation** | Image-to-Video, Text-to-Video (Wan2.2 14B), Video-to-Video |
| **Video Processing** | Upscaling, Frame Interpolation (RIFE), Reframing |
| **Audio Pipeline** | Text-to-Speech (ChatterBox), Voice Cloning (F5-TTS), Lip Sync |
| **User System** | Supabase Auth (Google/GitHub), Credits, Stripe payments |
| **Gallery** | Publish, Like, View counts, SFW/NSFW filtering |
| **Infrastructure** | Multi-GPU (DisTorch2), WebSocket progress, systemd services |

### 🔄 In Progress

- Auto-upload generated content to user storage
- Storage quota tracking and display
- Admin dashboard for user management

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OELALA PLATFORM                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│  │   React     │───▶│   FastAPI   │───▶│    ComfyUI      │ │
│  │  Frontend   │    │   Backend   │    │  (port 8188)    │ │
│  │ (port 5174) │    │ (port 7998) │    │                 │ │
│  └─────────────┘    └─────────────┘    └─────────────────┘ │
│         │                  │                    │          │
│         ▼                  ▼                    ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│  │  Supabase   │    │  oelala-    │    │   GPU Models    │ │
│  │  Auth/DB    │    │  storage    │    │  (GGUF/LoRA)    │ │
│  └─────────────┘    │ (port 7990) │    └─────────────────┘ │
│                     └─────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | React 18, Vite 7, CSS |
| Backend | FastAPI, Python 3.10 |
| AI Engine | ComfyUI with DisTorch2 |
| Database | Supabase (PostgreSQL) |
| Auth | Supabase Auth (OAuth) |
| Payments | Stripe Checkout |
| Storage | oelala-storage (Go/Fiber) |
| GPU | NVIDIA RTX (CUDA 12.x) |

---

## Directory Structure

```
/home/flip/oelala/
├── src/
│   ├── backend/          # FastAPI application
│   └── frontend/         # React/Vite application
├── ComfyUI/              # AI generation engine
│   ├── models/           # GGUF, LoRA, VAE, CLIP
│   └── output/           # Generated media
├── workflows/            # ComfyUI workflow templates
├── docs/                 # Documentation
└── tests/                # Test suites
```

---

## Hardware Configuration

| GPU | VRAM | CUDA Device | Role |
|-----|------|-------------|------|
| RTX 5060 Ti | 16GB | cuda:1 | Primary generation |
| RTX 3060 | 12GB | cuda:0 | Secondary/overflow |
| **Total** | **28GB** | | DisTorch2 distribution |

### Multi-GPU Setup (DisTorch2)
```
cuda:0,12gb;cuda:1,16gb
```

---

## Services

| Service | Port | Type | Command |
|---------|------|------|---------|
| Frontend (dev) | 5174 | Vite | npm run dev |
| Backend API | 7998 | systemd | oelala-api.service |
| ComfyUI | 8188 | systemd | comfyui.service |
| Storage | 7990 | systemd | oelala-storage.service |

---

## Development

### Quick Start
```bash
# Start dev servers
cd /home/flip/oelala/src/frontend && npm run dev
# Backend runs via systemd: sudo systemctl restart oelala-api

# Run tests
pytest tests/ -v

# Lint
ruff check src/
```

### Environment
- **Python venv**: /home/flip/venvs/gpu
- **Node**: 18+
- **CUDA**: 12.1+

---

## Documentation

| Document | Description |
|----------|-------------|
| [README.md](../README.md) | Main project readme |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture |
| [ROADMAP.md](ROADMAP.md) | Product roadmap |
| [TODO_LIST.md](TODO_LIST.md) | Development tasks |
| [CREDITS_SETUP.md](CREDITS_SETUP.md) | Credit system setup |
| [COMFYUI_INVENTORY.md](COMFYUI_INVENTORY.md) | Model inventory |
| [MULTI_GPU_SETUP.md](MULTI_GPU_SETUP.md) | GPU configuration |

---

## Links

- **Repository**: [github.com/m0nklabs/oelala](https://github.com/m0nklabs/oelala)
- **Issues**: [GitHub Issues](https://github.com/m0nklabs/oelala/issues)

---

*Maintained by m0nk111*
