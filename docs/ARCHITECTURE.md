# Oelala Architecture - ComfyUI as Processing Middleware

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         OELALA PLATFORM                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐  │
│  │   FRONTEND  │────▶│     BACKEND      │────▶│      ComfyUI        │  │
│  │  (React/TS) │     │    (FastAPI)     │     │   (Middleware)      │  │
│  │  Port 5173  │◀────│    Port 7999     │◀────│    Port 8188        │  │
│  └─────────────┘     └──────────────────┘     └─────────────────────┘  │
│         │                    │                         │                │
│         ▼                    ▼                         ▼                │
│  User Interface      Workflow Config          Processing Pipeline      │
│  - Upload images     - Store workflows        - Load models (GGUF)     │
│  - Select presets    - Queue jobs             - Run inference          │
│  - View results      - Track status           - Encode video           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### Frontend (React/Vite)
- **Location**: `/home/flip/oelala/src/frontend/` (TBD)
- **Port**: 5173 (dev), 80/443 (prod via nginx)
- **Role**:
  - User-facing interface
  - Workflow configuration UI
  - Real-time status updates
  - Result gallery

### Backend (FastAPI)
- **Location**: `/home/flip/oelala/src/backend/` (TBD)
- **Port**: 7999
- **Role**:
  - Workflow management
  - Job queue management
  - ComfyUI API communication
  - User authentication
  - Result storage

### ComfyUI (Processing Middleware)
- **Location**: `/home/flip/oelala/ComfyUI/`
- **Port**: 8188
- **Role**:
  - Actual AI processing
  - Model loading (GGUF, safetensors)
  - Multi-GPU management
  - Video generation pipelines
  - WebSocket status updates

## Workflow Pipeline

```
User Request → Frontend → Backend → ComfyUI API → GPU Processing → Output
     │            │          │           │              │            │
     │            │          │           │              │            ▼
     │            │          │           │              │       Video File
     │            │          │           │              ▼            │
     │            │          │           │        Model Inference    │
     │            │          │           ▼              │            │
     │            │          │      Queue Prompt        │            │
     │            │          ▼           │              │            │
     │            │     Store Workflow   │              │            │
     │            ▼          │           │              │            │
     │       Show UI         │           │              │            │
     ▼           │           │           │              │            ▼
  Upload ────────┴───────────┴───────────┴──────────────┴───────▶ Result
```

## ComfyUI API Integration

### Key Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/prompt` | POST | Submit workflow for execution |
| `/queue` | GET | Get current queue status |
| `/history/{prompt_id}` | GET | Get execution result |
| `/view` | GET | View generated images/videos |
| `/system_stats` | GET | Get system/GPU status |
| `/interrupt` | POST | Cancel current execution |

### Backend → ComfyUI Communication
```python
import aiohttp

async def submit_workflow(workflow: dict) -> str:
    """Submit workflow to ComfyUI and return prompt_id."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8188/prompt",
            json={"prompt": workflow}
        ) as resp:
            result = await resp.json()
            return result["prompt_id"]

async def get_status(prompt_id: str) -> dict:
    """Get execution status from ComfyUI."""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"http://localhost:8188/history/{prompt_id}"
        ) as resp:
            return await resp.json()
```

## Tested Workflows (Production Ready)

### Wan2.2 I2V Q5 DisTorch2
- **File**: `wan22_i2v_14b_Q5_distorch2.json`
- **Models**: Q5_K_S (10GB per model)
- **Features**: SageAttention, CPU offload
- **Settings**: 6 steps, CFG 5.0

### Hardware Limits (RTX 5060 Ti 16GB + RTX 3060 12GB)

| Parameter | Min | Recommended | Max (tested) |
|-----------|-----|-------------|--------------|
| Resolution | 256×256 | 480×480 | 720×720 |
| Frames | 17 | 41 | 81 |
| Model Size | Q4_K_M (9GB) | Q5_K_S (10GB) | Q5_K_S (10GB) |
| Steps | 4 | 6 | 20 |

### Pending Hardware Tests
- [ ] 720×480 (16:9 aspect ratio)
- [ ] 81 frames at 480×480
- [ ] 121 frames at lower resolution
- [ ] Higher CFG values (6.0, 7.0)
- [ ] Different step counts (8, 10, 12)

## Workflow Configuration Schema

Workflows stored by backend should follow this schema:

```json
{
  "id": "uuid",
  "name": "workflow_name",
  "description": "User-friendly description",
  "category": "i2v|t2v|img2img|upscale",
  "comfyui_workflow": { /* full ComfyUI JSON */ },
  "parameters": {
    "resolution": {"width": 480, "height": 480},
    "frames": 41,
    "steps": 6,
    "cfg": 5.0,
    "model": "wan2.2_i2v_low_noise_14B_Q5_K_S.gguf"
  },
  "presets": [
    {"name": "Fast", "steps": 4, "frames": 25},
    {"name": "Quality", "steps": 8, "frames": 41},
    {"name": "Maximum", "steps": 12, "frames": 81}
  ],
  "hardware_requirements": {
    "min_vram_gb": 12,
    "recommended_vram_gb": 16,
    "cpu_offload_gb": 8
  }
}
```

## Directory Structure (Proposed)

```
/home/flip/oelala/
├── ComfyUI/                    # Processing middleware (existing)
│   ├── models/unet/            # GGUF models
│   ├── models/loras/           # LoRA files
│   ├── input/                  # Input images
│   ├── output/                 # Generated videos
│   └── user/default/workflows/ # ComfyUI workflows
├── src/
│   ├── backend/                # FastAPI backend (TBD)
│   │   ├── api/
│   │   ├── services/
│   │   │   └── comfyui.py      # ComfyUI integration
│   │   └── main.py
│   └── frontend/               # React frontend (TBD)
│       ├── components/
│       ├── services/
│       └── App.tsx
├── configs/
│   └── workflows/              # Validated workflow configs
│       ├── i2v_fast.json
│       ├── i2v_quality.json
│       └── i2v_maximum.json
└── docs/
    ├── ARCHITECTURE.md         # This file
    └── COMFYUI_INTEGRATION.md
```

## Next Steps

### Phase 1: Workflow Validation
1. Test all resolution/frame combinations
2. Document VRAM usage per configuration
3. Identify stable presets for frontend

### Phase 2: Backend Development
1. Create FastAPI service
2. Implement ComfyUI API wrapper
3. Add job queue management
4. Store workflow configurations

### Phase 3: Frontend Development
1. Build workflow selector UI
2. Add parameter sliders
3. Implement real-time progress
4. Create result gallery

---

**Last Updated**: December 28, 2025
**Status**: Architecture defined, ComfyUI working, Q5 workflow tested
