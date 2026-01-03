# Oelala Architecture Documentation

## System Overview

Oelala is een AI video generation platform met een modern dashboard UI. Het platform integreert met ComfyUI voor video generatie via Wan2.2 workflows.

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Browser                            │
│                    http://localhost:7998                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend (React + Vite)                      │
│                    Port: 5174 (dev) / 7998 (prod via backend)   │
│                    Dashboard UI with sidebar navigation         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI/Uvicorn)                    │
│                    Port: 7998                                   │
│                    Serves static frontend + API                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ComfyUI (Workflow Engine)                    │
│                    Port: 8188                                   │
│                    DisTorch2 dual-pass workflows                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Service Configuration

### Frontend Service
**File**: \`~/.config/systemd/user/oelala-frontend.service\`

```ini
[Unit]
Description=Oelala Frontend Dev Server (Vite)
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/flip/oelala/src/frontend
ExecStart=/usr/bin/npm run dev -- --host 0.0.0.0 --port 5174
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

**Commands:**
```bash
systemctl --user start oelala-frontend
systemctl --user stop oelala-frontend
systemctl --user status oelala-frontend
journalctl --user -u oelala-frontend -f
```

### Backend Service
**File**: \`~/.config/systemd/user/oelala-backend.service\`

```ini
[Unit]
Description=Oelala Backend API (FastAPI/Uvicorn)
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/flip/oelala/src/backend
Environment="PATH=/home/flip/venvs/gpu/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/flip/venvs/gpu/bin/uvicorn app:app --host 0.0.0.0 --port 7998 --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

---

## Port Inventory

| Service         | Port | Protocol | Description                    |
|-----------------|------|----------|--------------------------------|
| Frontend (Vite) | 5174 | HTTP     | React dev server               |
| Backend (API)   | 7998 | HTTP     | FastAPI REST endpoints         |
| ComfyUI         | 8188 | HTTP/WS  | Workflow engine + WebSocket    |

---

## Backend Architecture

### Main Components

**File**: \`src/backend/app.py\`
- FastAPI application with REST endpoints
- Static file serving for generated videos
- CORS middleware for frontend access

**File**: \`src/backend/comfyui_client.py\`
- ComfyUI API client
- Workflow templating (DisTorch2 workflow)
- WebSocket progress monitoring

### Key Endpoints

| Endpoint                     | Method | Description                        |
|------------------------------|--------|------------------------------------|
| `/health`                    | GET    | Health check (ComfyUI + legacy)    |
| `/loras`                     | GET    | List available LoRA models         |
| `/unet-models`               | GET    | List GGUF unet models with pairs   |
| `/extract-metadata`          | POST   | Extract prompt from PNG metadata   |
| `/wan22/image-to-video`      | POST   | Generate video via ComfyUI         |
| `/list-comfyui-media`        | GET    | List output media with metadata    |
| `/api/presets`               | GET    | List available workflow presets    |
| `/comfyui-output/{filename}` | GET    | Serve generated videos/images      |

### LoRA Endpoint Response

```json
{
  "loras": [...],
  "high_noise": [...],
  "low_noise": [...],
  "general": [...],
  "by_category": { "subfolder": [...] }
}
```

### Unet Models Endpoint Response

```json
{
  "models": [...],
  "high_noise": [...],
  "low_noise": [...],
  "pairs": [{ "name": "...", "high": {...}, "low": {...} }]
}
```

### Metadata Extraction

The `/extract-metadata` endpoint extracts prompts from uploaded images:
- **oelala_params**: Our custom format with `original_t2i_prompt` preservation
- **ComfyUI workflow**: CLIPTextEncode nodes, WanVideo positive_prompt
- **A1111 format**: Parameters text block parsing

The `/list-comfyui-media` endpoint returns comprehensive metadata when `include_metadata=true`:
- **Prompt extraction**: Positive/negative prompts from workflow JSON
- **Generation params**: Steps, CFG, seed
- **Sampler info**: Sampler name, scheduler
- **Resolution**: Width × height from EmptyLatentImage
- **LoRAs**: Array of `{name, strength}` from LoraLoader nodes
- **Model**: Checkpoint/GGUF model name
- **Heuristic detection**: Negative prompts identified by keywords (worst, bad, ugly, etc.)

---

## ComfyUI Workflow: DisTorch2 I2V

De primaire workflow voor image-to-video generatie gebruikt **DisTorch2 dual-pass sampling**.

### Workflow Nodes

| Node | Type                              | Purpose                           |
|------|-----------------------------------|-----------------------------------|
| 1    | UnetLoaderGGUFAdvancedDisTorch2   | Load high noise Q6_K model        |
| 2    | UnetLoaderGGUFAdvancedDisTorch2   | Load low noise Q6_K model         |
| 3    | VAELoaderDisTorch2MultiGPU        | Load VAE decoder                  |
| 4    | CLIPLoaderDisTorch2MultiGPU       | Load T5 text encoder (CONVERTED)  |
| 5-6  | ModelSamplingSD3                  | Configure shift=5 for both models |
| 7-8  | SageAttention                     | Apply SageAttn optimization       |
| 9-10 | CLIPTextEncode                    | Encode positive/negative prompts  |
| 11   | LoadImage                         | Load input image                  |
| 12   | WanImageToVideo                   | I2V conditioning                  |
| 13   | KSamplerAdvanced                  | High noise pass (steps 0-3)       |
| 14   | KSamplerAdvanced                  | Low noise pass (steps 3+)         |
| 15   | VAEDecode                         | Decode latents to frames          |
| 16   | VHS_VideoCombine                  | Combine frames to video           |
| 17   | LoraLoaderModelOnly               | Optional LoRA for high noise      |
| 18   | LoraLoaderModelOnly               | Optional LoRA for low noise       |
| 19   | AspectRatioResolution_Warper      | Calculate width/height            |

### Memory Allocation

```
expert_mode_allocations: "cuda:0,0.25gb;cuda:1,8gb;cpu,*"
```

- **GPU 0**: 0.25 GB (control overhead)
- **GPU 1**: 8 GB (primary compute)
- **CPU**: Remaining weights (offload)

### Model Files

```
ComfyUI/models/
├── diffusion_models/
│   ├── wan2.2_i2v_high_noise_14B_Q6_K.gguf
│   └── wan2.2_i2v_low_noise_14B_Q6_K.gguf
├── clip/
│   └── umt5-xxl-enc-bf16-uncensored-CONVERTED.safetensors
├── vae/
│   └── wan_2.2_vae.safetensors
└── loras/
    ├── wan 2.2/           # Wan2.2 specific LoRAs
    └── ...                # 53+ total LoRA files
```

---

## Frontend Architecture

### Tech Stack
- **React 18** with hooks
- **Vite** dev server with HMR
- **CSS Variables** for theming
- **Lucide React** icons

### Dashboard Navigation (Sidebar)

The dashboard uses a collapsible sidebar with grouped tools:

```
┌─────────────────────────────────┐
│ 🎬 Oelala                      │
├─────────────────────────────────┤
│ ▶ Video Tools                   │
│   • Text to Video               │
│   • Image to Video      ✓       │
│   • Video to Video      soon    │
├─────────────────────────────────┤
│ ▶ Image Tools                   │
│   • Text to Image               │
│   • Image to Image      soon    │
│   • Reframe             soon    │
│   • Face Swap           soon    │
│   • Upscaler            soon    │
├─────────────────────────────────┤
│ ▶ My Media                      │
│   • All                         │
│   • Images                      │
│   • Videos                      │
│   • Favorites                   │
│   • Prompts             NEW     │
├─────────────────────────────────┤
│ ▶ Training                      │
│   • Train LoRA          soon    │
└─────────────────────────────────┘
```

### Key Components

**File**: `src/frontend/src/dashboard/Dashboard.jsx`
- Main dashboard container with sidebar + content layout
- Tool routing based on active tool ID
- Full-width mode for My Media tools

**File**: `src/frontend/src/dashboard/nav.js`
- Navigation structure definition
- Tool IDs and labels
- Status badges (new, soon, etc.)

**File**: `src/frontend/src/dashboard/tools/ImageToVideoTool.jsx`
- Main I2V generation interface
- Image upload (drag & drop, URL, creations)
- Parameter controls (resolution, duration, fps, aspect ratio)
- Prompt persistence via localStorage
- Metadata extraction from uploaded images (auto-fill prompts)
- Preset selector for workflow selection

**File**: `src/frontend/src/dashboard/tools/MyMediaTool.jsx`
- Media gallery with grid view (images, videos, all, prompts)
- Favorites system (localStorage)
- Sort/filter controls (date, name, size, favorites)
- Start image hiding (source images for videos)
- Multi-select with keyboard shortcuts
- **Prompt bubble (💬)** on thumbnails with prompt metadata
- **Prompt popup modal** showing:
  - Positive/negative prompts with copy button
  - Generation settings (steps, CFG, seed, sampler, scheduler)
  - LoRA models used with strength percentages
  - Model/checkpoint name
  - Resolution and video duration
- Dedicated **Prompts list view** for browsing generation history

**File**: `src/frontend/src/components/PresetSelector.jsx`
- Workflow preset selection dropdown
- API-driven preset list from `/api/presets`

### Model Selection UI

**Unet Models:**
- Collapsible panel in Model Selection
- Model Pair dropdown (recommended) - auto-selects matching high/low
- Advanced: separate high/low noise selectors

**LoRA Models:**
- Collapsible panel in Sampling Settings
- Category-grouped dropdowns (by subfolder)
- Separate high/low noise selectors
- Strength slider (0 - 2.0, default 1.5)

### Prompt System

**Positive Prompt:**
- Persisted to `localStorage.getItem('oelala_last_prompt')`
- Auto-filled from uploaded image metadata

**Negative Prompt:**
- Collapsible section with default text
- Preserved in generated video metadata

---

## Development Commands

### Starting All Services

```bash
# Start all services
systemctl --user start oelala-frontend oelala-backend

# Check status
systemctl --user status oelala-frontend oelala-backend

# View logs
journalctl --user -u oelala-frontend -f
journalctl --user -u oelala-backend -f
```

### Building Frontend for Production

```bash
cd /home/flip/oelala/src/frontend
npm run build
```

### Testing Endpoints

```bash
curl http://localhost:7998/health
curl http://localhost:7998/loras | jq .
```

---

## Environment

### Python
\`/home/flip/venvs/gpu/\` (canonical GPU venv)

### Node.js
\`/usr/bin/node\` (v22.x)

---

## File Structure

```
/home/flip/oelala/
├── src/
│   ├── backend/
│   │   ├── app.py
│   │   ├── comfyui_client.py
│   │   └── generated/
│   └── frontend/
│       └── src/dashboard/tools/
├── docs/
│   ├── ARCHITECTURE.md
│   └── PROJECT_OVERVIEW.md
├── ComfyUI/
│   ├── models/
│   └── user/default/workflows/
└── README.md
```

---

*Last Updated: January 3, 2026 - Added Prompts section, extended metadata extraction, preset support*
