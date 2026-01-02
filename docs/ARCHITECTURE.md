# Oelala Architecture Documentation

## System Overview

Oelala is een AI video generation platform dat bestaat uit drie hoofdcomponenten:

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Browser                            │
│                    http://ai-kvm2:5174                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend (React + Vite)                      │
│                    Port: 5174 (dev server)                      │
│                    Service: oelala-frontend                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI/Uvicorn)                    │
│                    Port: 7998                                   │
│                    Service: oelala-backend                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ComfyUI (Workflow Engine)                    │
│                    Port: 8188                                   │
│                    Service: comfyui                             │
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
| \`/health\`                    | GET    | Health check                       |
| \`/loras\`                     | GET    | List available LoRA models         |
| \`/generate-wan22-comfyui\`    | POST   | Generate video via ComfyUI         |
| \`/videos/{filename}\`         | GET    | Serve generated videos             |
| \`/list-videos\`               | GET    | List all generated videos          |

### LoRA Endpoint Response

```json
{
  "loras": [...],
  "high_noise": [...],
  "low_noise": [...],
  "general": [...]
}
```

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

### Key Components

**File**: \`src/frontend/src/dashboard/tools/ImageToVideoTool.jsx\`
- Main I2V generation interface
- Image upload (drag & drop, URL, creations)
- Parameter controls (resolution, duration, fps, aspect ratio)
- Advanced settings (steps, cfg, seed, LoRA)

### LoRA Selection UI
- Collapsible panel in Advanced Settings
- Separate dropdowns for high/low noise models
- Strength slider (0 - 2.0)
- "Active" badge when LoRA selected

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

*Last Updated: January 2, 2026 - Added LoRA support, DisTorch2 workflow, systemd services*
