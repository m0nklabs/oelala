# Agent Context Transfer - oelala

> **Mind meld document** - Alle context voor de volgende agent sessie.
> Laatst bijgewerkt: 2026-01-04

---

## 🎯 Project Missie

**oelala** is een AI-powered content creation platform - video, audio, images genereren via een web interface met ComfyUI als backend.

### Core Value Proposition
- **Geen technische kennis nodig** - Simpele UI voor complexe AI workflows
- **Local-first** - Draait op eigen hardware, geen cloud lock-in
- **Multi-modal** - Video, audio, images, prompts in één platform
- **Async workflows** - Queue-based generatie, geen blocking UI

---

## 🏗️ Architectuur

```
┌─────────────────────────────────────────────────────────┐
│                  Frontend (React/Vite)                   │
│                     Port 5174                            │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Dashboard.jsx → Tools (Image/Video/Audio/etc)  │    │
│  │  nav.js → Menu structure & tool IDs             │    │
│  │  QueuePanel → Job monitoring                    │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP
                      ▼
┌─────────────────────────────────────────────────────────┐
│                Backend (FastAPI/Uvicorn)                 │
│                     Port 7998                            │
│  ┌─────────────────────────────────────────────────┐    │
│  │  app.py (4400+ lines) - All endpoints           │    │
│  │  /generate-* → Queue ComfyUI workflows          │    │
│  │  /queue/* → Job status & history                │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────┘
                      │ WebSocket + HTTP
                      ▼
┌─────────────────────────────────────────────────────────┐
│                   ComfyUI (Python)                       │
│                     Port 8188                            │
│  ┌─────────────────────────────────────────────────┐    │
│  │  3700+ nodes (Wan2.2, F5-TTS, LatentSync, etc)  │    │
│  │  GPU inference (NVIDIA 5090, 32GB VRAM)         │    │
│  │  Workflow execution via API                      │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              oelala-storage (Go)                         │
│                     Port 7990                            │
│  Content-addressed storage, P2P sync, quotas            │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Code Structuur

```
oelala/
├── src/
│   ├── frontend/           # React app (Vite)
│   │   └── src/
│   │       ├── dashboard/
│   │       │   ├── Dashboard.jsx    # Main router
│   │       │   ├── nav.js           # Menu & tool IDs
│   │       │   └── tools/           # Individual tool components
│   │       │       ├── ImageToVideoTool.jsx
│   │       │       ├── TextToImageTool.jsx
│   │       │       ├── AudioGenerationTool.jsx
│   │       │       ├── SpeechToVideoTool.jsx  # NEW
│   │       │       └── ...
│   │       └── config.js            # Backend URLs
│   └── backend/
│       └── app.py           # FastAPI (4400+ lines)
├── deploy/                  # Systemd services
│   ├── oelala-backend.service
│   ├── oelala-frontend.service
│   └── comfyui.service
├── ComfyUI/                 # Git submodule
│   ├── custom_nodes/        # Extensions
│   └── models/              # AI models (LoRAs, checkpoints)
├── docs/
│   ├── TODO_TOOLS.md        # Tool implementation status
│   ├── ROADMAP.md           # Product roadmap
│   └── MEDIA_STORAGE.md     # Storage architecture
└── CHANGELOG.md             # Per-day changes
```

---

## 🔧 Services & Ports

| Service | Port | Systemd Unit | Status |
|---------|------|--------------|--------|
| Frontend | 5174 | `oelala-frontend.service` | ✅ Running |
| Backend API | 7998 | `oelala-backend.service` | ✅ Running |
| ComfyUI | 8188 | `comfyui.service` | ✅ Running |
| Storage | 7990 | `oelala-storage.service` | ✅ Running |

### Service Commands
```bash
# Status
sudo systemctl status oelala-backend oelala-frontend comfyui

# Restart
sudo systemctl restart oelala-backend

# Logs
journalctl -u oelala-backend -f
```

---

## ✅ Wat is AF (per 2026-01-04)

### Video Tools
- [x] **Image to Video** - Wan2.2 480p/720p, LoRAs, camera motions
- [x] **Text to Video** - Direct T2V
- [x] **Text to Image to Video** - Combined T2I+I2V
- [x] **Video to Video** - Style transfer, VHS nodes
- [x] **Speech to Video** - TTS + Lip Sync combined (NEW)

### Image Tools
- [x] **Text to Image** - SDXL, LoRAs, models
- [x] **Image to Image** - Denoise, style transfer
- [x] **Upscaler** - RealESRGAN 2x/4x
- [x] **Reframe** - Aspect ratio + outpainting
- [x] **Face Swap** - ReActor node

### Prompt Tools
- [x] **Prompt Generator** - LLM-powered (Ollama)
- [x] **Image to Text** - Florence2, SmolVLM
- [x] **Video to Text** - Frame-by-frame captioning

### Audio Tools
- [x] **Audio Generation** - TTS, Music (MMAudio), SFX
- [x] **Voice Cloning** - F5-TTS (F5v1, E2)
- [x] **Lip Sync** - LatentSyncNode
- [x] **TTS Gender Selection** - Female/Male voice groups (NEW)

### Infrastructure
- [x] **Async pattern** - Fire-and-forget, no polling
- [x] **Queue panel** - Real-time job status
- [x] **System services** - Survive user logout
- [x] **ComfyUI Manager** - pip package + --enable-manager

---

## ❌ Wat moet nog

### Prioriteit 1 (UX) 🔴
- [ ] **Global NSFW toggle** - Header switch to show/hide NSFW content
  - LoRA filtering (most are NSFW)
  - Prompt generator uncensored mode
  - Model filtering
- [ ] **User system** - Accounts, login, media ownership
  - Migrate current media to "dev" account
  - Auth via JWT or session

### Prioriteit 2 (Features) 🟡
- [ ] **ElevenLabs integration** - Alternative TTS/SFX provider
- [ ] **Batch processing** - Multiple images/videos at once
- [ ] **Pipeline builder** - Visual workflow editor
- [ ] **LoRA training** - Custom model fine-tuning

### Prioriteit 3 (Stability) 🟢
- [ ] **Error recovery** - Better error messages, retry logic
- [ ] **Model management** - Download/update models from UI
- [ ] **WebSocket reconnect** - Handle connection drops

---

## 🎨 UI/UX Conventions

### Tool Structure
```jsx
// Every tool follows this pattern:
export default function SomeTool({ onOutput, onJobSubmitted }) {
  const [submitting, setSubmitting] = useState(false)
  
  const handleGenerate = async () => {
    setSubmitting(true)
    try {
      const res = await postForm(`${BACKEND_BASE}/endpoint`, formData)
      if (res.data?.prompt_id) {
        onJobSubmitted?.(res.data)  // Notify queue panel
      }
    } finally {
      setSubmitting(false)
    }
  }
  
  return (
    <div className="tool-container">
      {/* Tool sections */}
    </div>
  )
}
```

### Nav Menu (nav.js)
```js
export const TOOL_IDS = {
  IMAGE_TO_VIDEO: 'image-to-video',
  // ...
}

export const NAV_GROUPS = [
  { id: 'video-tools', title: 'Video Tools', items: [...] },
  { id: 'image-tools', title: 'Image Tools', items: [...] },
  { id: 'prompt-tools', title: 'Prompt Tools', items: [...] },
  { id: 'audio-tools', title: 'Audio Tools', items: [...] },
  { id: 'advanced', title: 'Advanced', items: [...] },
  { id: 'my-media', title: 'My Media', items: [...] },
]
```

---

## 🖥️ Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 5090 (32GB VRAM) |
| CPU | 26 threads |
| OS | Ubuntu Linux |
| Python venv | `/home/flip/venvs/gpu` → `torch-sm120` (CUDA 12.8) |

### GPU Memory Budget
- Wan2.2 I2V 720p: ~24GB VRAM
- SDXL: ~8GB VRAM
- F5-TTS: ~4GB VRAM

---

## 📝 Coding Conventions

### Backend (Python)
- FastAPI endpoints in `app.py`
- ComfyUI workflow dicts built inline
- `client.queue_prompt(workflow)` for async jobs
- Form parameters (not JSON body) for file uploads

### Frontend (React)
- Functional components with hooks
- Inline `<style>` blocks (no separate CSS)
- `postForm()` helper for FormData requests
- Tool components in `src/dashboard/tools/`

### Git Commits
```
feat: description
fix: description
docs: description
refactor: description
```

---

## 🔗 Related Repos

| Repo | Purpose |
|------|---------|
| `oelala` | Main app (this repo) |
| `oelala-storage` | Go storage service |
| `ComfyUI` | Submodule (m0nk-fixes branch) |
| `Wan2.2` | Video model |

---

## 💡 Context voor Volgende Sessie

### Lopende Discussies
1. **NSFW toggle** - Needs React context, header component, filtering in tools
2. **User system** - JWT auth, user_id in media paths, account management
3. **ElevenLabs** - API integration for TTS/SFX (low priority)

### Recente Beslissingen
- Services zijn nu **system** services (niet user) - survive logout
- TTS voices gegroepeerd op **gender** (female default)
- Video to Text verplaatst naar **Prompt Tools** (logischer)
- **Fire-and-forget** pattern voor alle generate buttons

### Bestanden die vaak veranderen
- `src/backend/app.py` - Backend endpoints (4400+ lines)
- `src/frontend/src/dashboard/Dashboard.jsx` - Tool routing
- `src/frontend/src/dashboard/nav.js` - Menu structure
- `docs/TODO_TOOLS.md` - Implementation status
- `CHANGELOG.md` - Daily changes
