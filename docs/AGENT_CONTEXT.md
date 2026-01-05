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
- [x] **Global NSFW toggle** - Header switch to show/hide NSFW content
  - Toggle button in top-bar (🛡️ SFW / 🔞 NSFW)
  - LoRA filtering (keyword-based detection)
  - Persisted via localStorage
- [ ] **NSFW catalogisering** - Proper tagging of NSFW content
  - Current: keyword-based detection (50+ keywords)
  - Todo: Manual tagging in LoRA/model metadata
  - Todo: UI for marking content as NSFW
  - Todo: Backend database for NSFW flags
- [ ] **NSFW + User System integratie** (vereist user system)
  - Standaard (niet ingelogd): SFW only, geen toggle zichtbaar
  - Ingelogd als adult: NSFW toggle beschikbaar
  - Adult verificatie bij registratie (geboortedatum/checkbox)
  - Content gegenereerd met NSFW componenten → automatisch NSFW tagged
  - NSFW flag in media metadata opslaan
- [ ] **User system** - Accounts, login, media ownership
  - Migrate current media to "dev" account
  - Auth via JWT or session
  - Age verification for NSFW access

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

### Zojuist Afgerond (2026-01-04)
1. **NSFW toggle geïmplementeerd** ✅
   - `NSFWContext.jsx` - React context voor global state
   - Toggle button in `Dashboard.jsx` top-bar (🛡️ SFW / 🔞 NSFW)
   - Keyword-based NSFW detectie in backend `/loras` endpoint
   - Filtering in `TextToImageTool.jsx` en `ImageToVideoTool.jsx`
   - localStorage persistence (`oelala_nsfw_enabled`)

### Open Todos (prioriteit volgorde)
1. **NSFW catalogisering** - Manual tagging systeem, metadata DB voor NSFW flags
2. **NSFW + User integratie** - Toggle alleen voor adults, auto-tag bij NSFW generation
3. **User system** - JWT auth, registratie, leeftijdsverificatie, media ownership
4. **ElevenLabs** - API integration voor TTS/SFX (low priority)

### Recente Beslissingen
- NSFW toggle standaard **SFW** (false) - safe default
- Keyword-based detection is **tijdelijk** - later manual tagging
- Zonder login: **geen NSFW toggle** zichtbaar (toekomstig)
- Content met NSFW componenten: **automatisch tagged** (toekomstig)
- Services zijn **system** services (niet user) - survive logout

### Bestanden Recent Gewijzigd
- `src/frontend/src/contexts/NSFWContext.jsx` - **NEW** NSFW state
- `src/frontend/src/dashboard/Dashboard.jsx` - NSFW toggle in top-bar
- `src/frontend/src/App.jsx` - NSFWProvider wrapper
- `src/frontend/src/App.css` - NSFW toggle styling
- `src/backend/app.py` - NSFW detection in /loras endpoint
- `src/frontend/src/dashboard/tools/TextToImageTool.jsx` - LoRA filtering
- `src/frontend/src/dashboard/tools/ImageToVideoTool.jsx` - LoRA filtering
- `docs/ROADMAP.md` - Phase 5.5 Content Filtering toegevoegd

### Bestanden die vaak veranderen
- `src/backend/app.py` - Backend endpoints (4400+ lines)
- `src/frontend/src/dashboard/Dashboard.jsx` - Tool routing
- `src/frontend/src/dashboard/nav.js` - Menu structure
- `docs/TODO_TOOLS.md` - Implementation status
- `CHANGELOG.md` - Daily changes
