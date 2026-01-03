# Oelala Tools Implementation Todo List

> Created: 2026-01-03  
> Status: Active Development  
> Priority: High = 🔴, Medium = 🟡, Low = 🟢

---

## Overview

Dit document beschrijft alle tools die geïmplementeerd moeten worden in de Oelala web interface. Elke tool heeft een frontend component en mogelijk backend endpoints nodig.

### Architecture
- **Frontend**: React components in `src/frontend/src/dashboard/tools/`
- **Backend**: FastAPI endpoints in `src/backend/app.py`
- **ComfyUI**: Workflows in `ComfyUI/user/default/workflows/`

---

## 📹 Video Tools

### ✅ Image to Video (I2V) - DONE
- [x] Upload image
- [x] Prompt input
- [x] Resolution selector (480p, 720p)
- [x] Aspect ratio selector
- [x] Frame count slider
- [x] FPS selector
- [x] LoRA selector
- [x] Unet model selector
- [x] Sampling settings (steps, CFG, seed)
- [x] Queue-based generation
- [x] Progress polling
- [x] Result display

### 🟡 Text to Video (T2V) - IN PROGRESS
- [x] Backend endpoint (`/generate-text`)
- [x] Queue-based generation
- [ ] Frontend component update (currently shows "Coming Soon")
- [ ] T2I checkpoint selector
- [ ] T2I settings (steps, CFG)
- [ ] Video settings (frames, fps, resolution)
- [ ] LoRA support for both T2I and I2V stages
- [ ] Two-stage preview (show generated image before video)

### ✅ Text to Image to Video (T2I2V) - DONE
- [x] Combined workflow
- [x] Preset selector
- [x] Queue-based

### 🔴 Video to Video (V2V) - NOT STARTED
- [ ] Upload source video
- [ ] Style transfer options
- [ ] Motion preservation slider
- [ ] Frame interpolation
- [ ] Backend endpoint
- [ ] ComfyUI workflow

**Backend needs:**
- `/generate-v2v` endpoint
- Video frame extraction
- Frame-by-frame or batch processing
- Video reassembly

**ComfyUI workflow:**
- Video load node
- Frame extraction
- I2I per frame or AnimateDiff approach
- Video encode

---

## 🖼️ Image Tools

### 🟡 Text to Image (T2I) - PARTIAL
- [x] Backend endpoint (`/generate-sdxl`)
- [x] Queue-based generation
- [x] Basic model (CyberRealistic Pony)
- [ ] Model/checkpoint selector dropdown
- [ ] Negative prompt input
- [ ] LoRA selector with strength
- [ ] Sampler selector
- [ ] Scheduler selector  
- [ ] Batch generation (1-4 images)
- [ ] Image size presets
- [ ] Seed history/favorites

### 🔴 Image to Image (I2I) - NOT STARTED
- [ ] Upload source image
- [ ] Denoise strength slider (0.0-1.0)
- [ ] Prompt input
- [ ] Negative prompt
- [ ] Model selector
- [ ] LoRA support
- [ ] Mask painting (inpainting)
- [ ] Outpainting support

**Backend needs:**
- `/generate-i2i` endpoint
- Image preprocessing
- Mask handling

**ComfyUI workflow:**
- LoadImage node
- VAE encode
- KSampler with denoise < 1.0
- VAE decode

### 🔴 Reframe - NOT STARTED
- [ ] Upload image
- [ ] Target aspect ratio selector
- [ ] Position control (center, top, bottom, left, right)
- [ ] AI outpainting for extended areas
- [ ] Preview before generation

**Backend needs:**
- `/reframe` endpoint
- Image padding calculation
- Outpainting workflow trigger

**ComfyUI workflow:**
- Outpainting/inpainting hybrid
- SDXL or Flux for fill generation

### 🔴 Face Swap - NOT STARTED
- [ ] Upload target image/video
- [ ] Upload face source image
- [ ] Face detection preview
- [ ] Multi-face support (select which faces)
- [ ] Strength slider
- [ ] Post-processing (blend, enhance)

**Backend needs:**
- `/face-swap` endpoint
- Face detection (InsightFace/RetinaFace)
- Face embedding extraction
- Face swap execution

**ComfyUI nodes needed:**
- ReActor or IPAdapter face
- InsightFace loader

### 🔴 Upscaler - NOT STARTED
- [ ] Upload image
- [ ] Scale factor selector (2x, 4x)
- [ ] Model selector (RealESRGAN, GFPGAN, etc.)
- [ ] Face enhancement toggle
- [ ] Tile size for large images
- [ ] Comparison slider (before/after)

**Backend needs:**
- `/upscale` endpoint
- Multiple upscale models
- Tiled processing for large images

**ComfyUI workflow:**
- UpscaleModelLoader
- ImageUpscaleWithModel
- Optional face restore

---

## 💬 Prompt Tools (NEW SECTION)

### 🔴 Image to Text (I2T) - NOT STARTED
Caption/describe images using vision models.

- [ ] Upload image
- [ ] Model selector:
  - Florence-2 (fast, accurate)
  - BLIP-2 (detailed)
  - LLaVA (conversational)
  - CogVLM (detailed)
- [ ] Caption mode:
  - Brief (1 line)
  - Detailed (paragraph)
  - Tags (comma-separated)
  - Structured (subject, style, mood, etc.)
- [ ] Copy to clipboard
- [ ] Send to T2I/I2V prompt

**Backend needs:**
- `/caption-image` endpoint
- Vision model loading (Florence, BLIP, etc.)
- Model caching for speed

**Options:**
1. Local models via transformers
2. ComfyUI Florence2 node
3. External API (OpenAI Vision, Claude)

### 🔴 Prompt Generator - NOT STARTED
AI-powered prompt enhancement and generation.

- [ ] Input: basic idea/keywords
- [ ] Style presets:
  - Cinematic
  - Anime
  - Photorealistic
  - Abstract
  - Vintage
  - Cyberpunk
  - Fantasy
  - etc.
- [ ] Enhancement modes:
  - Expand (add details)
  - Refine (improve grammar/structure)
  - Variations (3-5 alternatives)
- [ ] Negative prompt auto-generation
- [ ] Motion prompts for video (camera movements)
- [ ] History of generated prompts
- [ ] Favorites/bookmarks

**Backend needs:**
- `/generate-prompt` endpoint
- LLM integration (local or API)
- Style templates

**Options:**
1. Local LLM (Mistral, Llama via Ollama)
2. OpenAI API
3. Anthropic API
4. Template-based (no AI)

### 🔴 Prompt Library - NOT STARTED
Save and organize prompts.

- [ ] Save prompts with tags
- [ ] Categories (portrait, landscape, action, etc.)
- [ ] Search/filter
- [ ] Import/export
- [ ] Share functionality
- [ ] Community prompts (future)

---

## ⚙️ Advanced Tools

### 🟡 Pipeline - PARTIAL
- [x] Basic pipeline component
- [ ] Node-based editor
- [ ] Drag-and-drop workflow builder
- [ ] Custom node creation
- [ ] Save/load pipelines
- [ ] Batch processing

### 🟡 LoRA Training - PARTIAL
- [x] Image upload
- [x] Basic config
- [ ] Actual training (currently placeholder)
- [ ] Training progress
- [ ] Preview generations during training
- [ ] Kohya-ss integration
- [ ] Training presets (face, style, object)

---

## 📁 My Media

### ✅ All - DONE
### ✅ Videos - DONE
### ✅ Images - DONE
### ✅ Prompts - DONE
- [x] Prompt bubble on thumbnails
- [x] Popup with full metadata
- [x] Copy to clipboard

### 🟢 Enhancements (Low Priority)
- [ ] Folders/albums
- [ ] Tags
- [ ] Search
- [ ] Bulk operations (delete, download, move)
- [ ] Favorites
- [ ] Cloud sync

---

## 🎯 Implementation Priority

### Phase 1 - This Week (Jan 3-10)
1. 🔴 **Prompt Tools section in nav**
2. 🔴 **Image to Text tool** (Florence-2 via ComfyUI)
3. 🔴 **Prompt Generator** (template-based first, then LLM)
4. 🟡 **Text to Video frontend** (backend done)
5. 🟡 **Text to Image** enhancements (model selector, LoRA)

### Phase 2 - Next Week (Jan 10-17)
1. 🔴 **Image to Image** (I2I)
2. 🔴 **Upscaler**
3. 🔴 **Reframe**
4. 🟡 **LoRA Training** real implementation

### Phase 3 - Week 3 (Jan 17-24)
1. 🔴 **Video to Video**
2. 🔴 **Face Swap**
3. 🟡 **Pipeline** node editor
4. 🟢 **My Media** enhancements

---

## 📋 Technical Requirements

### ComfyUI Custom Nodes Needed
- [x] Wan2.2 video nodes
- [x] GGUF support
- [ ] Florence-2 captioning
- [ ] ReActor (face swap)
- [ ] ControlNet suite
- [ ] AnimateDiff (V2V)
- [ ] IPAdapter

### Backend Dependencies
- [x] FastAPI
- [x] ComfyUI client
- [ ] transformers (for local vision models)
- [ ] Ollama client (for LLM prompts)
- [ ] InsightFace (face swap)

### Frontend Components Needed
- [ ] ImageCaptionTool.jsx
- [ ] PromptGeneratorTool.jsx
- [ ] ImageToImageTool.jsx
- [ ] UpscalerTool.jsx
- [ ] ReframeTool.jsx
- [ ] FaceSwapTool.jsx
- [ ] VideoToVideoTool.jsx

---

## 📝 Notes

### Reusable Components
Veel tools delen dezelfde componenten:
- Image upload dropzone
- Prompt input (positive/negative)
- Model/checkpoint selector
- LoRA selector
- Sampling settings (steps, CFG, seed)
- Progress indicator
- Result display

**Action**: Create shared components in `src/frontend/src/components/`

### API Consistency
Alle generate endpoints moeten dezelfde structuur volgen:
```json
// Request
POST /generate-xxx
FormData: { file?, prompt, settings... }

// Response (immediate)
{ "status": "queued", "prompt_id": "xxx" }

// Poll
GET /comfyui/job/{prompt_id}
{ "status": "completed|running|error", "url": "..." }
```

---

*Last updated: 2026-01-03*
