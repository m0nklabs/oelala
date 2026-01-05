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

### 🔴 Video to Video (V2V) - ✅ DONE
- [x] Upload source video
- [x] Style transfer options
- [x] Motion preservation slider
- [x] Denoise strength
- [x] Backend endpoint
- [x] ComfyUI workflow (VHS nodes)

**Backend implemented:**
- `/generate-v2v` endpoint ✅
- Video frame extraction (VHS_LoadVideo)
- Batch img2img processing
- Video reassembly (VHS_VideoCombine)

---

## 🖼️ Image Tools

### 🟡 Text to Image (T2I) - ✅ UPDATED
- [x] Backend endpoint (`/generate-sdxl`)
- [x] Queue-based generation
- [x] Basic model (CyberRealistic Pony)
- [x] Model/checkpoint selector dropdown
- [x] Negative prompt input
- [x] LoRA selector with strength
- [ ] Sampler selector
- [ ] Scheduler selector
- [ ] Batch generation (1-4 images)
- [ ] Image size presets
- [ ] Seed history/favorites

### 🔴 Image to Image (I2I) - ✅ DONE
- [x] Upload source image
- [x] Denoise strength slider (0.0-1.0)
- [x] Prompt input
- [x] Negative prompt
- [x] Model selector
- [x] LoRA support
- [ ] Mask painting (inpainting)
- [ ] Outpainting support

**Backend implemented:**
- `/generate-i2i` endpoint ✅
- VAE encode → KSampler → VAE decode
- Mask handling

**ComfyUI workflow:**
- LoadImage node
- VAE encode
- KSampler with denoise < 1.0
- VAE decode

### 🔴 Reframe - ✅ DONE
- [x] Upload image
- [x] Target aspect ratio selector (8 presets)
- [x] Position control (9 positions)
- [x] AI outpainting for extended areas
- [x] Preview before generation
- [x] Model selector (SDXL, Flux)
- [x] Advanced settings (steps, CFG, denoise, feathering)

**Backend implemented:**
- `/reframe` endpoint ✅
- Uses inpainting/outpainting workflow
- Image compositing with mask generation

### 🔴 Face Swap - ✅ DONE
- [x] Upload target image/video
- [x] Upload face source image
- [x] Face detection preview
- [x] Multi-face support (select which faces)
- [x] Strength slider
- [x] Post-processing (blend, GFPGAN, CodeFormer)
- [x] Model selector (InSwapper, SimSwap)

**Backend implemented:**
- `/face-swap` endpoint ✅
- `/detect-faces` endpoint ✅
- Uses ReActor node via ComfyUI
- OpenCV face detection for preview

### 🔴 Upscaler - ✅ DONE
- [x] Upload image
- [x] Scale factor selector (2x, 4x)
- [x] Model selector (RealESRGAN, GFPGAN, etc.)
- [x] Comparison slider (before/after)
- [ ] Face enhancement toggle (GFPGAN)
- [ ] Tile size for large images

**Backend implemented:**
- `/upscale` endpoint ✅
- RealESRGAN via ComfyUI

---

## 💬 Prompt Tools (NEW SECTION)

> **Note (Jan 4)**: Video to Text moved here from Video Tools. Prompt Generator is now first item.

### 🔴 Prompt Generator - ✅ PARTIAL
AI-powered prompt enhancement and generation.

- [x] Input: basic idea/keywords
- [x] Style presets (Cinematic, Anime, Photorealistic, etc.)
- [x] Enhancement modes (Expand, Refine, Variations)
- [x] Negative prompt auto-generation
- [x] History of generated prompts
- [x] Copy to clipboard
- [ ] **Priority**: Positive prompt focus (neg optional toggle)
- [ ] Output mode selector:
  - Positive only
  - Negative only
  - Both (default)
- [ ] Motion prompts for video (camera movements)
- [ ] Favorites/bookmarks

**Backend implemented:**
- `/generate-prompt` endpoint ✅
- LLM integration via Ollama
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

### 🔴 Audio Generation - ✅ PARTIAL
Text-to-Speech and Music generation.

- [x] Text to Speech (TTS) mode
- [x] Music generation mode
- [x] Sound effects mode
- [x] Voice selector (6 voices)
- [x] Music style presets (8 styles)
- [x] Duration control
- [x] Speed/pitch controls
- [x] Audio preview player
- [x] Download functionality

**Backend implemented:**
- `/generate-audio` endpoint ✅
- Modes: tts, music, sfx
- (Real integration TODO - see Audio Tools section below)

### 🔴 Prompt Library - NOT STARTED
Save and organize prompts.

- [ ] Save prompts with tags
- [ ] Categories (portrait, landscape, action, etc.)
- [ ] Search/filter
- [ ] Import/export
- [ ] Share functionality
- [ ] Community prompts (future)

---

## 🎧 Audio Tools (NEW SECTION)

> **Priority**: High - Full audio pipeline for video production
> **Reference**: ElevenLabs Clone (https://github.com/BernieTv/ElevenLabs-Clone)
> **NSFW Support**: Yes - uncensored voice cloning

### ComfyUI Audio Nodes Available

| Node Package | Function | Stars | Status |
|--------------|----------|-------|--------|
| **ComfyUI-MMAudio** | Video-to-Audio sync | 493 | ✅ Installed |
| **TTS-Audio-Suite** | All-in-One TTS + Voice Cloning | 506 | ✅ Installed (Jan 3) |
| **ComfyUI-F5-TTS** | Voice cloning TTS | 250 | ✅ Installed (Jan 3) |
| **ComfyUI-VibeVoice** | Long-form multi-speaker TTS | 547 | ✅ Bundled in TTS-Audio-Suite |
| **ComfyUI-XTTS** | Coqui XTTS v2 | 65 | 🟡 Available via TTS-Audio-Suite |
| **ComfyUI-IndexTTS** | Voice cloning + dialogue | 480 | ✅ Bundled in TTS-Audio-Suite |
| **ComfyUI-LatentSyncWrapper** | Lip sync (ByteDance) | 923 | ✅ Installed (Jan 3) |
| **ComfyUI-ytdl_nodes** | YouTube download | 29 | ✅ Installed (Jan 3) |
| **ComfyUI-StableAudioSampler** | Music generation | 265 | 🔴 TODO |
| **audio-separation-nodes** | Stem separation | 402 | 🔴 TODO |

**Nodes Available After Installation (Jan 3):**
- TTS-Audio-Suite: ChatterBox, F5TTS, VibeVoice, IndexTTS, CosyVoice, HiggsAudio, StepAudioEditX
- F5-TTS: F5TTSAudio, F5TTSAudioAdvanced, F5TTSAudioInputs
- ytdl: YTDLDownloader, YTDLLinksInput, YTDLPreview, YTDLPreviewAudio
- LatentSync: LatentSyncNode (lip sync to audio)

### 🔴 Text to Speech (TTS) - ✅ UPDATED
Advanced TTS with voice cloning.

- [x] Multi-engine support:
  - [x] F5-TTS (F5v1, E2 models)
  - [x] ChatterBox
  - [ ] XTTS v2 (17 languages)
  - [ ] VibeVoice (expressive long-form)
  - [ ] IndexTTS (dialogue)
- [x] Voice cloning:
  - [x] Upload reference audio (5-15 sec)
  - [x] Zero-shot cloning
  - [ ] Voice library management
- [x] **Gender selection (Jan 4)**:
  - [x] Female voices: Nova (default), Shimmer, Alloy
  - [x] Male voices: Echo, Fable, Onyx
  - [x] Visual grouping in UI
- [ ] Emotion/style control
- [x] Speed/pitch adjustment
- [ ] Multi-speaker support (tag-based)
- [ ] SRT subtitle import
- [ ] 23+ languages

**Backend implemented:**
- `/generate-audio` endpoint ✅ (tts, music, sfx modes)
- `/voice-clone` endpoint ✅ (F5-TTS models)

### 🔴 Voice Training/Cloning - NOT STARTED
Train custom voices (ElevenLabs-style).

- [ ] Upload training audio (multiple samples)
- [ ] Voice embedding extraction
- [ ] Fine-tuning interface
- [ ] Voice quality preview
- [ ] Export trained voice
- [ ] Voice library management
- [ ] NSFW/uncensored option

**Models to integrate:**
- StyleTTS2 (fine-tuning)
- Seed-VC (voice conversion)
- GPT-SoVITS (voice cloning)

**Backend needs:**
- `/voice/train` endpoint (background job)
- `/voice/list` - trained voices
- `/voice/preview` - test voice

### 🔴 Text to Sound Effects (SFX) - NOT STARTED
Generate sound effects from text prompts.

- [ ] Prompt input for SFX
- [ ] Duration control (1-30 sec)
- [ ] Categories:
  - Nature (rain, wind, ocean)
  - Mechanical (engines, gears)
  - UI sounds (notifications, clicks)
  - Sci-fi (lasers, spaceships)
  - Action (explosions, impacts)
  - Animals
  - Fantasy
- [ ] Preview and download
- [ ] Use in video sync

**Backend options:**
- MMAudio via ComfyUI (text-to-audio)
- Make-An-Audio
- AudioLDM

### 🔴 Sound Effects to Video (Audio Sync) - NOT STARTED
Automatically generate and sync audio to video.

- [ ] Upload video
- [ ] Auto-analyze video content
- [ ] Generate matching audio/SFX
- [ ] Timeline editor for audio placement
- [ ] Mix with existing audio
- [ ] Export combined video+audio

**ComfyUI integration:**
- ComfyUI-MMAudio (video-conditioned audio generation)
- Uses CLIP + Synchformer for video understanding

### ✅ Lip Sync - DONE (Jan 3)
Sync video lips to audio.

- [x] Upload video (face)
- [x] Upload or generate audio
- [x] Auto lip sync (LatentSyncNode)
- [x] Settings: lips_expression, inference_steps, seed
- [x] Progress tracking & result display
- [ ] Preview before render

**Backend implemented:**
- `/lip-sync` endpoint ✅
- Uses LatentSyncNode via ComfyUI
- VHS_LoadVideo + LoadAudio + VHS_VideoCombine

**Frontend:**
- LipSyncTool.jsx ✅

### ✅ YouTube Video Importer - DONE (Jan 3)
Import videos from YouTube for processing.

- [x] Paste YouTube URL
- [x] Preview video info (title, duration, thumbnail)
- [x] Quality selector (format_id)
- [x] Download as:
  - [x] Video only
  - [x] Audio only (mp3/wav/flac)
  - [x] Both
- [ ] Time range cropping
- [ ] Auto-caption extraction
- [x] Use in Video-to-Text tool
- [ ] Playlist support

**Backend implemented:**
- `/youtube/info` endpoint ✅ (yt-dlp metadata)
- `/youtube/download` endpoint ✅ (format selection)
- Fixed: yt-dlp PATH issue with shutil.which() fallback

**Frontend:**
- Integrated in VideoToTextTool.jsx ✅

### 🔴 Audio Stem Separation - NOT STARTED
Separate vocals, drums, bass, other from audio.

- [ ] Upload audio file
- [ ] Stem selection (vocals, drums, bass, other)
- [ ] Individual download
- [ ] Recombine with adjustments
- [ ] Tempo matching

**ComfyUI nodes:**
- audio-separation-nodes-comfyui

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

### ✅ Phase 1 - DONE (Jan 3)
1. ✅ **Prompt Tools section in nav**
2. ✅ **Image to Text tool** (Florence-2 via ComfyUI)
3. ✅ **Video to Text tool** (SmolVLM integration)
4. ✅ **Prompt Generator** (Ollama LLM)
5. ✅ **Text to Image** enhancements (model selector, LoRA)
6. ✅ **Image to Image** (I2I)
7. ✅ **Upscaler** (RealESRGAN)
8. ✅ **Video to Video** (V2V)
9. ✅ **Audio Generation** (TTS, Music, SFX placeholders)

### ✅ Phase 2 - DONE (Jan 3)
1. ✅ **Reframe** (outpainting)
2. ✅ **Face Swap** (ReActor/InsightFace)
3. 🟡 **LoRA Training** real implementation
4. 🟡 **Text to Video frontend** update

### ✅ Phase 3 - Audio Pipeline (COMPLETE - Jan 3, 2026)
> Full audio production suite - ElevenLabs-style functionality

1. ✅ **Install ComfyUI audio nodes** (Jan 3):
   - ✅ TTS-Audio-Suite (all-in-one: ChatterBox, F5-TTS, VibeVoice, IndexTTS, CosyVoice, HiggsAudio)
   - ✅ ComfyUI-F5-TTS (standalone voice cloning)
   - ✅ ComfyUI-ytdl_nodes (YouTube import)
   - ✅ ComfyUI-LatentSyncWrapper (lip sync)
2. ✅ **YouTube Video Importer** - `/youtube/info` & `/youtube/download` endpoints
3. ✅ **Real TTS integration** - ChatterBox via ComfyUI (multi-language)
4. ✅ **Voice Cloning** - F5-TTS with 8 model variants, VoiceCloningTool.jsx
5. ✅ **Lip Sync** - LatentSyncNode, LipSyncTool.jsx
6. 🟡 **Text to SFX** (MMAudio integration) - partial
7. 🟡 **Audio to Video Sync** (MMAudio) - partial

### 🔴 Phase 4 - Media Management & Storage (CURRENT PRIORITY)
> Unified storage, local-first distributed nodes

1. 🔴 **Unify storage locations** - Merge `generated/` + `ComfyUI/output/`
2. 🔴 **ComfyUI symlinks** - Point ComfyUI to unified `/media/` directory
3. 🔴 **Storage Node service** - Cross-platform (Windows/Linux) node daemon
4. 🔴 **Node API** - REST endpoints for node discovery & file sync
5. 🔴 **Sync engine** - P2P file synchronization between nodes
6. 🔴 **SQLite metadata** - Local database for file index & sync state
7. 🟡 **My Media** enhancements - Search, tags, folders
8. 🟢 **Encryption** - Optional at-rest encryption

### Phase 5 - Advanced Generation
1. 🟡 **Pipeline** node editor
2. 🔴 **ControlNet integration**
3. 🔴 **LoRA browser & loading**
4. 🟢 **Prompt Library**
5. 🟢 **Audio Stem Separation**

---

## 📋 Technical Requirements

### ComfyUI Custom Nodes Needed
- [x] Wan2.2 video nodes
- [x] GGUF support
- [x] ComfyUI-MMAudio (video-to-audio)
- [ ] Florence-2 captioning
- [ ] ReActor (face swap)
- [ ] ControlNet suite
- [ ] AnimateDiff (V2V)
- [ ] IPAdapter

**Audio Nodes (Phase 3)**:
- [x] TTS-Audio-Suite (all-in-one TTS + voice cloning) ✅
- [x] ComfyUI-F5-TTS (voice cloning) ✅
- [ ] ComfyUI-XTTS (17 languages) - available via TTS-Audio-Suite
- [x] ComfyUI-ytdl_nodes (YouTube download) ✅
- [x] ComfyUI-LatentSyncWrapper (lip sync) ✅
- [ ] audio-separation-nodes (stem separation)
- [ ] ComfyUI-StableAudioSampler (music gen)

### Backend Dependencies
- [x] FastAPI
- [x] ComfyUI client
- [x] OpenCV (video frame extraction)
- [ ] transformers (for local vision models)
- [ ] Ollama client (for LLM prompts)
- [ ] InsightFace (face swap)
- [ ] yt-dlp (YouTube download)
- [ ] ffmpeg (audio/video processing)

### Frontend Components Needed
- [x] ImageToTextTool.jsx ✅
- [x] PromptGeneratorTool.jsx ✅
- [x] ImageToImageTool.jsx ✅
- [x] UpscalerTool.jsx ✅
- [x] VideoToVideoTool.jsx ✅
- [x] VideoToTextTool.jsx ✅
- [x] AudioGenerationTool.jsx ✅
- [x] ReframeTool.jsx ✅
- [x] FaceSwapTool.jsx ✅

**Audio Tools (Phase 3)**: ✅ COMPLETE
- [x] AudioGenerationTool.jsx ✅ (TTS via ChatterBox)
- [x] VoiceCloningTool.jsx ✅ (F5-TTS integration)
- [ ] SoundEffectsTool.jsx (text-to-SFX) - MMAudio pending
- [ ] AudioSyncTool.jsx (video-to-audio) - MMAudio pending
- [x] LipSyncTool.jsx ✅
- [x] YouTubeImporterTool.jsx (integrated in VideoToTextTool) ✅
- [ ] StemSeparationTool.jsx - audio-separation-nodes pending

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

## 🔗 Reference Links

### Audio/Voice Projects
- **ElevenLabs Clone**: https://github.com/BernieTv/ElevenLabs-Clone
  - StyleTTS2, Seed-VC, Make-An-Audio
  - Self-hosted, Docker-based
- **ElevenLabs Official**: https://github.com/elevenlabs
  - elevenlabs-python SDK (2.8k stars)
  - elevenlabs-mcp (MCP server)
- **ElevenLabs Sound Effects**: https://elevenlabs.io/sound-effects

### ComfyUI Audio Nodes (GitHub)
- **TTS-Audio-Suite**: https://github.com/diodiogod/TTS-Audio-Suite (506⭐)
- **ComfyUI-MMAudio**: https://github.com/kijai/ComfyUI-MMAudio (493⭐)
- **ComfyUI-F5-TTS**: https://github.com/niknah/ComfyUI-F5-TTS (250⭐)
- **ComfyUI-VibeVoice**: https://github.com/wildminder/ComfyUI-VibeVoice (547⭐)
- **ComfyUI-XTTS**: https://github.com/AIFSH/ComfyUI-XTTS (65⭐)
- **ComfyUI-IndexTTS**: https://github.com/billwuhao/ComfyUI_IndexTTS (480⭐)
- **ComfyUI-LatentSyncWrapper**: https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper (923⭐)
- **ComfyUI-ytdl_nodes**: https://github.com/Saganaki22/ComfyUI-ytdl_nodes (29⭐)
- **audio-separation-nodes**: https://github.com/christian-byrne/audio-separation-nodes-comfyui (402⭐)
- **ComfyUI-StableAudioSampler**: https://github.com/lks-ai/ComfyUI-StableAudioSampler (265⭐)

### Voice Cloning Models
- **StyleTTS2**: High-quality TTS with style transfer
- **F5-TTS**: Zero-shot voice cloning (5-15 sec reference)
- **XTTS v2**: Coqui TTS with 17 languages
- **Seed-VC**: Voice conversion
- **GPT-SoVITS**: Chinese-focused voice cloning

---

*Last updated: 2026-01-03*
