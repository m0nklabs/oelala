# Face System

Face identity, cloning, swapping (image + video) and LoRA training.

---

## Architecture

```
FaceSwapTool.jsx (3 tabs: Swap / Profiles / Train LoRA)
        │
        ▼
   FastAPI (app.py)  ──── face_service.py       ── insightface (inswapper_128)
                     ──── face_train_service.py  ── ai-toolkit (SDXL Dreambooth)
```

**No ComfyUI needed for face swap** — uses insightface directly (synchronous, CPU/GPU via ONNX).

---

## Models & Paths

| Model | Path | Size |
|-------|------|------|
| buffalo_l analyzer | `ComfyUI/models/insightface/models/buffalo_l/` | ~320MB |
| inswapper_128 | `ComfyUI/models/insightface/inswapper_128.onnx` | 554MB |
| Face LoRAs (trained) | `ComfyUI/models/loras/face_loras/` | varies |
| Base model for training | `ComfyUI/models/checkpoints/juggernautXL_ragnarok.safetensors` | ~7GB |

---

## Backend Endpoints

### Face Detection
| Method | Path | Description |
|--------|------|-------------|
| POST | `/detect-faces` | Detect faces in image, returns bounding boxes + confidence |

### Image Face Swap
| Method | Path | Description |
|--------|------|-------------|
| POST | `/face-swap` | Swap face(s) in image using uploaded source photo |
| POST | `/face-swap/profile` | Swap face(s) in image using saved face profile |

### Video Face Swap
| Method | Path | Description |
|--------|------|-------------|
| POST | `/face-swap-video` | Swap face(s) in every frame using uploaded source photo |
| POST | `/face-swap-video/profile` | Swap face(s) in every frame using saved face profile |

### Face Profiles
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/face-profiles` | List all saved profiles |
| GET | `/api/face-profiles/{id}` | Get single profile |
| POST | `/api/face-profiles` | Create profile from 1+ reference images |
| DELETE | `/api/face-profiles/{id}` | Delete profile |

### Face LoRA Training
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/face-train` | Start a Dreambooth LoRA training job |
| GET | `/api/face-train` | List all training jobs + status |
| GET | `/api/face-train/loras` | List completed face LoRAs |

---

## Face Profiles

Profiles store averaged embeddings from multiple reference photos for more stable identity matching.

**Storage:** `data/face_profiles/{profile_id}/`
- `index.json` — profile metadata
- `images/ref_*.jpg` — reference photos
- `embedding.npy` — averaged normed embedding (numpy float32 array)

**Usage:** Upload 5–20 clear frontal photos for best identity stability.

---

## Video Face Swap

Frame-by-frame swap using `cv2.VideoCapture` + `inswapper_128.onnx`:

1. Write video to temp file
2. Open with OpenCV, read FPS/resolution
3. Swap each frame via insightface
4. Write swapped frames with `cv2.VideoWriter` (mp4v codec)
5. Remux audio from original using ffmpeg (`-map 1:a:0?`)
6. Return MP4 bytes

**Audio:** Preserved via ffmpeg remux. If original has no audio, output is silent (no error).

**Performance:** ~1–5 seconds per frame depending on resolution. A 10-second 30fps video ≈ 5–25 min.

---

## Face LoRA Training

SDXL Dreambooth LoRA training via [ai-toolkit](https://github.com/ostris/ai-toolkit).

**Location:** `external/ai-toolkit/` (in `.gitignore`, not committed)

**Trigger word convention:** `ohwx_{name_snake_case}`
- Name "John Doe" → trigger word `ohwx_john_doe`

**Base model:** `juggernautXL_ragnarok.safetensors` (loaded via `from_single_file`)

**Output:** `ComfyUI/models/loras/face_loras/{name}_{timestamp}.safetensors`

**Job tracking:** `data/face_train_jobs/index.json`

### Recommended settings
| Parameter | Value | Notes |
|-----------|-------|-------|
| Steps | 1000–2000 | 1000 = fast test, 2000 = better quality |
| Photos | 10–20 | Clear frontal, varied lighting preferred |
| Resolution | 512×512 | ai-toolkit default for faces |

### Dependencies (in `/home/flip/venvs/gpu`)
- `lycoris-lora==1.8.3` ✅
- `prodigyopt` ✅
- `torchao==0.10.0` ⚠️ May be missing — run: `pip install torchao==0.10.0`

---

## Frontend: FaceSwapTool.jsx

**3-tab component:** `src/frontend/src/dashboard/tools/FaceSwapTool.jsx`

### Tab 1: Swap
- Target: image or video (drag/drop or click)
- Source mode toggle: **Upload photo** vs **Saved profile**
- Face detection (images only) → select which face(s) to swap
- Video target → uses `/face-swap-video/*` endpoints, shows `<video>` result
- Image target → uses `/face-swap/*` endpoints, shows `<img>` result
- Download result (`.mp4` or `.png`)

### Tab 2: Profiles
- Create from 1+ photos (multi-upload)
- List profiles with photo count
- Delete profiles

### Tab 3: Train LoRA
- Name + steps slider + photo upload
- Start training job (background)
- Poll job status with progress bar
- List trained LoRAs + copy trigger word

---

## Known Issues & Next Steps

| Priority | Item | Status |
|----------|------|--------|
| HIGH | End-to-end test: image swap (image → image) | ⏳ needs testing |
| HIGH | End-to-end test: video swap (mp4 → mp4) | ⏳ needs testing |
| HIGH | End-to-end test: face profile create + swap | ⏳ needs testing |
| HIGH | End-to-end test: LoRA training (200 steps, 2 photos) | ⏳ needs testing |
| MED | GFPGAN face enhancement post-swap | ❌ not implemented |
| MED | Batch video upload (multiple videos, same profile) | ❌ not implemented |
| MED | Use LoRA in ComfyUI I2V workflow for face consistency | ❌ not implemented |
| LOW | Face detection in video (per-frame) | ❌ not implemented |
| LOW | Multiple source faces (per-face-in-target mapping) | ❌ not implemented |

---

## Testing

```bash
# Image swap (direct)
curl -X POST http://localhost:7998/face-swap \
  -F target=@/path/to/target.jpg \
  -F source=@/path/to/source.jpg \
  -F face_indices=0 > result.png

# Video swap (direct)
curl -X POST http://localhost:7998/face-swap-video \
  -F video=@/path/to/video.mp4 \
  -F source=@/path/to/face.jpg \
  -F face_indices=0 > result.mp4

# Create profile
curl -X POST http://localhost:7998/api/face-profiles \
  -F name="Test Person" \
  -F images=@/path/to/face1.jpg \
  -F images=@/path/to/face2.jpg

# Video swap with profile
curl -X POST http://localhost:7998/face-swap-video/profile \
  -F video=@/path/to/video.mp4 \
  -F profile_id=<profile_id> \
  > result.mp4
```
