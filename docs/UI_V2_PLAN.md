# Oelala UI v2 Plan (Grok-Imagine inspired)

## Goal

Build a new dashboard UI inspired by the provided Grok Imagine screenshots:
- Left sidebar navigation (grouped tools)
- Main area split into **Controls** (left) and **Generated output** (right)
- A consistent **History** affordance in the output panel
- Minimal, fast operator UX for driving the backend generation/training endpoints

This is a UX/layout reference plan, not a pixel-perfect clone.

## Current backend capabilities (source of truth)

Implemented FastAPI endpoints (see `src/backend/app.py`):

### Generation
- `POST /generate` — Image → Video (multipart upload: `file`, plus `prompt`, `num_frames`, `output_filename`)
- `POST /generate-text` — Text → Video (form: `prompt`, `num_frames`, `model_type`, `output_filename`)
- `POST /generate-pose` — Pose-guided Image → Video (multipart upload: `file`, `num_frames`, `output_filename`)

### Output / history
- `GET /health`
- `GET /files/{filename}` (generic file serving)
- `GET /videos/{filename}`
- `GET /images/{filename}`
- `GET /list-videos` — lists generated mp4 files

### Training
- `POST /train-lora` — LoRA training (multipart multiple files)
- `POST /train-lora-placeholder` — placeholder artifact creation

### Telemetry
- `POST /client-log` — persist client-side logs

## Feature matrix (UI modules vs backend support)

| UI module (Grok reference) | Backend support today | Notes |
|---|---:|---|
| Text to Image | ❌ missing | No endpoint yet |
| Text to Video | ✅ supported | `POST /generate-text` |
| Image to Video | ✅ supported | `POST /generate` |
| Video to Video | ❌ missing | No endpoint yet |
| Image to Image | ❌ missing | No endpoint yet |
| Reframe | ❌ missing | No endpoint yet |
| Face Swap | ❌ missing | No endpoint yet |
| Upscaler | ❌ missing | No endpoint yet |
| History | ✅ supported | `GET /list-videos` + direct video URLs |
| LoRA training | ✅ supported (but may be limited by env) | `POST /train-lora`, plus placeholder endpoint |

## v1 (MVP) scope — end-to-end working

v1 should deliver a Grok-style shell but only wire the flows that are actually available:

1. **Text to Video**
   - Prompt textarea
   - Duration/frames control (maps to `num_frames`)
   - Model type selection (maps to `model_type`)
   - Submit -> show generated video + download

2. **Image to Video**
   - Upload image
   - Prompt (optional)
   - Frames control
   - Submit -> show generated video + download

3. **Pose-guided Image to Video (optional toggle under Image→Video)**
   - Upload image
   - Pose toggle -> routes to `POST /generate-pose`

4. **History (always available on the right panel)**
   - “History” button opens a list of prior generated mp4s from `/list-videos`
   - Clicking an item loads it into the output preview

5. **LoRA Training**
   - Multi-image upload
   - Model name + epochs + learning rate
   - Submit -> show training response + artifact path

## v2 scope — requires backend work

These modules can be added after we implement backend endpoints:
- Text to Image
- Image to Image
- Video to Video
- Reframe
- Face Swap
- Upscaler

When implementing these, keep the exact same pattern:
- Controls (left) -> submit -> output (right) -> History

## UI information architecture (IA)

### Sidebar groups (mirrors Grok)
- **Video tools**
  - Text to Video (v1)
  - Image to Video (v1)
  - Video to Video (v2)
- **Image tools**
  - Text to Image (v2)
  - Image to Image (v2)
  - Reframe (v2)
  - Face Swap (v2)
  - Upscaler (v2)
- **My content**
  - History (v1)
  - LoRA Training (v1)

### Shared page layout
- Left sidebar (collapsible)
- Main content area:
  - **Controls panel (left)**: mode settings, upload/prompt, advanced settings
  - **Output panel (right)**: preview player/image, status/errors, History button

## API wiring notes (practical)

- Prefer a single lightweight client helper around `fetch` (already in `src/frontend/src/api.js`).
- Use `POST /client-log` for any UI exceptions.
- `GET /list-videos` can drive History; videos can be played from `GET /videos/{filename}`.

## Known gaps / risks

- Current frontend is React+Vite but not yet TypeScript/Tailwind/Zustand/React Query.
  - For v1, we can keep it minimal and re-layout with existing tooling.
  - Alternatively, migrate to TS/Tailwind and follow the stricter frontend conventions.
- Backend parameter mismatch: frontend currently sends `model_type` to `/generate`, but backend `/generate` does not accept it.
  - v1 should align UI fields with actual backend params to avoid confusion.

## Next implementation steps

1. Implement new UI shell (sidebar + two-panel layout)
2. Implement MVP pages: Text→Video, Image→Video (pose toggle)
3. Add output preview + downloads + History panel
4. Wire LoRA training screen
5. Polish error/loading states and logging
