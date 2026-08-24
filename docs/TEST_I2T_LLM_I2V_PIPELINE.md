# Test Plan: I2T → LLM Refinement → I2V Pipeline

> **Date**: 2026-03-02
> **Status**: 🔄 In Progress
> **Goal**: Validate an end-to-end pipeline that takes an existing image, analyzes it
> with a Vision LLM (I2T), refines the caption with a text LLM into a cinematic video
> prompt, and then generates a 10-second video from the original image + enhanced prompt.

---

## Pipeline Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Source Image    │────▶│  I2T Analysis   │────▶│  LLM Refine     │────▶│  I2V Generation  │
│  (recent I2I)   │     │  (Guardian VLM)  │     │  (GLM-4.7-Flash)│     │  (DisTorch2 Q8)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └──────────────────┘
       │                       │                       │                       │
       │                       ▼                       ▼                       ▼
  oelala_i2i_       Raw description of        Enhanced cinematic       10s 480p video
  00025_.png        image content             I2V prompt               @ 16fps (161 frames)
```

---

## Step-by-Step Plan

### Step 1: Source Image Selection

| Property | Value |
|----------|-------|
| **File** | `ComfyUI/output/oelala_i2i_00025_.png` |
| **Created** | 2026-03-02 21:31 |
| **Type** | I2I generation output |

### Step 2: I2T — Image-to-Text Analysis

| Property | Value |
|----------|-------|
| **Endpoint** | `POST /caption-image` |
| **Method** | Multipart form upload |
| **Mode** | `detailed` (rich description first) |
| **Model** | Guardian VLM (env: `VISION_MODEL` → Gemma3-27B-it-vl) |
| **Expected Output** | Detailed text description of the image contents |

**What happens**: The image is base64-encoded and sent to the Guardian VLM proxy
via the OpenAI-compatible `/v1/chat/completions` endpoint. The VLM analyzes the
visual content and returns a natural language description.

### Step 3: LLM Prompt Refinement

| Property | Value |
|----------|-------|
| **Model** | `glm-4-9b-chat` (GLM-4.7-Flash) |
| **API** | Guardian proxy `/v1/chat/completions` |
| **Input** | Raw I2T description from Step 2 |
| **Output** | Cinematic video generation prompt |

**What happens**: The raw description is sent to a text-only LLM with a system
prompt that instructs it to:
1. Preserve the key visual elements (subject, setting, colors, mood)
2. Add cinematic motion descriptions (camera movements, subject actions)
3. Add quality boosters for video generation
4. Keep the prompt focused and concise (under 200 tokens)
5. Format it as a Wan2.2-optimized I2V prompt

**System prompt template**:
```
You are a cinematic video prompt engineer. Given an image description,
create an enhanced prompt for AI video generation (Wan2.2 model).

Rules:
- Start with the main subject and action
- Add natural, subtle motion (not jarring)
- Include camera movement if appropriate (slow pan, gentle zoom)
- Add atmospheric details (lighting, particles, wind)
- Keep under 150 words
- Do NOT use markdown, just plain text
- Focus on what MOVES, not static details
```

### Step 4: I2V — Image-to-Video Generation

| Property | Value |
|----------|-------|
| **Endpoint** | `POST /generate-distorch2-q8-async` |
| **Image** | Same source image from Step 1 |
| **Prompt** | Enhanced prompt from Step 3 |
| **Resolution** | `480p` |
| **FPS** | `16` |
| **Duration** | 10 seconds → `161` frames (snapped to 4k+1) |
| **Steps** | `8` (default) |
| **CFG** | `3.5` |
| **NAG Scale** | `11.0` |
| **Shift** | `8.0` |
| **Florence2** | `false` (we already have an enhanced prompt) |
| **GPU Allocation** | `cuda:0,11gb;cuda:1,15gb;cpu,*` |
| **Expected VRAM** | ~20-22GB (480p 161 frames is well within limits) |
| **Expected Time** | ~12 min (161 frames × 8 steps) |

**What happens**: The original image + refined prompt are sent to the DisTorch2 Q8
pipeline. The workflow loads the Wan2.2 14B Q8 model across both GPUs, uses the
image as the starting frame, and generates 161 frames of video guided by the
enhanced prompt.

---

## Success Criteria

- [ ] Step 2 produces a meaningful description of the image
- [ ] Step 3 produces a cinematic prompt that's specific to the image content
- [ ] Step 4 successfully queues a ComfyUI job
- [ ] Generated video is coherent with the source image
- [ ] Video is approximately 10 seconds at 16fps
- [ ] Resolution is 480p
- [ ] Total pipeline completes (I2T + LLM + I2V queue) within budget

---

## Execution Log

### Step 2 Result (I2T)
- **Model**: Gemma3-27B-it-vl-GLM-4.7-Uncensored-Heretic
- **Mode**: detailed
- **Caption length**: 1543 chars
- **Key content**: Young woman with blonde hair, standing by window in natural lighting, soft inviting expression, purple object near hip, clear blue sky outside, professional photoshoot aesthetic

### Step 3 Result (LLM Refinement)
- **Model**: glm-4-9b-chat (GLM-4.7-Flash)
- **API**: Guardian at `GUARDIAN_BASE_URL`
- **Refined prompt** (960 chars):

> A young woman with fair skin and long, flowing blonde hair stands in profile by a window, her slender form etched in soft, natural light. Her hair cascades over her shoulders, moving slightly with the gentle breeze that whispers through the room. Her expression is soft, inviting, eyes soft and open, her smile gentle and inviting, as she turns slightly to look directly at the camera. One hand rests on the sill, the other gently cradling a small purple object near her hip. The camera gently zooms in, capturing the subtle play of light across her skin and the soft shadows that define her curves. Outside, the house and clear blue sky suggest daytime, while the movement of light shifting across the scene adds depth and life to the cinematic experience. The background remains still, emphasizing the woman's serene and intimate presence, as the cinematic quality of the lighting and the natural color palette create an enchanting, professional atmosphere.

### Step 4 Result (I2V)
- **Prompt ID**: `381a0bb4-529a-42f8-8e97-a028be376880`
- **Status**: ✅ QUEUED (generating...)
- **Workflow**: DisTorch2 Q8, 29 nodes
- **Settings**: 161 frames, 16fps, 480×848, 8 steps, cfg 3.5, NAG 11.0, shift 8.0
- **Florence2**: disabled (using LLM-refined prompt instead)
- **Expected time**: ~12 minutes

---

## Notes

- Guardian VLM requires VRAM — ComfyUI models may need to be unloaded first
- The `/caption-image` endpoint handles VRAM management automatically
- Auth is required for `/generate-distorch2-q8-async` — will use the API directly
  via curl with session cookie or via the backend's internal flow
- 16fps at 161 frames = 10.0625 seconds (close enough to 10s target)
- Florence2 is disabled because we provide our own enhanced prompt
