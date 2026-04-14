# Story Writer — Vision & Architecture

> **Status**: Vision / Design Phase
> **Author**: Flip + MARK1
> **Created**: 2026-04-12
> **Last Updated**: 2026-04-12

## TL;DR

A tool that turns reference images + a story idea into a complete AI-generated movie.

Upload reference images → LLM writes a screenplay with per-scene director instructions →
AI generates 15-second video clips per scene → stitch into a cohesive movie.

**Goal**: Make movie creation accessible through scripting and LLM assistance.
No camera, no actors, no budget — just vision.

---

## Core Concept

```
┌─────────────────────────────────────────────────────────────────────┐
│  REFERENCE IMAGES (tone, style, characters, locations)              │
│  User uploads 1-N images that define the visual world               │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CONCEPT ANALYSIS (existing I2T pipeline)                           │
│  Vision LLM extracts: scene, subjects, mood, style, color palette   │
│  Per-image: structured JSON concept cards                           │
│  Aggregate: world bible (characters, locations, visual language)     │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STORY WRITER (LLM)                                                 │
│  Input: world bible + user story prompt/outline                     │
│  Output: screenplay with N scenes, each containing:                 │
│    - scene_description: what happens                                │
│    - subjects: who appears (linked to reference images)             │
│    - location: where (linked to reference images)                   │
│    - mood / tone                                                    │
│    - suggested_motion: subject animation                            │
│    - suggested_camera: full cinematic camera direction              │
│    - suggested_audio: soundscape + music                            │
│    - suggested_dialogue: [{subject, line, emotion}]                 │
│    - duration: target seconds (max 15)                              │
│    - transition_to_next: cut/fade/dissolve/match-cut               │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STORYBOARD REVIEW (UI)                                             │
│  User sees all scenes as cards with:                                │
│    - Reference image thumbnail                                      │
│    - Scene description + director's notes                           │
│    - Editable fields (camera, motion, dialogue, audio)              │
│    - Drag-to-reorder scenes                                         │
│    - Add/remove/split/merge scenes                                  │
│    - Per-scene "Refine" button (LLM adjust)                         │
│    - Per-scene "Generate Preview" (low-res quick test)              │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  VIDEO GENERATION (per scene)                                       │
│  For each scene:                                                    │
│    1. Build positive prompt from:                                   │
│       - scene_description + camera direction + motion + mood        │
│    2. Use reference image as I2V input (character/location anchor)  │
│    3. Optional: IP-Adapter FaceID for character consistency         │
│    4. Generate 15s max clip via Wan2.2 / LTX / RunPod Cloud Max    │
│    5. Add audio (ambient + dialogue via TTS if available)           │
│  Parallel generation where VRAM allows                              │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ASSEMBLY                                                           │
│  - Stitch clips in scene order with transitions                     │
│  - Add audio track (ambient + music + dialogue)                     │
│  - Optional: title card, credits                                    │
│  - Export as single MP4                                              │
│  - Store in MinIO storage                                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## What Already Exists in oelala

| Capability | Status | Where |
|-----------|--------|-------|
| Image analysis → structured concept JSON | ✅ Working | `POST /caption-image` mode=concept |
| LLM-driven camera direction | ✅ Working (just shipped) | `suggested_camera` in concept schema |
| Director's notes (motion, audio, dialogue, camera) | ✅ Working | Concept Studio UI |
| Notes refinement via LLM | ✅ Working | `refinement_target=notes` |
| I2V generation (Wan2.2, local multi-GPU) | ✅ Working | ComfyUI + DisTorch2 |
| I2V generation (Cloud Max, RunPod) | ✅ Working | RunPod serverless |
| LTX-2.3 generation (RunPod) | ✅ Working | RunPod LTX23 endpoint |
| IP-Adapter FaceID (character consistency) | ✅ Working | I2I pipeline |
| Camera motion → prompt prefix | ✅ Working | All video tools |
| Audio prompt generation | ✅ Working | `audio_context` in caption pipeline |
| Video stitching | ❌ Not built | Need ffmpeg assembly step |
| Multi-image world building | ❌ Not built | Need aggregate concept from N images |
| Story/screenplay generation | ❌ Not built | Need LLM story writer |
| Scene-based generation queue | ❌ Not built | Need batch orchestration |
| Storyboard UI | ❌ Not built | New frontend component |

**Bottom line**: ~60% of the infrastructure exists. The missing pieces are orchestration,
multi-image aggregation, story writing, and the storyboard UI.

---

## Scene Schema (proposed)

```json
{
  "story_id": "uuid",
  "title": "My Movie",
  "world": {
    "characters": [
      {
        "id": "char_1",
        "name": "Alice",
        "reference_image_id": "img_abc",
        "description": "tall woman with red hair, leather jacket",
        "face_embedding_id": "face_xyz"
      }
    ],
    "locations": [
      {
        "id": "loc_1",
        "name": "The Beach",
        "reference_image_id": "img_def",
        "description": "rocky beach at sunset, dramatic cliffs"
      }
    ],
    "style": {
      "visual_tone": "cinematic noir with warm highlights",
      "color_palette": "deep blues, amber highlights, dark shadows",
      "era": "modern"
    }
  },
  "scenes": [
    {
      "scene_number": 1,
      "scene_description": "Alice walks along the beach, lost in thought. Waves crash against the rocks.",
      "location_id": "loc_1",
      "character_ids": ["char_1"],
      "mood": "melancholic, introspective",
      "suggested_motion": "Alice walks slowly from right to left, wind blowing her hair. Waves crash rhythmically.",
      "suggested_camera": "Wide establishing shot slowly dollying in to medium shot of Alice from behind, slight crane up to reveal the horizon.",
      "suggested_audio": "crashing waves, distant seagulls, melancholic piano underscore",
      "suggested_dialogue": [
        {"subject": "Alice", "line": "I keep coming back here.", "emotion": "wistful"}
      ],
      "duration_seconds": 12,
      "transition_to_next": "slow dissolve",
      "reference_image_id": "img_def",
      "generation_config": {
        "resolution": "576x1024",
        "model": "wan22",
        "use_face_id": true,
        "steps": 6
      }
    }
  ]
}
```

---

## Technical Challenges & Solutions

### 1. Cross-Scene Character Consistency

**Problem**: Same character must look the same across different scenes/clips.

**Solutions available in oelala**:
- **IP-Adapter FaceID Plus V2** — already working in I2I pipeline
- **Reference image anchoring** — use same reference image per character across scenes
- **Face embedding storage** — face system already exists (`FACE_SYSTEM.md`)

### 2. Scene Transition Coherence

**Problem**: Cuts between scenes look jarring, no visual flow.

**Solutions**:
- LLM generates explicit `transition_to_next` instructions
- Last frame of scene N used as reference for scene N+1 (sliding window approach, like StoryGen-Atelier)
- Transition types: hard cut, dissolve, fade to black, match-cut

### 3. Long-Form Narrative Coherence

**Problem**: Story drifts, characters act out of character, locations change unexpectedly.

**Solutions**:
- **World bible** as persistent context for all LLM calls
- Scene refinement always receives full story context (not just single scene)
- Character sheets with personality, appearance, relationships

### 4. Generation Time

**Problem**: 15s × 10 scenes = potentially hours of generation time.

**Solutions**:
- **Preview mode**: low-res, fewer frames (4s preview per scene) for rapid iteration
- **Cloud burst**: offload to RunPod Cloud Max for parallel generation
- **Progressive**: generate scene 1 while user reviews/edits scene 2-N
- **Priority queue**: generate in scene order, allow re-generation of individual scenes

### 5. Camera Direction → Prompt Engineering

**Problem**: Rich camera descriptions need to map to effective video generation prompts.

**Solutions**:
- Camera direction → prompt prefix (already working)
- LLM translates "slow dolly-in revealing the sky" → "camera dollying forward, slowly tilting up to reveal sky, cinematic"
- Backend `_build_story_prompt()` assembles: camera + motion + scene + mood

---

## Competitive Landscape (as of 2026-04)

| Project | Type | Status | Differentiator |
|---------|------|--------|---------------|
| **StoryGen-Atelier** | Open source | Active but hobby | Closest competitor; uses Gemini + Veo (closed APIs) |
| **VideoDirectorGPT** | Research paper | Code unreleased | LLM-guided multi-scene planning with layout control |
| **InfinityStory** (2026) | Paper | Research only | World consistency + character-aware transitions |
| **FilmAgent** (2025) | Paper | Research only | Multi-agent film automation in 3D spaces |
| **Runway / Pika / Luma** | Commercial | Production | Single-clip T2V/I2V only, no story orchestration |

**Our advantages**:
- Open-source video models (Wan2.2, LTX) — no vendor lock-in
- Existing multi-GPU infrastructure (28GB VRAM local + RunPod cloud)
- Reference image pipeline already built (I2I, FaceID, concept analysis)
- LLM-driven director's notes already working
- Self-hosted — no API costs for video generation (local) or controlled costs (RunPod)

---

## Implementation Phases

### Phase 1: World Builder (reference images → world bible)
- Multi-image upload UI
- Per-image concept analysis (existing pipeline)
- Aggregate into world bible: characters, locations, style
- Character ↔ reference image linking
- **Depends on**: existing I2T pipeline

### Phase 2: Story Writer (world bible + idea → screenplay)
- LLM generates N-scene screenplay from world bible + user prompt
- Structured JSON output per scene (schema above)
- Storyboard UI: view all scenes, edit, reorder, refine
- Per-scene LLM refinement (extend existing notes refinement)
- **Depends on**: Phase 1, Guardian LLM

### Phase 3: Scene-to-Video Generation
- Per-scene prompt assembly (camera + motion + scene + mood)
- Reference image selection per scene (character/location)
- Individual scene generation via existing I2V pipeline
- Preview mode (low-res/short) for rapid iteration
- **Depends on**: Phase 2, existing video gen infrastructure

### Phase 4: Assembly & Export
- ffmpeg-based clip stitching with transitions
- Audio track assembly (ambient + dialogue placeholders)
- Title cards, credits
- Export to MP4, store in MinIO storage
- **Depends on**: Phase 3

### Phase 5: Polish & Advanced Features
- TTS for dialogue (integrate with TTS service)
- Music generation or library integration
- Scene-to-scene consistency via last-frame anchoring
- Parallel cloud generation for speed
- Collaborative story editing
- **Depends on**: Phase 4

---

## Key Design Decisions (to be made)

1. **Story length limit**: How many scenes max? (suggestion: 20 scenes = 5 min movie)
2. **Resolution strategy**: Generate all at same res, or allow per-scene?
3. **Audio handling**: TTS for dialogue? Music generation? Or placeholder slots?
4. **Storage**: Stories stored in DB? Or as JSON files? (suggestion: DB with MinIO for media)
5. **Collaboration**: Single user or multi-user story editing?
6. **Credits cost**: How to price per-scene generation in the credits system?

---

## Related oelala Docs

- [GENERATION_MODES_TREE.md](GENERATION_MODES_TREE.md) — all tested video gen configurations
- [FACE_SYSTEM.md](FACE_SYSTEM.md) — face embedding & IP-Adapter pipeline
- [COMFYUI_INVENTORY.md](COMFYUI_INVENTORY.md) — available models
- [DISTORCH2_MULTI_GPU_SETTINGS.md](DISTORCH2_MULTI_GPU_SETTINGS.md) — multi-GPU video gen
- [ADVANCED_VIDEO.md](ADVANCED_VIDEO.md) — video generation capabilities
- [ROADMAP.md](ROADMAP.md) — overall project roadmap
