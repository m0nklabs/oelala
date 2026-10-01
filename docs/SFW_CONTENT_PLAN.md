# SFW Content Generation Plan

## Objective
Generate 100 diverse SFW videos for the frontpage gallery to welcome guest users.

## Current Status: ✅ Testing Complete

### Phase 1: Find Optimal Settings ✅
- [x] Test single random generation
- [x] Determine best resolution/fps/frames for quality vs speed
- [x] Find reliable prompt variety technique

### Phase 2: Batch Generation (NEXT)
- [x] Create batch script for 100 videos ✅
- [x] Implement diverse prompt generation (10 categories × 10 prompts)
- [ ] Run overnight generation

**Ready to run:**
```bash
python scripts/generate_sfw_batch.py --count 100
```

Estimated: ~6-19 hours at MiniMax-H3 cloud timings (100 × 3.5-11.5 min per 5 sec clip — see `docs/GENERATION_MODES_TREE.md`); retest once the H3 variant is chosen

### Phase 3: Upload & Display
- [ ] Upload all to storage under admin account
- [ ] Mark as SFW + public
- [ ] Verify gallery display for guests

---

## Technical Approach

### Generation Method: Text-to-Image → Image-to-Video
MiniMax-H3 also does pure T2V (`minimax-h3-cloud-t2v` / `minimax-h3-local-t2v`, 24 fps with native audio), so the 2-step pipeline below is for image-controlled starting frames:
1. **T2I**: Generate diverse SFW images with SDXL/Flux
2. **I2V**: Animate with MiniMax-H3 (leading) or LTX-2.3 — the local Wan 2.2 I2V path was retired on 2026-10-01; see `docs/LEGACY.md`

### Prompt Diversity Strategy
Use categories to ensure variety:

| Category | Examples |
|----------|----------|
| Nature | Mountains, oceans, forests, sunsets, aurora |
| Animals | Wildlife, birds, fish, insects in motion |
| Urban | City streets, architecture, neon lights |
| Abstract | Geometric patterns, liquid motion, particles |
| Space | Galaxies, planets, nebulae, stars |
| Weather | Storms, rain, snow, clouds timelapse |
| Water | Waterfalls, waves, underwater, reflections |
| Fire/Light | Flames, fireworks, light rays, candles |
| Plants | Flowers blooming, leaves falling, trees |
| Machines | Clocks, gears, vehicles, technology |

### Video Settings (To Test)

> MiniMax-H3 values, taken from the verified H3 configuration in
> `docs/GENERATION_MODES_TREE.md` (768×1344 @ 124 frames) and the H3 adapters (24 fps,
> megapixel canvas). The Wan-era values that stood here (480p, 41 frames @ 16 fps, local
> DisTorch2) do not apply to H3.

| Setting | Test Value | Notes |
|---------|------------|-------|
| Model | MiniMax-H3 (leading) | FL2VA, joint video+audio, no CFG/negative prompt |
| Resolution | 0.98 MP → native 768×1344 | Template default is 0.4 MP; 0.98 MP is the recommended canvas |
| Frames / fps | 124 frames @ 24 fps | ~5 sec; H3 supports 4-15 sec |
| Compute | Cloud (RunPod `oelala-minimax-h3`, 48GB+ tiers / A40) or local Windows-PC ComfyUI | H3 does not use local DisTorch2 |
| Expected time | ~3.5-11.5 min per clip (cloud) | draft 4 ≈ 205 s, standard 8 ≈ 327 s, official 20 ≈ 689 s |

### SFW Prompt Template
```
[subject] in [setting], [lighting], [style],
cinematic quality, detailed, vibrant colors,
professional photography, safe for work
```

### Animation Prompt Template
```
gentle [motion type], smooth camera movement,
natural motion, fluid animation
```

Motion types: panning, zooming, floating, flowing, drifting, swaying

---

## Test Results

> **Retired 2026-10-01:** Wan 2.2 was retired from the product on 2026-10-01; references to it below are historical. See `docs/LEGACY.md`.

### Test 1: 2026-01-06 (Aurora Borealis)
- **T2I Prompt**: aurora borealis over frozen lake, green and purple lights dancing, masterpiece, highly detailed, professional photography, 8k uhd, cinematic lighting, safe for work
- **T2I Model**: dreamshaperXL_lightningDPMSDE.safetensors
- **T2I Duration**: 42s
- **I2V Prompt**: gentle camera movement, natural motion, cinematic quality, smooth animation, professional video, 4k quality
- **I2V Model**: Wan 2.2 14B Q6_K (DisTorch2 multi-GPU)
- **I2V Settings**: 480x480, 41 frames, 6 steps
- **I2V Duration**: 120s (2 min)
- **Total Pipeline**: ~3 min per video
- **Output Size**: 529KB MP4
- **Quality**: ✅ Good (needs visual review)

### Optimal Settings Found (Wan-era measurement — retest for H3)

> The values below were measured on the retired Wan 2.2 pipeline (16 fps, DisTorch2). They are kept as a record; the current video settings live in the section above.
| Parameter | Value | Notes |
|-----------|-------|-------|
| T2I Model | DreamShaper XL Lightning | Fast, 8 steps |
| T2I Resolution | 576x1024 (9:16) | Matches video aspect |
| I2V Resolution | 576x1024 (9:16) | Mobile-friendly |
| I2V Frames | 81 | 5 sec @ 16fps |
| I2V Steps | 6 | Fast cascade |
| Total Time | ~8 min | Per video |

### Batch Estimate
- 100 videos × 8 min = 800 min = **13.3 hours**
- Can run overnight

---

## Batch Script Location
`scripts/generate_sfw_batch.py` (to be created)

## Output Location
`/home/flip/oelala/generated/sfw_batch/`

## Success Criteria
- [ ] 100 unique SFW videos
- [ ] No NSFW content
- [ ] Visually diverse (10+ categories)
- [ ] Good quality (no major artifacts)
- [ ] All uploaded to storage
- [ ] Visible on guest frontpage
