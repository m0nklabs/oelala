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

Estimated: ~3.5 hours (100 × 124s)

### Phase 3: Upload & Display
- [ ] Upload all to storage under admin account
- [ ] Mark as SFW + public
- [ ] Verify gallery display for guests

---

## Technical Approach

### Generation Method: Text-to-Image → Image-to-Video
Since we don't have pure T2V, we'll use a 2-step pipeline:
1. **T2I**: Generate diverse SFW images with SDXL/Flux
2. **I2V**: Animate with Wan 2.2 14B Q6_K

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

| Setting | Test Value | Notes |
|---------|------------|-------|
| Resolution | 480p (848x480) | Balance quality/speed |
| Frames | 41 | ~2.5 sec @ 16fps |
| Model | Wan 2.2 14B Q6_K | Best quality available |
| GPU | DisTorch2 multi-GPU | cuda:0+cuda:1 |

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

### Optimal Settings Found
| Parameter | Value | Notes |
|-----------|-------|-------|
| T2I Model | DreamShaper XL Lightning | Fast, 8 steps |
| T2I Resolution | 1024x1024 | Source for I2V |
| I2V Resolution | 480x480 | Good quality, fast |
| I2V Frames | 41 | ~2.5 sec @ 16fps |
| I2V Steps | 6 | Fast cascade |
| Total Time | ~3 min | Per video |

### Batch Estimate
- 100 videos × 3 min = 300 min = **5 hours**
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
