# LTX-2 GGUF Performance Report

**Date:** January 10, 2026  
**Hardware:** RTX 5060 Ti (16GB) + RTX 3060 (12GB) = 28GB total VRAM

## Configuration

- **Model:** `ltx-video-2b-v0.9.5-Q4_K_M.gguf` (~12.7GB VRAM)
- **Text Encoder:** Gemma-3 12B on CPU (~24GB RAM, ~55 sec encode time)
- **Resolution:** 512x320 (native LTX-2 aspect)
- **FPS:** 25
- **DisTorch2 Allocation:** Not used for LTX-2 (single GPU sufficient)

## Performance Results

| Frames | Video Duration | Generation Time | VRAM Peak | Notes |
|--------|---------------|-----------------|-----------|-------|
| 17 | 0.68 sec | ~165 sec | ~14 GB | Baseline test |
| 49 | 1.96 sec | ~146 sec | ~15 GB | Standard quality |
| 81 | 3.24 sec | ~150 sec | ~15.8 GB | Default setting |
| 121 | 4.84 sec | ~163 sec | ~15.8 GB | Good balance |
| 161 | 6.44 sec | ~172 sec | ~15.8 GB | Tiled VAE fallback |
| 201 | 8.04 sec | ~118 sec | ~15.8 GB | Gemma cached |
| 241 | 9.64 sec | ~60 sec | ~15.8 GB | Super fast (cached) |
| **281** | **11.24 sec** | **~194 sec** | **~15.8 GB** | **Maximum tested** |

## Key Findings

### 1. CPU Gemma Encoding Works
The custom `GemmaEncoderCPU` node successfully offloads text encoding to CPU RAM, freeing GPU VRAM for the diffusion model. Initial encode takes ~55 seconds, but subsequent generations with same text are near-instant due to caching.

### 2. Tiled VAE Fallback
For frame counts above ~160, the VAE decoder automatically falls back to tiled decoding, preventing OOM while maintaining quality.

### 3. VRAM Ceiling
Peak VRAM usage stabilizes around 15.8GB regardless of frame count, suggesting the model efficiently handles longer sequences without linear VRAM growth.

### 4. Generation Time Scaling
- Cold start (new prompt): ~165-194 seconds regardless of frame count
- Warm start (cached Gemma): 60-120 seconds
- The Gemma encoding is the main bottleneck, not the diffusion steps

## Workflow Files

- **API Workflow:** `workflows/ltx2_cpu_gemma_api.json`
- **Output Directory:** `ComfyUI/output/ltx2_cpu_gemma_*.mp4`

## Recommended Settings

For production use:
- **Short clips (social media):** 81 frames = 3.24 sec
- **Medium clips:** 161 frames = 6.44 sec  
- **Long clips:** 241 frames = 9.64 sec
- **Maximum:** 281 frames = 11.24 sec (use with caution)

## Custom Nodes Required

1. `ComfyUI-GGUF` - For GGUF model loading
2. `ComfyUI-LTX-Video` - LTX-2 sampling nodes
3. Custom `GemmaEncoderCPU` node - CPU-based text encoding

## Example Prompts Tested

```
"Professional ballet dancer performing elegant pirouette in grand theater, dramatic stage lighting, slow motion, 4k cinematic"

"Epic motorcycle chase through neon city streets at night, rain reflections, headlights, cinematic action movie style, 4k"

"Seductive fashion show runway, models walking in designer outfits, dramatic lighting, slow motion camera pans, backstage atmosphere, 4k cinematic"
```

## Negative Prompt

```
blurry, low quality, distorted, watermark, static, deformed
```
