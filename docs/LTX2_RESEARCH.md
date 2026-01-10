# LTX-2 Research: Audio + Video Generation

> Research date: January 2026
> Status: **Very Promising** - Native audio+video in one model!

## NSFW Status: ✅ NOT EXPLICITLY RESTRICTED

After reviewing the full LTX-2 Community License Agreement (Attachment A: Use Restrictions), **there is NO explicit prohibition on adult/NSFW content**.

**Restricted uses are:**
- Exploiting/harming minors
- Deepfakes without consent
- Harassment/defamation
- Discrimination
- Military/weapons applications
- Malware generation
- False information/disinformation

**NOT mentioned:** Adult content, nudity, explicit material (unlike some Stability AI licenses).

⚠️ **Disclaimer**: This is research, not legal advice. Use responsibly.

## 3GB VRAM Claim

YouTube claims 3GB VRAM - official docs say 32GB+ recommended. The truth is probably:
- 3GB possible with **extreme CPU offloading** (slow but works)
- Uses `low_vram_loaders.py` for sequential model loading
- FP8 quantization reduces memory
- Tiled VAE decode for large outputs

**We need to test this ourselves!**

## Executive Summary

LTX-2 is Lightricks' next-generation video foundation model with **native synchronized audio+video generation**. This is exactly what we need for SFW content creation.

### Key Differentiators from WAN 2.2

| Feature | LTX-2 | WAN 2.2 (Current) |
|---------|-------|-------------------|
| Audio Generation | ✅ Native, synchronized | ❌ Requires separate model |
| Model Size | 19B parameters | 14B parameters |
| Max Duration | 10 seconds native | 5-15s (with limits) |
| Resolution | Native 4K @ 50fps | 720p-1080p practical |
| FP8 Support | ✅ Yes | ✅ Yes (GGUF) |
| ComfyUI | ✅ Official nodes | ✅ DisTorch2 |

## Model Specifications

### Available Weights

| Model | Size | Purpose | VRAM Estimate |
|-------|------|---------|---------------|
| `ltx-2-19b-dev` | ~38GB | Full quality | ~48GB (multi-GPU required) |
| `ltx-2-19b-dev-fp8` | ~19GB | Quantized full | ~24GB |
| `ltx-2-19b-distilled` | ~38GB | Fast inference (8 steps) | ~48GB |
| `ltx-2-19b-distilled-fp8` | ~19GB | **Recommended** | ~24GB |

### Additional Components

- **Spatial Upscaler**: `ltx-2-spatial-upscaler-x2-1.0` - 2x upscaling
- **Temporal Upscaler**: `ltx-2-temporal-upscaler-x2-1.0` - Frame interpolation
- **Distilled LoRA**: `ltx-2-19b-distilled-lora-384` - For two-stage pipelines
- **Text Encoder**: Gemma 3 12B (google/gemma-3-12b-it-qat-q4_0-unquantized)

### Control LoRAs

| LoRA | Purpose |
|------|---------|
| IC-LoRA-Canny-Control | Edge-guided generation |
| IC-LoRA-Depth-Control | Depth-guided generation |
| IC-LoRA-Detailer | Upscaling/detailing |
| IC-LoRA-Pose-Control | Pose-guided generation |
| LoRA-Camera-Control-* | Dolly In/Out/Left/Right, Jib Up/Down, Static |

## Pipeline Options

### For Speed: DistilledPipeline
- 8 predefined sigmas (steps)
- Fastest inference
- Slight quality reduction
- **Best for iterative work**

### For Quality: TI2VidTwoStagesPipeline
- Two-stage: generate low-res, then upscale
- Production-quality output
- Requires spatial upscaler
- **Best for final renders**

### For Control: ICLoraPipeline
- Video-to-video transformations
- Image-to-video with control
- Uses control LoRAs

## Hardware Requirements

### Our Hardware: RTX 5060 Ti (16GB) + RTX 3060 (12GB) = 28GB

**Assessment**:
- ⚠️ FP8 distilled model (~24GB) is **borderline**
- ✅ Would need CPU offload similar to WAN 2.2
- ⚠️ May need smaller resolutions than 4K
- ✅ DisTorch2 might work if adapted

### Optimization Options

1. **FP8 Transformer**: `--enable-fp8` reduces memory
2. **xFormers/Flash Attention**: Memory-efficient attention
3. **Gradient Estimation**: Reduce steps from 40 → 20-30
4. **Single-stage Pipeline**: Skip upscaling for faster/smaller gen

## ComfyUI Integration

Official support via: https://github.com/Lightricks/ComfyUI-LTXVideo

This is the **same repo** we already have for LTX-Video! It should support LTX-2.

### Existing Custom Node in ComfyUI

We already have `ComfyUI-LTXVideo` installed. Need to verify if it's updated for LTX-2.

## Audio Capabilities

### What LTX-2 Generates

- **Synchronized audio + video** in one pass
- Motion, dialogue, ambience, and music flowing together
- Native audio track embedded in output
- Up to 10 seconds with audio

### Audio Tasks Supported

Based on HuggingFace tags:
- `audio-to-video` - Generate video from audio
- `text-to-audio` - Generate audio from text
- `video-to-audio` - Add audio to existing video
- `text-to-audio-video` - Generate both from text
- `image-to-audio-video` - Image input, audio+video output

## SFW Use Case Fit

### Perfect For:
- ✅ Product videos with ambient sounds
- ✅ Nature/travel content with environmental audio
- ✅ Educational content with narration-ready audio
- ✅ Social media clips with music
- ✅ Logo animations with sound effects

### Current WAN 2.2 Workflow:
1. Generate video (WAN 2.2)
2. Separately generate audio (different model)
3. Sync in post-processing

### LTX-2 Workflow:
1. Generate video + audio together ✅

## Implementation Plan

### Phase 1: Assessment
1. [ ] Check if ComfyUI-LTXVideo supports LTX-2
2. [ ] Download `ltx-2-19b-distilled-fp8` (~19GB)
3. [ ] Download Gemma 3 text encoder
4. [ ] Test with minimal workflow

### Phase 2: Multi-GPU Setup
1. [ ] Verify DisTorch2 compatibility
2. [ ] Create allocation string for 28GB VRAM
3. [ ] Test with CPU offload if needed

### Phase 3: Workflow Integration
1. [ ] Create API workflow like WAN 2.2
2. [ ] Add to oelala backend
3. [ ] Frontend integration

## Links

- **GitHub**: https://github.com/Lightricks/LTX-2
- **HuggingFace**: https://huggingface.co/Lightricks/LTX-2
- **Paper**: https://videos.ltx.io/LTX-2/grants/LTX_2_Technical_Report_compressed.pdf
- **Demo**: https://app.ltx.studio/ltx-2-playground/i2v
- **ComfyUI Nodes**: https://github.com/Lightricks/ComfyUI-LTXVideo
- **Docs**: https://docs.ltx.video

## Next Steps

1. **Download FP8 distilled model** (smallest practical option)
2. **Check ComfyUI node compatibility** for LTX-2
3. **Run test generation** with audio enabled
4. **Benchmark VRAM usage** on our hardware
5. **Create workflow** for I2V with audio

---

## Comparison: Recommendations

| Scenario | Best Choice |
|----------|-------------|
| Long video (15+ sec) | WAN 2.2 (up to 60s) |
| Video + Audio needed | **LTX-2** |
| Lowest VRAM | WAN 2.2 Q6_K |
| Highest quality | LTX-2 19B dev |
| Fastest iteration | LTX-2 distilled |
| Control/pose | Both have IC-LoRAs |
