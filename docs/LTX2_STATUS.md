# LTX-2 Video Generation Status

**Last Updated**: 2026-01-09
**Status**: ✅ GGUF Setup Ready! (via Kijai's GGUF quantizations)

---

## Summary

LTX-2 is Lightricks' latest audio-video generation model. Thanks to **Kijai's GGUF quantizations** released Jan 9, 2026, LTX-2 now fits on our 28GB multi-GPU setup!

## VRAM Requirements (Updated with GGUF)

| Configuration | Model | Text Encoder | Total VRAM | Status |
|---------------|-------|--------------|------------|--------|
| ❌ Original | LTX-2 FP8 (26GB) | Gemma 3 BF16 (23GB) | ~49GB | Too large |
| ✅ **GGUF Q4** | LTX-2 Q4_K_M (12.7GB) | Gemma FP4 (7.8GB) | **~25GB** | **Works!** |
| ⚠️ GGUF Q6 | LTX-2 Q6_K (15.9GB) | Gemma FP4 (7.8GB) | ~28GB | Borderline |

**Available VRAM**: 28GB (12GB + 16GB)

## Installed Components (GGUF Ready)

### ComfyUI-GGUF (PR #399 Branch)
```
/home/flip/oelala/ComfyUI/custom_nodes/ComfyUI-GGUF/
Branch: pr-399 (LTX-2 support)
```
Provides nodes:
- `LoaderGGUF` - Load GGUF diffusion models including LTX-2
- `LoaderGGUFAdvanced` - Advanced GGUF loading options
- `UnetLoaderGGUFAdvancedDisTorch2MultiGPU` - Multi-GPU GGUF loading

### ComfyUI-LTXVideo (Working)
```
/home/flip/oelala/ComfyUI/custom_nodes/ComfyUI-LTXVideo/
```
Provides nodes:
- `LTXVGemmaCLIPModelLoader` - Loads Gemma 3 via HuggingFace
- `LTXVGemmaEnhancePrompt` - Prompt enhancement
- Various LTX-2 specific nodes

### Models Installed
| File | Location | Size |
|------|----------|------|
| **`ltx-2-19b-distilled_Q4_K_M.gguf`** | diffusion_models/ | **12 GB** ✅ |
| **`gemma_3_12B_it_nvfp4.safetensors`** | text_encoders/ | **7.8 GB** ✅ |
| **`ltx-2-19b-embeddings_connector_bf16.safetensors`** | text_encoders/ | **2.7 GB** ✅ |
| **`LTX2_video_vae_bf16.safetensors`** | vae/ | **2.4 GB** ✅ |
| `ltx-2-19b-distilled-fp8.safetensors` | checkpoints/ | 26GB (legacy) |

## Workflow

Use Kijai's workflow for GGUF LTX-2:
- Workflow file: `/home/flip/oelala/workflows/ltx2_gguf_kijai.json`
- Source: https://huggingface.co/Kijai/LTXV2_comfy/discussions/2

### Key Nodes Setup
1. `LoaderGGUFAdvanced` → Load `ltx-2-19b-distilled_Q4_K_M.gguf`
2. Text encoder → Load `gemma_3_12B_it_nvfp4.safetensors` + embeddings connector
3. VAE → Load `LTX2_video_vae_bf16.safetensors`

## Resources

- **Kijai GGUF Models**: https://huggingface.co/Kijai/LTXV2_comfy
- **Gemma FP4**: https://huggingface.co/GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn
- **ComfyUI-GGUF PR#399**: https://github.com/city96/ComfyUI-GGUF/pull/399
- **Reddit Thread**: https://reddit.com/r/StableDiffusion/comments/1q8590s

## Legacy Technical Blockers (Solved)

These issues existed with the original full-precision models but are bypassed with GGUF:

1. **Tokenizer Format Mismatch** - GGUF workflow uses direct safetensors loading
2. **device_map Conflicts** - GGUF doesn't need HuggingFace accelerate
3. **VRAM Size** - Q4 quantization reduces 49GB to 25GB

---

## References

- [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) - Official HuggingFace repo
- [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) - Custom nodes
- [LTX-2 Paper](https://arxiv.org/abs/2601.03233) - Technical details
- [Kijai GGUF Release (Jan 9, 2026)](https://reddit.com/r/StableDiffusion/comments/1q8590s)
