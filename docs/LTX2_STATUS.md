# LTX-2 Video Generation Status

**Last Updated**: 2026-01-09  
**Status**: ❌ Not Practical (VRAM constraints)

---

## Summary

LTX-2 is Lightricks' latest video generation model. On our 28GB multi-GPU setup (RTX 3060 12GB + RTX 5060 Ti 16GB), it is **not currently practical** due to VRAM requirements.

## VRAM Requirements

| Configuration | Model | Text Encoder | Total VRAM |
|---------------|-------|--------------|------------|
| Recommended | LTX-2 19B FP8 (26GB) | Gemma 3 12B (23GB) | **~49GB** ❌ |
| Minimal | LTX-2 19B FP8 (26GB) | UMT5 FP8 (5.6GB) | **~32GB** ❌ |
| Future | LTX-2 19B FP4 (20GB) | UMT5 FP8 (5.6GB) | **~26GB** ⚠️ |

**Available VRAM**: 28GB (12GB + 16GB)

## Technical Blockers

### 1. Tokenizer Format Mismatch
- **Native ComfyUI loaders** (`LTXAVTextEncoderLoader`) expect `spiece_model` tensor embedded in safetensors
- **HuggingFace Gemma 3** uses separate `tokenizer.model` file
- Error: `ValueError: invalid tokenizer`

### 2. device_map Conflicts
- HuggingFace `device_map="auto"` distributes model across GPUs/CPU
- ComfyUI's `model_patcher` calls `model.to(device)` which breaks distributed models
- Error: `NotImplementedError: Cannot copy out of meta tensor`

### 3. Pure VRAM Size
- Even with perfect memory management, 49GB doesn't fit in 28GB
- CPU offloading via accelerate works for loading but OOMs during inference

## Installed Components

### ComfyUI-LTXVideo (Working)
```
/home/flip/oelala/ComfyUI/custom_nodes/ComfyUI-LTXVideo/
```
Provides nodes:
- `LTXVGemmaCLIPModelLoader` - Loads Gemma 3 via HuggingFace
- `LTXVGemmaEnhancePrompt` - Prompt enhancement
- `GemmaLoader`, `GemmaTextEncode` - Direct Gemma access

### Models Available
| File | Location | Size |
|------|----------|------|
| `ltx-2-19b-distilled-fp8.safetensors` | checkpoints/ | 26GB |
| `gemma-3-12b-it-qat-q4_0-unquantized/` | text_encoders/ | 23GB |
| `umt5_xxl_fp8_e4m3fn.safetensors` | text_encoders/ | 5.6GB |

## Future Options

### Option 1: Smaller Model (Recommended)
Download `ltx-2-19b-dev-fp4.safetensors` (20GB) from Hugging Face:
```bash
huggingface-cli download Lightricks/LTX-2 ltx-2-19b-dev-fp4.safetensors \
  --local-dir /home/flip/oelala/ComfyUI/models/checkpoints
```
Combined with UMT5 FP8 (5.6GB) = ~26GB, might fit with aggressive offloading.

### Option 2: Use Diffusers Directly
Bypass ComfyUI memory management conflicts:
```python
from diffusers import LTX2Pipeline
pipe = LTX2Pipeline.from_pretrained(
    "Lightricks/LTX-2",
    torch_dtype=torch.bfloat16,
    device_map="balanced"
)
```

### Option 3: Wait for Smaller Models
Lightricks may release:
- Smaller distilled versions
- Better quantized models
- Optimized text encoders

### Option 4: Hardware Upgrade
- Single 48GB GPU (RTX 6000 Ada, A6000)
- Or 2x 24GB GPUs (RTX 4090)

## Current Recommendation

**Use Wan 2.2 14B** for video generation. It works reliably with DisTorch2 multi-GPU:

| Resolution | Max Frames | VRAM Usage |
|------------|------------|------------|
| 480p | 81 frames | ~24GB ✅ |
| 720p | 41 frames | ~26GB ✅ |
| 1080p | 17-25 frames | ~27GB ⚠️ |

See [COMFYUI_INVENTORY.md](COMFYUI_INVENTORY.md) for full details.

## Test Script

A test script is available at `scripts/test_ltx2_umt5.py` for future testing when smaller models become available.

---

## References

- [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) - Official HuggingFace repo
- [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) - Custom nodes
- [LTX-2 Paper](https://arxiv.org/abs/2601.03233) - Technical details
