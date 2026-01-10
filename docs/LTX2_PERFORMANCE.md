# LTX-2 GGUF Performance Report

**Last Updated:** January 2026
**Hardware:** RTX 5060 Ti (16GB) + RTX 3060 (12GB) = 28GB total VRAM

## Model Variants

### 1. LTX-2 19B Distilled (Video Only)

- **File:** `ltx-2-19b-distilled_Q4_K_M.gguf` (12.03 GB)
- **Type:** Distilled, video-only model (faster inference)
- **Use case:** Fast video generation without audio

### 2. LTX-2 19B Dev (Audio + Video) ⭐ NEW

- **File:** `ltx-2-19b-dev-Q4_K_M.gguf` (12 GB)
- **Type:** Full development model with native audio support
- **Audio VAE:** `ltx2_audio_vae.safetensors` (208 MB, combined VAE+Vocoder)
- **Use case:** Video generation with synchronized native audio

## Configuration

- **Text Encoder:** Gemma-3 12B on CPU (~24GB RAM, ~55 sec encode time)
- **Resolution:** 768x512 (or 512x320 for speed)
- **FPS:** 25
- **DisTorch2 Allocation:** Not used for LTX-2 (single GPU sufficient)

---

## Video-Only Benchmark (Distilled Model)

| Frames | Video Duration | Generation Time | VRAM Peak | Notes |
|--------|---------------|-----------------|-----------|-------|
| 17 | 0.68 sec | ~29 sec | ~11.2 GB | Baseline test |
| 49 | 1.96 sec | ~62 sec | ~13 GB | Standard quality |
| 81 | 3.24 sec | ~95 sec | ~14.6 GB | Default setting |
| 97 | 3.88 sec | ~111 sec | ~15.2 GB | Good balance |
| 129 | 5.16 sec | ~145 sec | ~15.6 GB | Tiled VAE fallback |
| 161 | 6.44 sec | ~180 sec | ~15.7 GB | Long clip |
| 193 | 7.72 sec | ~215 sec | ~15.8 GB | Extended |
| 225 | 9.00 sec | ~250 sec | ~15.8 GB | Very long |
| 257 | 10.28 sec | ~285 sec | ~15.8 GB | Near max |
| **281** | **11.24 sec** | **~310 sec** | **~15.8 GB** | **Maximum tested** |

---

## Audio+Video Generation (Dev Model) ⭐ NEW

### First Successful Test

- **Resolution:** 768x512
- **Frames:** 97 (3.88 seconds @ 25fps)
- **Steps:** 30
- **Audio:** Native 24kHz stereo AAC
- **Output Size:** 705 KB (H.264 + AAC)
- **Example:** `examples/ltx2_audio_00001-audio.mp4`

### Audio Components Required

1. **Audio VAE:** Combined audio encoder/decoder (208 MB)
   - Contains CausalAudioAutoencoder + Vocoder
   - Sample rate: 16kHz input → 24kHz output (vocoder upsamples)
   - Stereo (2 channel) support

2. **Text Encoder:** Gemma 3 12B with audio embeddings connector
   - Must detect AV model type automatically
   - Loads additional audio projection weights from `connectors/`

3. **Empty Audio Latent:** Create matching audio latents for video frames
   - Must match frame count and frame rate exactly

### Audio+Video Workflow Pattern

```
┌──────────────────┐     ┌──────────────────┐
│ EmptyVideoLatent │     │ EmptyAudioLatent │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         └──────────┬─────────────┘
                    │
         ┌──────────▼──────────┐
         │  LTXVConcatAVLatent │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │ SamplerCustomAdvanced│ ◄── CFGGuider with LTXAV model
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │LTXVSeparateAVLatent │
         └────────┬─┬──────────┘
                  │ │
    ┌─────────────┘ └─────────────┐
    │                             │
┌───▼───┐                    ┌────▼────┐
│VAEDecode│                  │AudioVAE │
│ (video) │                  │ Decode  │
└────┬────┘                  └────┬────┘
     │                            │
     └────────────┬───────────────┘
                  │
         ┌────────▼────────┐
         │ VHS_VideoCombine│
         └─────────────────┘
```

**Important:** You MUST use `SamplerCustomAdvanced` with pre-combined AV latent!
The standard `LTXVBaseSampler` only outputs video latents, not combined audio+video.

---

## Key Findings

### 1. CPU Gemma Encoding Works
The `LTXVCPUGemmaEncode` node offloads text encoding to CPU RAM, freeing GPU VRAM. Initial encode takes ~55 seconds, cached generations are near-instant.

### 2. VRAM Ceiling
Peak VRAM usage stabilizes around 15.8GB regardless of frame count, suggesting efficient memory management for longer sequences.

### 3. Audio Model Detection
ComfyUI detects LTXAV models by checking for `audio_adaln_single.linear.weight` tensor in the GGUF. The Dev model contains 2229 audio-specific tensors.

### 4. Audio VAE Preparation Required
The HuggingFace audio components need to be combined into a single checkpoint with ComfyUI-compatible metadata. See preparation section below.

---

## Preparing Audio VAE Checkpoint

The Audio VAE from HuggingFace needs to be combined into a ComfyUI-compatible format:

```python
# Combine audio_vae.safetensors + vocoder into single checkpoint
from safetensors.torch import load_file, save_file
import json

# Load components
audio_vae_sd = load_file("models/vae/ltx2_audio/audio_vae.safetensors")
vocoder_sd = load_file("models/vocoder/diffusion_pytorch_model.safetensors")

# Combine with prefixes
combined_sd = {}
for k, v in audio_vae_sd.items():
    combined_sd[f"audio_vae.{k}"] = v
for k, v in vocoder_sd.items():
    combined_sd[f"vocoder.{k}"] = v

# Create ComfyUI-compatible metadata
# See ComfyUI/ltx2_audio_test.py for full config mapping
metadata = {"config": json.dumps({
    "model": {
        "params": {
            "encoder": {"ch": 256, "out_ch": 256, ...},
            "decoder": {"ch": 256, "out_ch": 2, ...}
        }
    }
})}

save_file(combined_sd, "models/checkpoints/ltx2_audio_vae.safetensors", metadata=metadata)
```

---

## Workflow Files

- **Video-only API:** `workflows/ltx2_cpu_gemma_api.json`
- **Audio+Video Test:** `ComfyUI/ltx2_audio_test.py`
- **Output Directory:** `ComfyUI/output/`

## Recommended Settings

| Use Case | Frames | Duration | Notes |
|----------|--------|----------|-------|
| Short clips (social) | 81 | 3.24 sec | Fast, reliable |
| Medium clips | 129 | 5.16 sec | Good balance |
| Long clips | 193 | 7.72 sec | Tiled VAE |
| Maximum | 281 | 11.24 sec | Use with caution |

## Custom Nodes Required

1. `ComfyUI-GGUF` - For GGUF model loading
2. `ComfyUI-LTX-Video` - LTX-2 sampling, audio nodes
3. `ComfyUI-VideoHelperSuite` - VHS_VideoCombine for audio muxing

## Example Prompts

```
"Professional ballet dancer performing elegant pirouette in grand theater, dramatic stage lighting, slow motion, 4k cinematic"

"Epic motorcycle chase through neon city streets at night, rain reflections, headlights, cinematic action movie style, 4k"
```

## Negative Prompt

```
blurry, low quality, distorted, watermark, static, deformed
```
