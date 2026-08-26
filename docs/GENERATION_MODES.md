# Generation Modes & Model Inventory

This document provides a comprehensive overview of all available generation modes per tool type, and the sub-models (VAE, text encoders, GGUF models) that can be combined for video/image generation.

**Hardware**: RTX 5060 Ti 16GB (cuda:0) + RTX 3060 12GB (cuda:1) = 28GB total VRAM

---

## Table of Contents

1. [Image-to-Video (I2V)](#image-to-video-i2v)
2. [Text-to-Video (T2V)](#text-to-video-t2v)
3. [Text-to-Image (T2I)](#text-to-image-t2i)
4. [Sub-Models Reference](#sub-models-reference)
5. [DisTorch2 Multi-GPU Configuration](#distorch2-multi-gpu-configuration)
6. [Workflow Compatibility Matrix](#workflow-compatibility-matrix)

---

## Image-to-Video (I2V)

### Available Generation Modes

| Mode | Model | Workflow File | VRAM Required | Notes |
|------|-------|---------------|---------------|-------|
| `standard` | Wan2.2 14B Q6_K DisTorch2 | `ImageToVideo/wan22_i2v_distorch2_api.json` | ~24GB | Default, high quality dual-noise |
| `nsfw_lora` | Wan2.2 Enhanced NSFW | `ImageToVideo/wan22_i2v_ltx2_audio_api.json` | ~24GB | Pre-configured NSFW LoRAs |

### I2V Model Components

| Component | Model File | Size | Location |
|-----------|------------|------|----------|
| **High Noise GGUF** | `wan2.2_i2v_high_noise_14B_Q6_K.gguf` | 12GB | `models/unet/` |
| **Low Noise GGUF** | `wan2.2_i2v_low_noise_14B_Q6_K.gguf` | 12GB | `models/unet/` |
| **Text Encoder** | `umt5-xxl-enc-bf16.safetensors` | 11GB | `models/text_encoders/` |
| **CLIP Vision** | `clip_vision_h.safetensors` | 2.5GB | `models/clip_vision/` |
| **VAE** | `wan_2.1_vae.safetensors` | 242MB | `models/vae/` |

### I2V Alternative Models (Untested/Experimental)

| Model | File | Quantization | Notes |
|-------|------|--------------|-------|
| Lightning High | `Wan22-I2V_A14B-Lightning-H-Q6_K.gguf` | Q6_K | Faster, may need fewer steps |
| Lightning Low | `Wan22-I2V_A14B-Lightning-L-Q6_K.gguf` | Q6_K | Faster, may need fewer steps |
| Enhanced NSFW High | `wan22EnhancedNSFW_V2_Q6K_HIGH.gguf` | Q6_K | NSFW optimized |
| Enhanced NSFW Low | `wan22EnhancedNSFW_V2_Q6K_LOW.gguf` | Q6_K | NSFW optimized |

> **Removed in cleanup** (unused/duplicated, freed ~98GB): `smoothMixWan22GGUF_high/lowQ6K.gguf`,
> `wan22EnhancedNSFWCameraPrompt_nsfwV2Q6KH/L.gguf`, `LTX-2-dev-Q2_K.gguf`, and the local LTX-2
> 19B set (`ltx-2-19b-dev-Q4_K_M`, `ltx-2-19b-distilled_Q4_K_M`, `ltx-2-19b-distilled-fp8`,
> `ltx-2-19b-embeddings_connector_bf16`, `LTX2_video_vae_bf16`, `ltx2_audio_vae`).

---

## Text-to-Video (T2V)

### Available Generation Modes

| Mode | Model | Workflow File | VRAM Required | Max Frames | Notes |
|------|-------|---------------|---------------|------------|-------|
| `wan22` | Wan2.2 14B | Built-in builder | ~24GB | 81 | T2I → I2V pipeline |
| `ltx2` | LTX-2.3 22B Distilled | Cloud (RunPod) builder | Cloud 80GB | 97 | Cloud-only, fast 8-step |
| `minimax_h3` | MiniMax-H3 FL2VA 22B | Cloud (RunPod) builder | Cloud 80GB | 362 | Cloud T2V+audio |
| `minimax_h3_local` | MiniMax-H3 FL2VA 22B | `build_local_minimax_h3_t2v_workflow` | ~16GB (Windows PC) | 362 | **Lokaal** op Windows-PC ComfyUI (int8 set) |

### T2V Model Components

#### Wan2.2 T2V Components
| Component | Model File | Size | Location |
|-----------|------------|------|----------|
| **High Noise FP8** | `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` | 14GB | `models/unet/` |
| **Low Noise FP8** | `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` | 14GB | `models/unet/` |
| **Text Encoder** | `umt5-xxl-enc-bf16.safetensors` | 11GB | `models/text_encoders/` |
| **VAE** | `wan_2.1_vae.safetensors` | 242MB | `models/vae/` |

#### MiniMax-H3 Local (Windows PC ComfyUI) Components
| Component | Model File | Location (Windows PC) |
|-----------|------------|----------------------|
| **Diffusion** | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `diffusion_models/` |
| **Text Encoder** | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | `text_encoders/` |
| **Video VAE** | `minimax_h3_video_vae_fp16.safetensors` | `vae/` |
| **Audio VAE** | `minimax_h3_audio_vae_fp32.safetensors` | `vae/` |

Lokaal draait MiniMax-H3 op een **tweede ComfyUI-server** (remote compute node, bijv. de
Windows-PC) — geconfigureerd via de compute backend inventory (`COMPUTE_NODE_{n}_HOST` in `.env`
als fresh-install fallback; Admin panel → Compute als bron van waarheid). De int8_convrot
text-encoder past in 16GB VRAM
(zie `workflows/README_MiniMax_H3_workflow.md` en `scripts/download_minimax_h3.*`).

### T2V Alternative Models (Untested/Experimental)

> Lokale LTX-2 19B-modellen zijn verwijderd (LTX-2.3 draait nu cloud-only).
> Verwijderd: `ltx-2-19b-dev-Q4_K_M.gguf`, `ltx-2-19b-distilled_Q4_K_M.gguf`,
> `ltx-2-19b-distilled-fp8.safetensors`, `LTX-2-dev-Q2_K.gguf`,
> `ltx-2-19b-embeddings_connector_bf16.safetensors`, `LTX2_video_vae_bf16.safetensors`.

---

## Text-to-Image (T2I)

### Available Model Categories

| Category | Models | Typical Use |
|----------|--------|-------------|
| `krea2` | Krea 2 Turbo (INT8 ConvRot) | Fast, expressive, 8 steps |
| `flux` | Flux.1 Dev FP8 + NSFW fine-tunes | High quality / NSFW photoreal |
| `flux2` | Flux.2 Dev (GGUF Q4, multi-GPU) | Nieuwste Flux, 32B, meest kwaliteit |
| `sdxl` | SDXL-Pony (Pony V6, CyberRealistic Pony, Reapony) | NSFW / stylized, LoRA-ecosysteem |

### T2I Checkpoints Available

| Model | File | Architecture | VRAM |
|-------|------|--------------|------|
| Krea 2 Turbo | `krea2_turbo_int8_convrot.safetensors` | Krea 2 (12.9B DiT, INT8) | ~10-12GB |
| CyberRealistic Pony | `CyberRealistic_Pony_v14.1_FP16.safetensors` | SDXL Pony | 6.5GB |
| Flux.1 Dev FP8 | `flux1-dev-fp8.safetensors` | Flux | 17GB |
| Flux.2 Dev (GGUF Q4) | `unet/flux2-dev-Q4_K_M.gguf` | Flux 2 (32B, multi-GPU) | ~19GB (2x GPU + CPU) |
| Pony Diffusion V6 | `ponyDiffusionV6XL_v6StartWithThisOne.safetensors` | SDXL Pony | 6.5GB |
| Reapony V9 | `reapony_v90.safetensors` | SDXL Pony | 6.5GB |
| Fluxed Up (NSFW) | `fluxedUpFluxNSFW_51FP8.safetensors` | Flux (diffusion) | ~12GB |
| Persephone (NSFW) | `persephoneFluxNSFWSFW_11FP8.safetensors` | Flux (diffusion) | ~12GB |

**Krea 2 notes**: text encoder = Qwen3-VL-4B (`qwen3vl_4b_bf16.safetensors`, CLIPLoader type `krea2`), VAE = `qwen_image_vae.safetensors`, CFG 1.0, 8 steps, euler/simple. ComfyUI ≥ 0.27 (INT8 ConvRot fix; v0.33.x tested). License: Krea 2 Community License.

**Flux 2 notes**: GGUF UNet verdeeld over beide GPU's via DisTorch2 (`cuda:1,8gb;cuda:0,4gb;cpu,*`), text encoder = Mistral3-small FP8 (`mistral_3_small_flux2_fp8.safetensors`, CLIPLoader type `flux2`, device=cpu), VAE = `flux2-vae.safetensors`. Geen negatieve prompt (guidance i.p.v. CFG). Getest: 1024×1024 ~297s. License: FLUX [dev] Non-Commercial.

---

## Sub-Models Reference

### Text Encoders

| Encoder | File | Size | Used By |
|---------|------|------|---------|
| UMT5-XXL BF16 | `umt5-xxl-enc-bf16.safetensors` | 11GB | Wan2.2 I2V/T2V |
| UMT5-XXL FP8 | `umt5_xxl_fp8_e4m3fn.safetensors` | 5.7GB | Wan2.2 (low VRAM) |
| Gemma 3 12B QAT | `gemma-3-12b-it-qat-q4_0-unquantized/` | 8GB | LTX-2 |
| Gemma 3 12B GGUF | `gemma-3-12b-it-q4_0.gguf` | 8GB | LTX-2 (alt) |
| Qwen 2.5 VL 7B | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | 9.4GB | Qwen vision |

### VAE Models

| VAE | File | Size | Used By |
|-----|------|------|---------|
| Wan 2.1 VAE | `wan_2.1_vae.safetensors` | 242MB | Wan2.2 I2V/T2V |
| LTX-2 VAE | `LTX2_video_vae_bf16.safetensors` | 2.5GB | LTX-2 |
| SDXL VAE | `sdxl_vae.safetensors` | 335MB | SDXL checkpoints |
| Flux AE | `ae.safetensors` | 335MB | Flux |
| Qwen Image VAE | `qwen_image_vae.safetensors` | 254MB | Qwen vision |

### CLIP Vision Models

| Model | Location | Used By |
|-------|----------|---------|
| CLIP Vision H | `models/clip_vision/` | Wan2.2 I2V |
| SigLIP | `models/clip_vision/` | Alternative vision |

---

## DisTorch2 Multi-GPU Configuration

### Default Allocation String
```
cuda:1,12gb;cuda:0,16gb
```

### Per-Model Recommended Allocations

| Model Type | Allocation | Notes |
|------------|------------|-------|
| Wan2.2 14B Q6_K | `cuda:1,11gb;cuda:0,15gb;cpu,2gb` | Allow CPU spillover |
| LTX-2 19B Q4 | `cuda:1,10gb;cuda:0,14gb` | Lighter model |
| UMT5-XXL | `cuda:1` | Keep on RTX 3060 |
| VAE | `cuda:1,3gb` | Small, fast |

### DisTorch2 Loader Nodes

| Loader | Purpose |
|--------|---------|
| `UnetLoaderGGUFAdvancedDisTorch2MultiGPU` | GGUF diffusion models |
| `VAELoaderDisTorch2MultiGPU` | VAE with GPU distribution |
| `CLIPLoaderDisTorch2MultiGPU` | T5/CLIP encoders |

### VRAM Budget by Resolution (Wan2.2)

| Resolution | Max Frames | VRAM Used | Fits in 28GB? |
|------------|------------|-----------|---------------|
| 480×848 | 81 | ~22GB | ✅ GPU-only |
| 576×1024 | 81 | ~24GB | ✅ GPU-only |
| 720×1280 | 81 | ~27GB | ⚠️ Tight |
| 720×1280 | 41 | ~22GB | ✅ GPU-only |
| 1080×1920 | 41 | ~30GB+ | ❌ CPU offload needed |

---

## Workflow Compatibility Matrix

### I2V Workflows

| Workflow | DisTorch2 | Single GPU | Models Required |
|----------|-----------|------------|-----------------|
| `wan22_i2v_distorch2_api.json` | ✅ | ❌ | Q6_K H+L, UMT5, CLIP, VAE |
| `wan22_i2v_ltx2_audio_api.json` | ✅ | ❌ | Q6_K H+L, UMT5, CLIP, VAE, LoRAs |

### T2V Workflows

| Workflow | DisTorch2 | Single GPU | Models Required |
|----------|-----------|------------|-----------------|
| `ltx2_distorch2_multigpu_api.json` | ✅ | ❌ | LTX-2 Q4, Gemma, LTX VAE |
| Built-in Wan2.2 T2V | ✅ | ⚠️ | FP8 H+L, UMT5, VAE |

### T2I Workflows

| Model Type | DisTorch2 | Single GPU | Models Required |
|------------|-----------|------------|-----------------|
| SDXL | ❌ | ✅ | Checkpoint, SDXL VAE |
| Flux | ✅ | ⚠️ | flux1-dev-fp8, T5, AE |
| Flux 2 | ✅ | ❌ | flux2-dev-Q4_K_M.gguf, mistral_3_small_flux2_fp8, flux2-vae |
| SD1.5 | ❌ | ✅ | Checkpoint |

---

## Adding New Generation Modes

### Backend (comfyui_client.py)

1. Add mode to appropriate dict:
```python
I2V_GENERATION_MODES = {
    "new_mode": {
        "name": "Display Name",
        "description": "What it does",
        "workflow_file": "path/to/workflow.json",
        "default_steps": 6,
        "default_cfg": 1.0,
    },
}
```

2. Or for T2V:
```python
T2V_GENERATION_MODES = {
    "new_model": {
        "name": "Model Name",
        "description": "Description",
        "workflow_file": "workflow.json",  # or None for built-in
        "model_type": "new_model",
        "default_steps": 8,
        "default_cfg": 1.0,
        "max_frames": 97,
        "default_frames": 25,
    },
}
```

### Frontend

Update the tool component (e.g., `TextToVideoTool.jsx`) to:
1. Fetch modes from `/api/t2v-modes` or `/api/i2v-modes`
2. Display model selector buttons
3. Pass `model_type` to generation endpoint

---

## Resource Management

### Memory Optimization Techniques

1. **SageAttention**: Use `PathchSageAttentionKJ` node for 15-20% VRAM reduction
2. **CPU Offload**: DisTorch2 `donor_device: cpu` for emergency spillover
3. **GGUF Quantization**: Q4_K_M uses ~50% less VRAM than FP8
4. **Sequential Loading**: `eject_models: true` frees VRAM between phases

### Recommended Workflow Structure (DisTorch2)

```
[Pass 1: High Noise]
├── UnetLoaderGGUFAdvancedDisTorch2MultiGPU (high noise model)
├── CLIPLoaderDisTorch2MultiGPU (text encoder)
├── VAELoaderDisTorch2MultiGPU (VAE)
├── SageAttention patch
├── Sampler (steps 1-3)
└── [eject models]

[Pass 2: Low Noise]
├── UnetLoaderGGUFAdvancedDisTorch2MultiGPU (low noise model)
├── Sampler (steps 4-6)
├── VAE Decode
└── Video output
```

---

## Future Models (Planned/TODO)

- [ ] Mochi-1 (DisTorch2 integration pending)
- [ ] CogVideoX (needs custom nodes)
- [ ] Hunyuan Video (very high VRAM)
- [ ] AnimateDiff Lightning (for quick animations)

---

**Last Updated**: 2026-01-12
**Maintainer**: Oelala Team
