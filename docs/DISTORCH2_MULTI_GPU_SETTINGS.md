# DisTorch2 Multi-GPU Settings Guide

> **Last Updated:** 2026-01-16
> **Hardware:** RTX 5060 Ti 16GB (cuda:0) + RTX 3060 12GB (cuda:1) = 28GB Total VRAM

## Overview

DisTorch2 is a multi-GPU model distribution system for ComfyUI that allows splitting large models across multiple GPUs and optionally CPU RAM. This document describes the optimal settings discovered through extensive testing.

## ⚠️ CRITICAL: PyTorch vs nvidia-smi GPU Indices

**PyTorch CUDA indices differ from nvidia-smi!**

| nvidia-smi | PyTorch | GPU | VRAM | Role |
|------------|---------|-----|------|------|
| 0 | **cuda:1** | RTX 3060 | 12GB | Model storage (donor) |
| 1 | **cuda:0** | RTX 5060 Ti | 16GB | Primary compute |

Always use **PyTorch indices** in allocation strings!

---

## DisTorch2 Node Settings

### Available Settings per Node

| Setting | Type | Description |
|---------|------|-------------|
| `compute_device` | Dropdown | Primary GPU for computation (where forward passes run) |
| `donor_device` | Dropdown | Secondary device for model overflow |
| `virtual_vram_gb` | Float | Simple mode: how much VRAM to use on compute_device |
| `expert_mode_allocations` | String | **Advanced**: Full control over model distribution |
| `eject_models` | Boolean | Unload other models before loading this one |

### Settings Interaction

```
If expert_mode_allocations is set:
    → Uses expert allocation string (ignores virtual_vram_gb/donor_device for distribution)
    → compute_device still determines WHERE computation happens

If expert_mode_allocations is empty:
    → Uses virtual_vram_gb + donor_device for simple 2-device split
```

---

## Expert Mode Allocation String Format

### Syntax
```
device1,size1;device2,size2;device3,size3;...
```

### Examples
```
cuda:0,10gb;cuda:1,4gb;cpu,*     # 10GB on GPU0, 4GB on GPU1, rest on CPU
cuda:1,11gb;cuda:0,15gb;cpu,*   # 11GB on GPU1 FIRST, then GPU0, rest on CPU
cuda:0,8gb;cpu,*                 # 8GB on GPU0, rest on CPU (single GPU + offload)
```

### Key Rules

1. **Order matters!** DisTorch fills devices in the order specified
2. **First device gets priority** - put the device you want to hold most of the model first
3. **`*` means "remaining"** - typically used for CPU as fallback
4. **Sizes are maximums** - if model is smaller, device gets less

---

## Optimal Settings for This Hardware

### Wan2.2 14B Q6_K Model (~11.5GB)

#### Maximum Video Length (480p Portrait)

**Tested Ceiling: 353-355 frames (~22 seconds @ 16fps)**

```json
{
  "expert_mode_allocations": "cuda:1,11gb;cuda:0,15gb;cpu,*",
  "compute_device": "cuda:0",
  "donor_device": "cuda:1",
  "virtual_vram_gb": 16,
  "eject_models": true
}
```

**Memory Distribution:**
| Component | cuda:1 (3060) | cuda:0 (5060 Ti) | CPU |
|-----------|---------------|------------------|-----|
| Model Layers | ~11.0 GB (97%) | ~0.3 GB (3%) | 0 GB |
| Activations | ~0.2 GB | ~15.5 GB | - |
| **Total** | **11.2 GB / 12 GB** | **15.8 GB / 16 GB** | **0 GB** |

**Why this works:**
- By putting `cuda:1` (3060) first with 11GB, it receives ~97% of the model
- `cuda:0` (5060 Ti) keeps only ~3% of model but has 15.5GB free for activations/KV-cache
- The 5060 Ti does ALL computation (100% utilization) while 3060 holds model weights

#### Alternative Allocations

**Balanced (shorter videos, faster):**
```
cuda:0,10gb;cuda:1,4gb;cpu,*
```
- Max frames: ~161 (10 sec)
- 5060 Ti: ~10GB model + ~5GB activations
- 3060: ~1.3GB model overflow
- Faster because less data transfer

**CPU Offload (longer videos, slower):**
```
cuda:1,8gb;cuda:0,12gb;cpu,*
```
- Enables 400+ frames if needed
- Model partially on CPU
- Significantly slower due to PCIe transfers

---

## Resolution/Frame Limits

### 480 × 848 (Portrait) - Recommended for Long Videos

| Frames | Duration | Allocation | Status |
|--------|----------|------------|--------|
| 161 | ~10 sec | `cuda:0,10gb;cuda:1,4gb;cpu,*` | ✅ Stable |
| 241 | ~15 sec | `cuda:1,10gb;cuda:0,14gb;cpu,*` | ✅ Stable |
| 321 | ~20 sec | `cuda:1,11gb;cuda:0,15gb;cpu,*` | ✅ **SAFE MAX** |
| 341 | ~21 sec | `cuda:1,11gb;cuda:0,15gb;cpu,*` | ✅ Tight |
| 351-355 | ~22 sec | `cuda:1,11gb;cuda:0,15gb;cpu,*` | ⚠️ OOM risk |
| 357+ | ~22.3 sec | Any | ❌ OOM |

⚠️ **Production recommendation:** Use **321 frames** max for stability.

### 576 × 1024 (Standard Portrait)

| Frames | Duration | Allocation | Status |
|--------|----------|------------|--------|
| 81 | ~5 sec | `cuda:0,10gb;cuda:1,4gb;cpu,*` | ✅ Stable |
| 121 | ~7.5 sec | `cuda:1,10gb;cuda:0,14gb;cpu,*` | ⚠️ Tight |
| 161 | ~10 sec | Any | ❌ OOM |

### 720 × 1280 (HD Portrait)

| Frames | Duration | Allocation | Status |
|--------|----------|------------|--------|
| 41 | ~2.5 sec | `cuda:1,8gb;cuda:0,12gb;cpu,*` | ✅ With CPU offload |
| 81 | ~5 sec | Any | ❌ OOM |

---

## Workflow Node Configuration

### UnetLoaderGGUFAdvancedDisTorch2MultiGPU

```json
{
  "inputs": {
    "unet_name": "wan2.2_i2v_14B_Q6_K.gguf",
    "dequant_dtype": "default",
    "patch_dtype": "default",
    "patch_on_device": false,
    "compute_device": "cuda:0",
    "virtual_vram_gb": 16,
    "donor_device": "cuda:1",
    "expert_mode_allocations": "cuda:1,11gb;cuda:0,15gb;cpu,*",
    "eject_models": true
  },
  "class_type": "UnetLoaderGGUFAdvancedDisTorch2MultiGPU"
}
```

### VAELoaderDisTorch2MultiGPU

```json
{
  "inputs": {
    "vae_name": "Wan2.1_VAE.safetensors",
    "compute_device": "cuda:0",
    "virtual_vram_gb": 16,
    "donor_device": "cuda:1",
    "expert_mode_allocations": "cuda:1,11gb;cuda:0,15gb;cpu,*",
    "eject_models": true
  },
  "class_type": "VAELoaderDisTorch2MultiGPU"
}
```

### CLIPLoaderDisTorch2MultiGPU

```json
{
  "inputs": {
    "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "type": "wan",
    "device": "cuda:0",
    "virtual_vram_gb": 16,
    "donor_device": "cuda:1",
    "expert_mode_allocations": "cuda:1,11gb;cuda:0,15gb;cpu,*",
    "eject_models": true
  },
  "class_type": "CLIPLoaderDisTorch2MultiGPU"
}
```

---

## Troubleshooting

### OOM During Sampling
- **Cause:** Activations/KV-cache too large for remaining VRAM
- **Fix:** Reduce frames OR put more model on secondary GPU (increase cuda:1 allocation)

### OOM During VAE Decode
- **Cause:** VAE decode needs contiguous memory
- **Fix:** Lower frame count, VAE decode is frame-count dependent

### 3060 Shows 0% Utilization
- **Normal!** The 3060 holds model weights but compute happens on 5060 Ti
- Model layers transfer to compute device during forward pass

### Model Won't Load
- Check `eject_models: true` to clear previous models
- Verify CLIP name exists: `umt5_xxl_fp8_e4m3fn_scaled.safetensors`

---

## CLIP/T5 Encoder Options

| Model | Size | Quality | VRAM |
|-------|------|---------|------|
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | ~2GB | Good | Low |
| `umt5-xxl-enc-bf16.safetensors` | ~4GB | Better | Medium |
| `umt5-xxl-enc-bf16-uncensored.safetensors` | ~4GB | Better (uncensored) | Medium |

**Recommendation:** Use `umt5_xxl_fp8_e4m3fn_scaled.safetensors` for max video length.

---

## Test Results Log

### 2026-01-16 Plafond Test Series

**Hardware:**
- RTX 5060 Ti 16GB (PyTorch cuda:0, nvidia-smi 1) - PRIMARY COMPUTE
- RTX 3060 12GB (PyTorch cuda:1, nvidia-smi 0) - MODEL STORAGE
- Total: 28GB VRAM

**Configuration:**
- Resolution: 480 × 848 (portrait)
- Model: wan2.2_i2v_14B_Q6_K.gguf (~11.5GB)
- CLIP: umt5-xxl-enc-bf16.safetensors
- Sampler: uni_pc, 6 steps, CFG 1.0
- Allocation: `cuda:1,11gb;cuda:0,15gb;cpu,*`

**Frame Limit Binary Search:**
| Frames | Duration | VRAM Total | Time/Step | Result |
|--------|----------|------------|-----------|--------|
| 161 | ~10 sec | ~22GB | ~110s | ✅ |
| 241 | ~15 sec | ~24GB | ~170s | ✅ |
| 281 | ~17.5 sec | ~25GB | ~190s | ✅ |
| 321 | ~20 sec | ~26GB | ~227s | ✅ SAFE MAX |
| 341 | ~21 sec | ~27GB | ~240s | ✅ Tight |
| 351 | ~22 sec | ~27GB | ~250s | ✅ |
| 353 | ~22 sec | ~27GB | ~250s | ⚠️ OOM risk |
| 355 | ~22 sec | ~27GB | - | ⚠️ OOM risk |
| 357 | ~22.3 sec | OOM | - | ❌ |
| 361 | ~22.5 sec | OOM | - | ❌ |

**VRAM Usage During 321 Frame Generation:**
```
RTX 3060 (cuda:1):    11,237 MB / 12,288 MB (91%) - holds model
RTX 5060 Ti (cuda:0): 14,854 MB / 16,311 MB (91%) - activations + 3% model
Total:                ~26 GB / 28 GB
5060 Ti Utilization:  100% (all compute happens here)
3060 Utilization:     0-2% (model weight storage only)
```

**Generation Times (6 steps, uni_pc):**
```
81 frames  @ 480p: ~50-60s/step   → ~5-6 min total
161 frames @ 480p: ~110-120s/step → ~12 min total
321 frames @ 480p: ~227s/step     → ~23 min total
```

**Key Discovery:** Allocation ORDER matters!
- `cuda:1,11gb;cuda:0,15gb` = 3060 gets model FIRST (97% of weights)
- `cuda:0,10gb;cuda:1,4gb` = 5060 Ti gets model first, overflow to 3060
- Putting smaller GPU first with high allocation = optimal for long videos

---

## Resolution Scaling Guide

Based on tested 480p limits, extrapolated for other resolutions:

| Resolution | Pixel Count | Safe Max Frames | Video Length | Notes |
|------------|-------------|-----------------|--------------|-------|
| 480×848 | 407K | **321** | ~20 sec | Tested ✅ |
| 576×1024 | 590K | ~81-121 | ~5-8 sec | 1.45x pixels vs 480p |
| 720×1280 | 922K | ~41-61 | ~2.5-4 sec | 2.26x pixels vs 480p |
| 1080×1920 | 2073K | ~17-25 | ~1-1.5 sec | 5.09x pixels vs 480p |

**Memory scaling formula:**
```
VRAM_needed ≈ base_vram + (pixels × frames × constant)

For Wan2.2 14B Q6_K:
- Base VRAM (model): ~11.5GB
- Per-frame overhead: ~50MB at 480p
- Scaling: quadratic with resolution
```

**576×1024 Recommended Settings:**
- Frames: 81-121 for safe operation
- Allocation: same `cuda:1,11gb;cuda:0,15gb;cpu,*`
- Expected VRAM: ~24-27GB

**Tip:** For higher resolutions, prioritize fewer frames over lower quality.
Always test with target resolution before production use.

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│  OPTIMAL SETTINGS FOR MAX VIDEO LENGTH (480p Portrait)     │
├─────────────────────────────────────────────────────────────┤
│  expert_mode_allocations: cuda:1,11gb;cuda:0,15gb;cpu,*    │
│  compute_device:          cuda:0                            │
│  donor_device:            cuda:1                            │
│  virtual_vram_gb:         16                                │
│  eject_models:            true                              │
├─────────────────────────────────────────────────────────────┤
│  Max Frames: 353 (~22 sec @ 16fps)                         │
│  VRAM Used:  27 GB / 28 GB (96%)                           │
└─────────────────────────────────────────────────────────────┘
```
