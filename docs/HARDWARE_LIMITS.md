# Hardware Limits - WAN 2.2 I2V on RTX 5060 Ti (16GB) + RTX 3060 (12GB)

## Tested Configuration (December 2025)

### Hardware
- **GPU0**: NVIDIA RTX 3060 12GB (cuda:0)
- **GPU1**: NVIDIA RTX 5060 Ti 16GB (cuda:1)
- **Total VRAM**: 28GB
- **RAM**: 115GB (CPU offload optional)

### Model
- **WAN 2.2 I2V 14B GGUF Q6_K** (~34GB uncompressed)
- **ComfyUI-MultiGPU DisTorch2** for distribution
- **SageAttention** enabled

## Allocation Modes

### GPU-Only Mode (Recommended)
```
cuda:0,12gb;cuda:1,16gb
```
- No CPU fallback
- Lower RAM usage (~24GB vs ~53GB)
- Full GPU utilization
- Slightly less VRAM overhead

### CPU Fallback Mode
```
cuda:0,12gb;cuda:1,16gb;cpu,*
```
- T5 encoder offloaded to CPU
- Higher resolution/frames possible
- Slower due to CPU<->GPU transfers

## Verified Working Configurations

### Portrait (9:16)
| Resolution | Frames | Video Length | Time | Peak VRAM | Mode |
|------------|--------|--------------|------|-----------|------|
| 576×1024 | 81 | 5.1s @16fps | 6.5 min | 21.4GB | CPU offload |
| 576×1024 | 81 | 5.1s @16fps | ~6 min | 17.4GB | GPU-only ✅ |
| 720×1280 | 81 | 5.1s @16fps | 11.3 min | ~20GB | CPU offload |

### Landscape (16:9)
| Resolution | Frames | Video Length | Time | Peak VRAM | Mode |
|------------|--------|--------------|------|-----------|------|
| 720×400 | 101 | 6.3s @16fps | 4.1 min | ~20GB | CPU offload |
| 1024×576 | 101 | 6.3s @16fps | 8.5 min | ~20GB | CPU offload |
| 1280×720 | 101 | 6.3s @16fps | ~10 min | ~21GB | CPU offload |

### Maximum Frames (720×400)
| Frames | Video Length | Time | Status |
|--------|--------------|------|--------|
| 101 | 6.3s | 4.1 min | ✅ Works |
| 241 | 15.1s | 23.7 min | ✅ Works |
| 289 | 18.1s | ~28 min | Testing |

## VRAM Usage Breakdown

### GPU-Only Mode (~17.4GB total)
```
Component                    GPU0 (3060)   GPU1 (5060Ti)
──────────────────────────────────────────────────────────
WAN Model Layers (43%)       6.6 GB        -
WAN Model Layers (57%)       -             10.8 GB
T5 Text Encoder              Shared        Shared
VAE Decoder                  -             Dynamic
Latents/Activations          -             Dynamic
──────────────────────────────────────────────────────────
Total                        ~6.6 GB       ~10.8 GB
```

### CPU Offload Mode (~20GB GPU + ~30GB RAM)
```
Component                    GPU           CPU/RAM
──────────────────────────────────────────────────────────
WAN Model (83%)              ~20 GB        -
WAN Model (17%)              -             ~5.7 GB
T5 Text Encoder              -             ~8 GB
VAE Decoder                  Dynamic       -
──────────────────────────────────────────────────────────
```

## Recommended Presets for Frontend

### Quick Preview
```json
{
  "resolution": {"width": 576, "height": 1024},
  "frames": 49,
  "steps": 6,
  "estimated_time": "3-4 min",
  "quality": "preview",
  "mode": "gpu_only"
}
```

### Standard Portrait
```json
{
  "resolution": {"width": 576, "height": 1024},
  "frames": 81,
  "steps": 6,
  "estimated_time": "6-7 min",
  "quality": "good",
  "mode": "gpu_only"
}
```

### HD Portrait
```json
{
  "resolution": {"width": 720, "height": 1280},
  "frames": 81,
  "steps": 6,
  "estimated_time": "11-13 min",
  "quality": "high",
  "mode": "cpu_offload"
}
```

### Long Video (Low Res)
```json
{
  "resolution": {"width": 720, "height": 400},
  "frames": 241,
  "steps": 6,
  "estimated_time": "20-25 min",
  "quality": "good",
  "video_length": "15s"
}
```

## Known Limits

### OOM Boundaries (GPU-Only Mode)
- **1080×1920** (Full HD Portrait): ❌ OOM - needs 9.49GB for latents alone
- **720×1280** (720p Portrait @ 241f): ❌ OOM - needs ~11GB working memory
- **720×1280** (720p Portrait @ 81f): ✅ Works but slow (45 min) - DisTorch2 uses single GPU
- **720×400** @ 241 frames: ✅ SUCCESS - 22.5GB peak VRAM, 10.4 min

### OOM Boundaries (CPU Offload Mode)
- **1280×720** @ 101 frames: ✅ Works
- Higher resolutions/frames: May OOM

### Key Finding: GPU-Only Mode Distribution Issue
When using pure GPU-only allocation (`cuda:0,12gb;cuda:1,16gb` without `cpu,*`), 
DisTorch2 v2.5.9 tends to place 100% of model weights on cuda:0 instead of 
distributing between both GPUs. This causes:
- Slower generation (using only RTX 3060 instead of both GPUs)
- Higher single-GPU VRAM pressure
- Better dual-GPU distribution occurs with CPU fallback mode

### Tips
1. **For max speed**: Use CPU offload mode (`cuda:0,12gb;cuda:1,16gb;cpu,*`) - better distribution
2. **For lower RAM usage**: GPU-only mode works but may be slower
3. **Best combo**: 720×400 @ 241 frames (15s video) in ~10 min with proper dual-GPU
4. Portrait (9:16) uses same VRAM as landscape (16:9) at same pixel count
5. SageAttention reduces VRAM ~15-20%

## Tested Results Summary (2025-01-01)

### GPU-Only Mode Tests
| Resolution | Frames | Time | Status | Notes |
|------------|--------|------|--------|-------|
| 720×400 | 241 | 10.4 min | ✅ | Peak 22.5GB, proper dual-GPU |
| 720×1280 | 81 | 45 min | ✅ | Slow - single GPU used |
| 720×1280 | 241 | - | ❌ OOM | Needs ~11GB latent memory |
| 1080×1920 | 241 | - | ❌ OOM | Needs 9.49GB latents |

### Pixel Budget
- **Max working pixels × frames**: ~288,000 × 241 = 69M pixel-frames
- **720p @ 15s**: 921,600 × 241 = 222M pixel-frames → OOM
- **Formula**: Keep (pixels × frames) under ~100M for GPU-only safety margin

## Changelog
- 2026-01-01: Tested 1080p and 720p HD - both OOM in GPU-only mode
- 2026-01-01: Discovered DisTorch2 single-GPU distribution issue
- 2025-12-31: Added GPU-only mode support via DisTorch2 fix
- 2025-12-31: Tested 241 frames (15s video) successfully at 720×400
- 2025-12-31: Tested 720×1280 HD portrait
