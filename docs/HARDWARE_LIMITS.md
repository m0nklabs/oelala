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

### OOM Boundaries (approximate)
- **GPU-only**: ~720×1280 @ 81 frames is near limit
- **CPU offload**: ~1280×720 @ 101 frames works, higher may OOM
- **Frame limit**: 241+ frames at 720×400 works, higher untested

### Tips
1. Use GPU-only mode for faster iteration
2. Switch to CPU offload for maximum resolution/frames
3. Portrait (9:16) uses same VRAM as landscape (16:9) at same pixel count
4. SageAttention reduces VRAM ~15-20%

## Changelog
- 2025-12-31: Added GPU-only mode support via DisTorch2 fix
- 2025-12-31: Tested 241 frames (15s video) successfully
- 2025-12-31: Tested 720×1280 HD portrait
