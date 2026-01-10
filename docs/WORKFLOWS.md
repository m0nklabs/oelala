# Oelala Workflows Guide

**Updated**: 2026-01-08

---

## Overview

Oelala uses ComfyUI as backend. This guide maps endpoints to workflows.

## API Endpoints

| Endpoint | Purpose | ComfyUI Workflow |
|----------|---------|------------------|
| `/generate` | I2V with image upload | `wan22_i2v_*.json` |
| `/generate-text` | T2V generation | T2I + I2V pipeline |
| `/generate-wan22-comfyui` | Dual-pass I2V | DisTorch2 multi-GPU |
| `/upscale-video` | Video upscaling | `video_upscale_realesrgan.json` |
| `/interpolate-video` | Frame interpolation | `rife_interpolation.json` |

---

## Quick Start

### Image-to-Video
```bash
curl -X POST http://localhost:7998/generate \
  -F "file=@image.jpg" \
  -F "prompt=cinematic video" \
  -F "num_frames=81"
```

### Text-to-Video
```bash
curl -X POST http://localhost:7998/generate-text \
  -F "prompt=A cat exploring forest" \
  -F "num_frames=49" \
  -F "model_type=wan2.2"
```

---

## Workflow Locations

### Oelala Workflows
```
workflows/
├── ImageToVideo/       # I2V with WAN 2.2
├── TextToImage/        # SDXL, Flux
├── VideoUpscale/       # Real-ESRGAN
├── FrameInterpolation/ # RIFE
└── VideoExtension/     # Extend videos
```

### ComfyUI Direct
- Web UI: `http://localhost:8188`
- Workflows: `ComfyUI/user/default/workflows/`

---

## Model Selection

| Model | Speed | Quality | VRAM |
|-------|-------|---------|------|
| `light` | ⚡ Fast | ⭐⭐ | 4GB |
| `svd` | ⚖️ Balanced | ⭐⭐⭐ | 8GB |
| `wan2.2` | 🐢 Slow | ⭐⭐⭐⭐⭐ | 16GB+ |

---

## Frame Counts

| Frames | Duration @15fps | Use Case |
|--------|-----------------|----------|
| 17 | ~1 sec | Quick test |
| 41 | ~2.7 sec | Standard |
| 81 | ~5.4 sec | Long video |
| 241 | ~16 sec | Extended |

---

## Troubleshooting

### Backend Issues
```bash
# Check health
curl http://localhost:7998/health

# View logs
journalctl -u oelala-api -f
```

### ComfyUI Issues
```bash
# Check service
systemctl status comfyui

# View logs
journalctl -u comfyui -f
```

### Common Fixes
- **OOM Error**: Reduce frames or resolution
- **Model not found**: Check `COMFYUI_INVENTORY.md`
- **Slow generation**: Enable SageAttention

---

## Oelala vs ComfyUI

| Aspect | Oelala | ComfyUI |
|--------|--------|---------|
| Interface | Web GUI | Node editor |
| Ease | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Flexibility | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Speed | ⭐⭐⭐⭐ | ⭐⭐⭐ |

**Use Oelala** for quick results and user-facing apps.
**Use ComfyUI** for custom workflows and experimentation.
