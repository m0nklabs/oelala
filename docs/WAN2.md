# WAN 2.2 Image-to-Video

**Updated**: 2026-01-08

---

## Overview

WAN 2.2 is the primary I2V model for Oelala. 14B parameters, supports text prompts.

## Hardware Requirements

| GPU | Max Resolution | Max Frames |
|-----|----------------|------------|
| 12GB | 480x848 | 41 |
| 16GB | 720x1280 | 41 |
| 28GB (multi) | 720x1280 | 81 |

## Quick Start

```python
from wan2_generator import Wan2VideoGenerator

generator = Wan2VideoGenerator()
generator.load_model()
generator.generate_video_from_image(
    image_path="input.jpg",
    prompt="A person dancing",
    output_path="output.mp4",
    num_frames=81
)
```

---

## Multi-GPU Setup (DisTorch2)

### Hardware
- RTX 5060 Ti 16GB (`cuda:1`)
- RTX 3060 12GB (`cuda:0`)
- Total: 28GB

### DisTorch2 Allocation
```
cuda:0,11gb;cuda:1,15gb;cpu,2gb
```

### ComfyUI Nodes
- `UnetLoaderGGUFAdvancedDisTorch2MultiGPU`
- `VAELoaderDisTorch2MultiGPU`
- `CLIPLoaderDisTorch2MultiGPU`
- `PathchSageAttentionKJ` (VRAM savings)

---

## Single GPU Setup

For 16GB GPU, use CPU offloading:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1  # 16GB card only
```

### Group Offloading
```python
from diffusers.hooks.group_offloading import apply_group_offloading

# Text Encoder: block-level (4 blocks/group)
apply_group_offloading(
    text_encoder,
    offload_type="block_level",
    num_blocks_per_group=4
)

# Transformer: leaf-level with streaming
transformer.enable_group_offload(
    offload_type="leaf_level",
    use_stream=True
)
```

---

## Model Variants

| Model | Size | Notes |
|-------|------|-------|
| wan22_i2v_14B_bf16 | 28GB | Full precision |
| wan22_i2v_14B_Q6_K | ~10GB | GGUF quantized |
| wan22_i2v_14B_Q5_K_M | ~8GB | More compression |

---

## Files

| File | Purpose |
|------|---------|
| `wan2_generator.py` | Python generator class |
| `workflows/ImageToVideo/wan22_*.json` | ComfyUI workflows |
| `docs/MULTI_GPU_SETUP.md` | Detailed multi-GPU guide |

---

## Performance

| Resolution | Frames | Time | VRAM |
|------------|--------|------|------|
| 480x848 | 81 | ~3min | 16GB |
| 720x1280 | 41 | ~4min | 24GB |
| 720x1280 | 81 | ~7min | 28GB |

---

## Integration

### Backend Endpoints
- `/generate` - I2V with queue
- `/generate-wan22-comfyui` - Dual-pass
- `/generate-wan22-async` - Async queue

### Frontend
- `ImageToVideoTool.jsx` - Main UI
- `CameraMotionSelector.jsx` - 16 presets
