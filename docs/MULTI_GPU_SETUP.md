# Multi-GPU Setup Guide for WAN 2.2 with ComfyUI

This guide explains how to run WAN 2.2 14B across multiple GPUs using ComfyUI-MultiGPU DisTorch2.

## Hardware Requirements
- 2+ NVIDIA GPUs with combined 24GB+ VRAM
- Example: RTX 5060 Ti (16GB) + RTX 3060 (12GB) = 28GB total

## Installation

### 1. Install ComfyUI-MultiGPU
```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/pollockjj/ComfyUI-MultiGPU.git
```

### 2. Apply Local Fixes (Required for ComfyUI 0.6.0+)

Two fixes are needed for proper multi-GPU operation:

#### Fix 1: Tuple Parsing in `distorch_2.py`
The `_load_list()` function returns 5-element tuples in ComfyUI 0.6.0+. Update three locations:

```python
# Line ~520: In _assign_blocks_by_memory()
name, size, target_device, _ = entry[0], entry[2], entry[3], entry[4]

# Line ~545: In _assign_blocks_greedy()  
name, size, target_device, _ = entry[0], entry[2], entry[3], entry[4]

# Line ~565: In _assign_blocks_even()
name, size, target_device, _ = entry[0], entry[2], entry[3], entry[4]
```

#### Fix 2: GPU-Only Mode (No CPU Fallback)
By default, DisTorch2 falls back to CPU for remaining model bytes. To force GPU-only:

In `distorch_2.py`, find the wildcard handling (~line 622):

```python
# Change from:
wildcard_device = "cpu"  # Default wildcard device

# To:
wildcard_device = None  # No default - must be explicit
has_wildcard = False
```

And update the remaining bytes handling (~line 656):

```python
if remaining_model_bytes > 0:
    if has_wildcard and wildcard_device:
        # Original wildcard behavior
        final_byte_allocations[wildcard_device] += remaining_model_bytes
    else:
        # GPU-ONLY MODE: Distribute to existing GPUs
        gpu_devices = [d for d in final_byte_allocations.keys() if 'cuda' in d]
        if gpu_devices:
            total_gpu_alloc = sum(final_byte_allocations[d] for d in gpu_devices)
            for dev in gpu_devices:
                proportion = final_byte_allocations[dev] / total_gpu_alloc
                final_byte_allocations[dev] += int(remaining_model_bytes * proportion)
```

## Allocation Strings

### GPU-Only Mode (Recommended)
```
cuda:0,12gb;cuda:1,16gb
```
- All model on GPUs
- No CPU RAM usage for model weights
- Lower latency

### CPU Fallback Mode
```
cuda:0,12gb;cuda:1,16gb;cpu,*
```
- Overflow goes to CPU RAM
- Enables higher resolution/frames
- Higher latency due to CPU<->GPU transfers

## Workflow Configuration

In your ComfyUI workflow, set these nodes:

### UnetLoaderGGUFAdvancedDisTorch2MultiGPU
```
expert_mode_allocations: cuda:0,12gb;cuda:1,16gb
```

### VAELoaderDisTorch2MultiGPU
```
expert_mode_allocations: cuda:0,12gb;cuda:1,16gb
```

### CLIPLoaderDisTorch2MultiGPU
```
expert_mode_allocations: cuda:0,12gb;cuda:1,16gb
```

## Verifying GPU Distribution

Check ComfyUI logs for:
```
[MultiGPU DisTorch V2] GPU-ONLY MODE: Distributed remaining X MB across 2 GPUs (no CPU fallback).
```

Or with CPU fallback:
```
[MultiGPU DisTorch V2] Assigning remaining X MB of model to wildcard device 'cpu'.
```

### Expected Distribution (WAN 2.2 14B Q6_K)
```
Device    VRAM GB    Model GB    Dist %
cuda:0      12.0       14.4       43%
cuda:1      16.0       19.2       57%
```

## Troubleshooting

### OOM Errors
1. Reduce resolution or frames
2. Switch to CPU fallback mode
3. Use smaller GGUF quantization (Q5_K_S instead of Q6_K)

### Model Not Distributed
- Check `expert_mode_allocations` is set on ALL loader nodes
- Verify ComfyUI restarted after fixing distorch_2.py

### High RAM Usage Despite GPU-Only
- T5 text encoder may still use CPU
- This is expected; focus is on model layers

## Performance Comparison

| Mode | VRAM Used | RAM Used | Speed |
|------|-----------|----------|-------|
| GPU-Only | ~17GB | ~24GB | Fast |
| CPU Fallback | ~20GB | ~53GB | Slower |

## References
- [ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU)
- [WAN 2.2 Model](https://huggingface.co/Wan-AI)
- [GitHub Issue #160](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/160) - Tuple parsing fix
