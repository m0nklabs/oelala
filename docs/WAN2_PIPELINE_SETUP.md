# Wan 2.2 & Flux Pipeline Setup Guide

This document summarizes the successful configuration for running a combined Text-to-Image (Flux) and Image-to-Video (Wan 2.2) pipeline on a consumer GPU setup (specifically targeting an RTX 5060 Ti 16GB).

## Hardware Context
- **Primary GPU**: NVIDIA RTX 5060 Ti (16GB VRAM)
- **Secondary GPU**: NVIDIA RTX 3060 (12GB VRAM) - *Ignored for this pipeline to prevent OOM*
- **System RAM**: Sufficient for offloading (32GB+ recommended)

## Key Challenges & Solutions

### 1. GPU Addressing & Isolation
PyTorch and `nvidia-smi` often enumerate GPUs differently. To ensure we use the 16GB card (ID 1 in nvidia-smi) as the primary device (ID 0 in PyTorch):

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
```

This hides the 12GB card from the script, preventing accidental allocation on the smaller GPU.

### 2. Flux.1-schnell (Text-to-Image) Optimization
Running Flux in standard `bfloat16` mode consumes ~24GB VRAM, which is too much for a 16GB card.

**Solution:**
- **INT8 Quantization**: We use `optimum.quanto` to quantize the transformer weights to 8-bit.
- **CPU Offloading**: `pipe.enable_model_cpu_offload()` moves unused components to RAM.
- **VAE Precision**: We force the VAE to `float32` to prevent visual artifacts (noise/grain) that can occur with `bfloat16` VAEs.

```python
from optimum.quanto import freeze, qint8, quantize

# Load in bfloat16
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)

# Quantize Transformer to INT8
quantize(pipe.transformer, weights=qint8)
freeze(pipe.transformer)

# Offload to CPU
pipe.enable_model_cpu_offload()
```

### 3. Wan 2.2 I2V 14B (Image-to-Video) Optimization
The 14B parameter model is massive. Standard loading requires >40GB VRAM.

**Solution: Group Offloading**
This technique loads only specific blocks of the model into VRAM during inference and sends them back to CPU immediately after.

- **Text Encoder**: Block-level offloading (4 blocks per group).
- **Transformer**: Leaf-level offloading with streaming.

```python
from diffusers.hooks.group_offloading import apply_group_offloading

# Text Encoder Offloading
apply_group_offloading(
    text_encoder,
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="block_level",
    num_blocks_per_group=4
)

# Transformer Offloading
transformer.enable_group_offload(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="leaf_level",
    use_stream=True
)
```

## Running the Pipeline

The scripts are located in `oelala/wan_experiments/`.

### 1. Configuration
Edit `wan_config.json` to set your prompt, resolution, and other parameters.
*Note: Portrait resolution (480x832) is supported and works well.*

```json
{
    "prompt": "Your prompt here...",
    "width": 480,
    "height": 832,
    "frames": 81,
    "gpu": 0
}
```

### 2. Execution
Always run with the environment variables set:

```bash
cd oelala/wan_experiments
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
source ../../venvs/torch-sm120/bin/activate
python run_t2i_i2v_pipeline.py
```

## File Structure
- `run_t2i_i2v_pipeline.py`: Main script combining both steps.
- `test_t2i_fixed.py`: Standalone test for the Flux image generation (useful for quick prompt iteration).
- `wan_config.json`: Central configuration file.
