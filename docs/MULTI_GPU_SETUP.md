# Multi-GPU Setup Guide for Wan 2.2

This guide explains how to leverage multiple GPUs (e.g., RTX 5060 Ti + RTX 3060) to run large models like Wan 2.2 14B by distributing the model weights across available VRAM.

## Concept: Model Sharding
Instead of loading the entire model onto one GPU (which requires offloading to CPU if VRAM is insufficient), we can split the model layers across multiple GPUs. This is handled automatically by the `accelerate` library using `device_map`.

**Goal:** Combine VRAM (16GB + 12GB = 28GB total) to keep more of the model on GPU, reducing inference time.

## Prerequisites
Ensure `accelerate` is installed:
```bash
pip install accelerate
```

## Implementation Strategy

### 1. Load the Transformer with `device_map`
The Transformer is the largest component (~28GB in FP16, ~14GB in INT8). We load it first and let `accelerate` distribute it.

```python
from diffusers import WanTransformer3DModel
import torch

transformer = WanTransformer3DModel.from_pretrained(
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    device_map="balanced",  # Or "auto"
    max_memory={0: "15GB", 1: "11GB"} # Reserve buffer for OS/Display
)
```

### 2. Load the Pipeline
Pass the sharded transformer to the pipeline. The pipeline will handle the other smaller components (Text Encoder, VAE).

```python
from diffusers import WanImageToVideoPipeline

pipe = WanImageToVideoPipeline.from_pretrained(
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
# Do NOT use enable_model_cpu_offload() if everything fits in VRAM!
# If it still doesn't fit, you can combine sharding with offloading.
```

## Example Script (`test_wan_multigpu.py`)

```python
import os
import torch
from diffusers import WanImageToVideoPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video
from PIL import Image

# Allow seeing both GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 

def run_multigpu():
    # 1. Load Sharded Transformer
    print("📦 Loading Transformer across GPUs...")
    transformer = WanTransformer3DModel.from_pretrained(
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        max_memory={0: "15GB", 1: "11GB"}
    )

    # 2. Load Pipeline
    print("🔗 Loading Pipeline...")
    pipe = WanImageToVideoPipeline.from_pretrained(
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )
    
    # Move other components to primary GPU or CPU as needed
    # pipe.enable_model_cpu_offload() # Optional: use only if OOM occurs

    # 3. Run Inference
    print("🚀 Generating Video...")
    image = Image.open("test_t2i_output.png")
    output = pipe(
        prompt="Your prompt",
        image=image,
        num_frames=81,
        num_inference_steps=20
    ).frames[0]
    
    export_to_video(output, "multigpu_output.mp4")

if __name__ == "__main__":
    run_multigpu()
```

## Troubleshooting
- **OOM on GPU 0**: Reduce `max_memory` for GPU 0 to leave space for activations.
- **Slow Performance**: Ensure PCI-E bandwidth is sufficient (x8/x16 lanes). If one card is x1 or x4, data transfer might be the bottleneck.
