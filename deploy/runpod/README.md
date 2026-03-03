# RunPod Serverless Deployment

Deploy Oelala's ComfyUI video generation pipeline to RunPod Serverless GPUs.

## Architecture

```
oelala-backend (FastAPI)
├── Local ComfyUI (28GB dual-GPU) → default, low-latency jobs
└── RunPod Serverless → heavy/burst jobs (A40/A100/H100)
    └── Docker: ComfyUI + GGUF + VHS + KJNodes + Florence2 + RIFE
        └── handler.py → receives workflow JSON, queues in ComfyUI, returns output
```

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | ComfyUI worker image with essential custom nodes |
| `handler.py` | RunPod serverless handler — bridges API to ComfyUI |
| `README.md` | This file |

## Prerequisites

1. **RunPod account** with API key
2. **RunPod Network Volume** with models (see [Model Setup](#model-setup))
3. **Docker** to build the image

## Build & Push

```bash
cd deploy/runpod

# Build the image
docker build -t oelala-comfyui-worker .

# Tag for Docker Hub or RunPod registry
docker tag oelala-comfyui-worker <your-registry>/oelala-comfyui-worker:latest

# Push
docker push <your-registry>/oelala-comfyui-worker:latest
```

## Model Setup (Network Volume)

Models are NOT baked into the Docker image. Create a RunPod Network Volume and upload:

```
/runpod-volume/models/
├── diffusion_models/
│   ├── wan2.2_i2v_high_noise_14B_Q6_K.gguf
│   └── wan2.2_i2v_low_noise_14B_Q6_K.gguf
├── vae/
│   └── wan_2.2_vae.safetensors
├── text_encoders/
│   └── umt5xxl_fp8_e4m3fn_scaled.safetensors
├── clip_vision/
│   └── clip_vision_h.safetensors
└── upscale_models/
    └── RealESRGAN_x2plus.pth
```

Estimated volume size: **~25GB**

## Deploy Endpoint

Via RunPod Web UI:
1. Go to **Serverless** → **+ New Endpoint**
2. Container image: `<your-registry>/oelala-comfyui-worker:latest`
3. GPU: `NVIDIA A40` (48GB) or `NVIDIA L40` (48GB) recommended
4. Min workers: `0` (scale to zero when idle)
5. Max workers: `3` (burst capacity)
6. Idle timeout: `5` seconds
7. Attach your **Network Volume** at `/runpod-volume`

## Configuration

Set these in `/home/flip/oelala/.env`:

```env
RUNPOD_API_KEY=rpa_xxxxxxxxxxxxx
RUNPOD_ENDPOINT_ID=<endpoint-id-from-dashboard>
RUNPOD_DEBUG=false
```

## Usage

The backend automatically routes jobs to RunPod when:
- `compute_target=cloud` is specified in the API request
- Or the requested resolution/duration exceeds local GPU capacity

Jobs sent to RunPod follow the same workflow format as local ComfyUI.
Output files are returned as base64 in the API response.

## Cost Estimates

| GPU | $/hr | 480p 81f (~5min) | 720p 81f (~8min) | 1080p 81f (~12min) |
|-----|------|-------------------|-------------------|---------------------|
| A40 (48GB) | ~$0.39 | ~$0.03 | ~$0.05 | ~$0.08 |
| L40 (48GB) | ~$0.44 | ~$0.04 | ~$0.06 | ~$0.09 |
| H100 (80GB) | ~$2.49 | ~$0.21 | ~$0.33 | ~$0.50 |

*Estimates based on RunPod pricing. Actual times depend on model and settings.*

## Troubleshooting

- **Cold start**: First job takes 30-60s while worker boots and loads models
- **Models not found**: Check Network Volume mount path (`/runpod-volume`)
- **OOM**: Use a larger GPU tier or reduce resolution/frames
- **Timeout**: Default 30min timeout; increase for very long generations
