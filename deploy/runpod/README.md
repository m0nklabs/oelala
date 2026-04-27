# RunPod Serverless Deployment

Deploy Oelala's ComfyUI video generation pipeline to RunPod Serverless GPUs.

## Architecture

```
oelala-backend (FastAPI)
├── Local ComfyUI (28GB dual-GPU) → default, low-latency jobs
└── RunPod Serverless → heavy/burst jobs (A40/A100/H100)
    └── Docker: ComfyUI + GGUF + VHS + KJNodes + LTXVideo helpers
        └── handler.py → receives workflow JSON, queues in ComfyUI, returns output
```

## Files

| File | Purpose |
|------|---------|
| `deploy.sh` | **One-command deploy**: build → tag → push → update RunPod template |
| `Dockerfile` | ComfyUI worker image with Wan + LTX custom nodes |
| `handler.py` | RunPod serverless handler — bridges API to ComfyUI |
| `README.md` | This file |

## Prerequisites

1. **RunPod account** with API key
2. Optional **RunPod Network Volume** for LoRAs/private assets only
3. **Docker** to build the image

## Build & Deploy

**Always use `deploy.sh`** — it builds, tags, pushes, AND updates the RunPod template
in one shot. This prevents the ":latest tag was pushed but RunPod still uses
the old explicit tag" mistake.

```bash
cd deploy/runpod

# Full deploy: build → tag → push → update RunPod template
./deploy.sh

# Skip build (reuse existing :latest image)
./deploy.sh --skip-build

# Preview what would happen
./deploy.sh --dry-run
```

> ⚠️ **NEVER manually `docker push :latest` without updating the RunPod template!**
> RunPod templates use explicit dated tags (e.g. `20260408-135917`), not `:latest`.
> If you push `:latest` only, RunPod will keep pulling the old tag and your
> changes will never reach production. This mistake wasted 3 deploys on 2026-04-08.

After updating the template, each deploy script reapplies the endpoint defaults
from `src/backend/runpod_defaults.py`. This keeps scaling and timeout behavior
stable even after template edits.

Current production defaults:

| Endpoint | GPU tiers | Scaling | Job policy |
|----------|-----------|---------|------------|
| Wan2.2 Cloud Max | 48GB+ | workers `0/2`, idle `120s`, `QUEUE_DELAY:4` | 60 min execution, 120 min TTL |
| LTX-2.3 | 80GB+ | workers `0/2`, idle `120s`, `QUEUE_DELAY:1` | 45 min execution, 120 min TTL |
| I2I/Qwen Edit | 48GB+ | workers `0/2`, idle `120s`, `QUEUE_DELAY:4` | 15 min execution, 60 min TTL |

## LoRA / Private Asset Storage

The RunPod Network Volume is reserved for LoRAs and private/custom assets only.
Do not use it for general Hugging Face models, cached public models, or broad
model prewarming.

Recommended volume:

```text
Name: oelala-runpod-lora-eu-cz
Datacenter: EU-CZ-1
Size: 50GB
```

Suggested layout:

```
/runpod-volume/models/
├── loras/
│   ├── private_character.safetensors
│   └── rare_style.safetensors
└── custom/
    └── hard_to_replace_asset.bin
```

Uploads from local Oelala storage can happen on demand later. Keep the volume
clean and intentionally curated.

### On-demand upload from local assets

The backend already supports on-demand LoRA delivery to cloud jobs via signed
download URLs built from the local LoRA store under `/mnt/ssd/loras/`.

If you want to push selected private/rare assets into the detached RunPod LoRA
volume ahead of time, use:

```bash
python deploy/runpod/upload_private_assets.py /mnt/ssd/loras/my-private-lora.safetensors
python deploy/runpod/upload_private_assets.py --remote-prefix models/custom /path/to/rare-asset.bin
```

This uploader blocks known public/general model filenames and only allows
uploads into `models/loras/` or `models/custom/`.

## Region Behavior

RunPod Serverless workers are constrained by attached Network Volumes.

- If you attach a single Network Volume to an endpoint, workers can only start in the datacenter where that volume exists.
- If you want broader EU GPU availability, you need either multiple Network Volumes attached to the same endpoint, or no Network Volume at all.
- Multiple volumes do **not** sync automatically. You must copy models/assets yourself between datacenters.

For Oelala this means the `oelala-runpod-lora-eu-cz` volume should stay unattached by default and be treated as remote LoRA/private-asset storage. Attach it only if you explicitly accept EU-CZ-1 placement constraints.

## Deploy Endpoint

Via RunPod Web UI:
1. Go to **Serverless** → **+ New Endpoint**
2. Container image: `<your-registry>/oelala-comfyui-worker:latest`
3. GPU: `NVIDIA A40` (48GB) or `NVIDIA L40` (48GB) recommended
4. Min workers: `0` (scale to zero when idle)
5. Max workers: `3` (burst capacity)
6. Idle timeout: `5` seconds
7. Leave **Network Volume** detached unless you explicitly want LoRA/private-asset access with EU-CZ-1 placement constraints

## Configuration

Set these in `/home/flip/oelala/.env`:

```env
RUNPOD_API_KEY=rpa_xxxxxxxxxxxxx
RUNPOD_ENDPOINT_ID=<endpoint-id-from-dashboard>
RUNPOD_ENDPOINT_IDS=<primary-endpoint-id>
RUNPOD_DEBUG=false
```

## Recommended Production Setup

### Lowest complexity

- One endpoint only.
- `workersMin=0`
- `idleTimeout=120`
- No attached Network Volume.
- Accept public/general model downloads on container disk or RunPod cached-model storage.

This is the best fit when Cloud Max and LTX-2 are burst paths and local GPUs remain the primary path.

### Single image, multi-model worker

The current worker image is intended to stay shared across cloud video modes.

- Wan 2.2 cloud jobs still use the existing Cloud Max workflow path.
- LTX-2 cloud jobs reuse the same worker image, but only pull in LTX-related nodes/models when a workflow references them.
- This avoids maintaining a second RunPod image unless model pressure or cold-start time proves it necessary.

### Best EU availability

- One endpoint only.
- `workersMin=0`
- `idleTimeout=120`
- Attach multiple Network Volumes, one per target EU datacenter, only if you later decide you really need region-scoped private asset caches.
- Keep the same model tree replicated to every attached volume.

This gives RunPod more datacenters to place workers in, but requires explicit model replication.

### Best cost-performance tradeoff for Oelala

- Keep a single endpoint.
- Keep `workersMin=0` and `idleTimeout=120`.
- Remove duplicate fallback endpoints unless they are intentionally mapped to different datacenters.
- Use RunPod cached models where possible for Hugging Face-hosted assets.
- Keep only LoRAs and hard-to-replace private/custom assets on the `EU-CZ-1` Network Volume.

Cached models reduce cold-start cost because RunPod does not bill worker time while the cached model is downloading onto the target host.

### Ephemeral Disk Reality

- `40GB` container disk is only viable when the worker can satisfy the big Wan 2.2 files from RunPod cached models.
- A pure live-download cold start needs about `41.7GB` of final required models plus staging/scratch overhead for the largest file.
- In practice, plan for roughly `60GB+ free disk` inside the worker filesystem, which usually means a noticeably larger `containerDiskInGb` once the base image is included.

If you do not attach a volume and cached models are unavailable, an undersized container will now fail fast during startup with a clear disk-capacity error instead of dying halfway through the second download.

## Public vs Private Asset Policy

The worker now defaults to this policy:

- Public Hugging Face models are linked from RunPod cached-model storage when available.
- If a core Cloud Max model is still missing, it is downloaded onto the container's local ComfyUI model directory.
- The RunPod Network Volume is used only for reusable LoRAs, private models, and other non-HF hard-to-recreate assets.
- General/public models must never be stored on the RunPod Network Volume.

This keeps persistent storage focused on assets you cannot trivially re-download.

## Cached Model Strategy For Oelala

RunPod cached models are useful here, but they do **not** replace the entire current model tree.

### What can move to RunPod cached models

The current worker downloads these files from Hugging Face in `handler.py`:

| File | Hugging Face repo | Recommended storage |
|------|-------------------|---------------------|
| `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` | `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` | RunPod cached model |
| `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` | `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` | RunPod cached model |
| `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` | `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` | RunPod cached model |
| `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` | `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` | RunPod cached model |
| `umt5_xxl_fp16.safetensors` | `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` | RunPod cached model |

These all come from the same 2.2 repo, so they are the best candidate for a cached-model endpoint.

### What should stay on a Network Volume or in the image

| File / asset | Current source | Why it should not be your primary cached-model path |
|--------------|----------------|-----------------------------------------------------|
| `wan_2.1_vae.safetensors` | `Comfy-Org/Wan_2.1_ComfyUI_repackaged` | Different HF repo than the 2.2 pack |
| `clip_vision_h.safetensors` | `Comfy-Org/Wan_2.1_ComfyUI_repackaged` | Different HF repo than the 2.2 pack |
| User/job LoRAs | backend download URLs or local uploads | Private/rare assets, suitable for the LoRA-only Network Volume |
| Custom private assets | custom/local assets | Not part of the current cached-model flow |

### Important limitation

RunPod currently supports only one cached model per endpoint. For Oelala that means:

- You can use a cached model for `Comfy-Org/Wan_2.2_ComfyUI_Repackaged`.
- You still need another source for the 2.1 VAE and CLIP Vision files.
- Therefore the best near-term setup is a **hybrid** approach, not a pure cached-model approach.

### Best hybrid setup for Oelala

1. Configure the endpoint cached model as `Comfy-Org/Wan_2.2_ComfyUI_Repackaged`.
2. Keep the `EU-CZ-1` Network Volume detached unless you explicitly need LoRA/private-asset access inside workers.
3. Use that volume only for:
    - persistent LoRA cache
    - private or hard-to-replace custom assets
4. RunPod exposes cached models at `/runpod-volume/huggingface-cache/hub/` using Hugging Face cache conventions; set `RUNPOD_CACHED_MODEL_DIRS` only if you need to override or extend that search path.
5. The worker now links files from cached-model storage before falling back to live Hugging Face downloads.
6. Keep `workersMin=0`, `workersMax=2`, and `idleTimeout=120` unless production traffic proves a different cost/latency tradeoff is needed.

### Startup vs workflow-specific preparation

- Worker startup should preload only the shared Cloud Max core: `umt5_xxl_fp16.safetensors` and `wan_2.1_vae.safetensors`.
- Mode-specific assets stay on-demand:
    - T2V jobs pull/link only the `wan2.2_t2v_*` UNETs they reference.
    - I2V jobs pull/link only the `wan2.2_i2v_*` UNETs plus `clip_vision_h.safetensors`.
    - LTX-2 jobs use the same worker image with the LTX/GGUF node stack and resolve their own model assets only when those workflows are queued.
- This avoids wasting cold-start time on I2V-only assets when the job is T2V, while still using RunPod cached-model storage whenever those files are available.

This gives the best cost/performance balance without forcing every Cloud Max job to pay for multi-GB Hugging Face downloads inside a running worker.

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
- **Private LoRA/custom asset not found**: Check Network Volume mount path (`/runpod-volume`) or use `upload_private_assets.py`
- **OOM**: Use a larger GPU tier or reduce resolution/frames
- **Timeout**: Request policies are set per endpoint in `src/backend/runpod_defaults.py`; increase the matching profile for very long generations.
