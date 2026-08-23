# RunPod: New Serverless Endpoint Checklist

> **Created**: 2026-04-14
> **Why**: We kept forgetting steps and wasting deploys. This is the canonical checklist.
> **Rule**: Follow ALL steps IN ORDER. Do not skip. Do not "do it later".

## Pre-requisites

- [ ] GHCR login: `docker login ghcr.io` (use PAT with `write:packages`)
- [ ] RunPod API key available in `.env` as `RUNPOD_API_KEY`
- [ ] Container registry auth ID on RunPod for GHCR (current: `cmmbssf3a00anky07egrtupgt`)

## Step 1: Create Deployment Files

Create `deploy/runpod-{name}/` with three files:

### handler.py
- [ ] Based on existing handler (copy from `deploy/runpod/handler.py` or `deploy/runpod-ltx23/handler.py`)
- [ ] Define `MODELS` list with HuggingFace URLs, filenames, target dirs
- [ ] `ensure_models()` — download all models on cold start
- [ ] `start_comfyui()` — launch ComfyUI subprocess + wait for health
- [ ] `handler(job)` — receive workflow, queue to ComfyUI, poll for result, return outputs
- [ ] `collect_outputs()` — handle both images (PNG/JPG) and videos (MP4)
- [ ] Set appropriate timeouts (COMFYUI_STARTUP_TIMEOUT, WORKFLOW_TIMEOUT)
- [ ] Support `lora_downloads` if LoRAs are used
- [ ] Support `images` dict for input images (base64-encoded)

### Dockerfile
- [ ] Base image: `runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404`
- [ ] Clone ComfyUI from GitHub (depth 1)
- [ ] Clone ONLY the custom nodes this handler actually needs
- [ ] Install Python deps: `runpod requests httpx huggingface_hub`
- [ ] Pre-create model directories (don't download models — they're fetched at cold start)
- [ ] Copy handler.py into the image
- [ ] CMD: `python handler.py`
- [ ] Set `CACHE_DATE` arg to bust Docker cache when needed

### deploy.sh
- [ ] Loads `RUNPOD_API_KEY` from `.env`
- [ ] Generates dated version tag: `$(date +%Y%m%d-%H%M%S)`
- [ ] Builds Docker image
- [ ] Tags with both `:latest` AND `:{dated-tag}`
- [ ] Pushes BOTH tags to GHCR
- [ ] **Updates RunPod template** via GraphQL `saveTemplate` mutation WITH the dated tag
- [ ] Supports `--skip-build`, `--dry-run`, `--help` flags
- [ ] Set executable: `chmod +x deploy.sh`

## Step 2: Build & Push Docker Image

```bash
cd deploy/runpod-{name}
./deploy.sh
```

> **⚠️ CRITICAL**: You MUST push the image BEFORE creating the template/endpoint.
> RunPod tries to pull the image when a worker spins up. If the image doesn't exist
> at GHCR, the worker will fail silently and jobs stay "IN_QUEUE" forever.

## Step 3: Create RunPod Serverless Template

Use the GraphQL API. **ALL of these fields are required for serverless**:

```graphql
mutation {
  saveTemplate(input: {
    name: "oelala-{name}-worker"
    imageName: "ghcr.io/m0nklabs/oelala-{name}-worker:{dated-tag}"
    containerDiskInGb: 100
    volumeInGb: 0              # ← MUST be 0 for serverless
    isServerless: true          # ← MUST be true
    dockerArgs: ""              # ← MUST be present (type String!, even if empty)
    containerRegistryAuthId: "cmmbssf3a00anky07egrtupgt"
    env: [{ key: "COMFYUI_PATH", value: "/comfyui" }]
  }) {
    id
    name
    imageName
    isServerless
  }
}
```

### Common mistakes that will bite you:

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Omit `dockerArgs` | Template creates as **pod template**, not serverless. `isServerless` silently ignored. | Always include `dockerArgs: ""` |
| Omit `isServerless: true` | Pod template, can't be used with `saveEndpoint` | Always set explicitly |
| Set `volumeInGb: >0` | Worker expects a network volume that may not exist in the target datacenter | Set to `0` unless you specifically attach a volume |
| Use `imageName: ":latest"` | Template keeps old tag forever, future deploys don't reach production | Always use explicit dated tags |
| Omit `containerRegistryAuthId` | Worker can't pull from private GHCR repo | Always include for private images |

### Verify template is serverless:
```graphql
{ myself { serverlessTemplates { id name isServerless imageName } } }
```

If `isServerless` is `false`, **delete and recreate**. Updating `isServerless` after creation is silently ignored.

## Step 4: Create RunPod Serverless Endpoint

```graphql
mutation {
  saveEndpoint(input: {
    name: "oelala-{name}"
    templateId: "{template-id-from-step-3}"
    gpuIds: "AMPERE_48,ADA_48_PRO,AMPERE_80,ADA_80_PRO,BLACKWELL_96,HOPPER_141,BLACKWELL_180"
    workersMin: 0
    workersMax: 2
    idleTimeout: 120
    scalerType: "QUEUE_DELAY"
    scalerValue: 4
  }) {
    id
    name
    templateId
  }
}
```

### GPU tier selection:
- See `docs/RUNPOD_GPU_TIERS.md` for all valid tier IDs
- **Use architecture-tier IDs** (e.g. `AMPERE_48`), NOT GPU model names (e.g. `"NVIDIA RTX 4090"`)
- API silently accepts wrong names but scheduler never matches them → jobs stuck IN_QUEUE forever
- 48GB+ for most ComfyUI workloads, 80GB+ for very large models (LTX 22B, etc.)

## Step 5: Update Backend Code

### 5a. Backend routing
In `src/backend/app.py`, find the `generate_{name}()` endpoint and pass `endpoint_id`:

```python
result = await _submit_to_runpod(
    workflow=workflow,
    ...,
    endpoint_id=os.getenv("RUNPOD_{NAME}_ENDPOINT_ID"),
)
```

### 5b. Environment variables
Add to `.env`:
```
RUNPOD_{NAME}_ENDPOINT_ID={endpoint-id}
RUNPOD_{NAME}_TEMPLATE_ID={template-id}
```

### 5c. Restart backend
```bash
sudo systemctl restart oelala-backend
```

> **⚠️ CRITICAL**: The backend reads `.env` via systemd's `EnvironmentFile`.
> Changes to `.env` are NOT picked up until the service is restarted.
> Verify with: `sudo cat /proc/$(pgrep -f "uvicorn app:app" | tail -1)/environ | tr '\0' '\n' | grep RUNPOD_{NAME}`

## Step 6: Update Documentation

- [ ] Add endpoint section to `.github/copilot-instructions.md`
- [ ] Update `deploy.sh` with real template + endpoint IDs (replace PLACEHOLDERs)
- [ ] Update `docs/RUNPOD_GPU_TIERS.md` if new tiers were discovered
- [ ] Commit all changes

## Step 7: Verify End-to-End

1. Submit a job via the frontend or API
2. Check RunPod dashboard: job should move from `IN_QUEUE` → `IN_PROGRESS` → `COMPLETED`
3. Check worker logs for model download progress (first cold start is slow: ~28GB of models)
4. Verify output is returned correctly to the frontend

### Troubleshooting

| Symptom | Likely Cause |
|---------|-------------|
| Jobs stuck `IN_QUEUE`, 0 workers | Image not pushed, or template has wrong `imageName` |
| Jobs stuck `IN_QUEUE`, "Initializing" | Worker starting, downloading models (can take 5-10 min on cold start) |
| Worker starts then exits immediately | Handler crash — check RunPod worker logs |
| `endpoint_id` is None in backend logs | `.env` not reloaded — restart the systemd service |
| "Serverless endpoints cannot use pod templates" | Template is not serverless — recreate with `dockerArgs: ""` + `isServerless: true` |
| Worker runs but output is empty | `collect_outputs()` looking in wrong dir, or ComfyUI workflow error |

## Quick Reference: Current Endpoints

| Name | Endpoint ID | Template ID | Image | GPU Tiers |
|------|-------------|-------------|-------|-----------|
| oelala-wan22 | `x2x496ymkidl3m` | `tkpy0pi8gt` | `ghcr.io/m0nklabs/oelala-comfyui-worker` | 48GB+ |
| oelala-ltx23 | `ctpoa610dva4ww` | `c1fz26l07d` | `ghcr.io/m0nklabs/oelala-ltx23-worker` | 80GB+ |
| oelala-i2i | `8djiexluyybooj` | `ed2614hd8k` | `ghcr.io/m0nklabs/oelala-i2i-worker` | 48GB+ |
| oelala-minimax-h3 | `5xuvnvyww4ujnc` | `fpfo4gmnrw` | `ghcr.io/m0nklabs/oelala-minimax-h3-worker` | 80GB+ |

All current endpoints use `workersMin=0`, `workersMax=2`, and `idleTimeout=120`. Wan/I2I scale at `QUEUE_DELAY:4`; LTX-2.3 and MiniMax-H3 use `QUEUE_DELAY:1` because the big workers' cold starts are more expensive to wait on. Runtime job policies are applied by `src/backend/runpod_defaults.py`; add every new endpoint profile there so requests get an explicit `executionTimeout` and `ttl` in milliseconds.

### MiniMax-H3 endpoint notes (`deploy/runpod-minimax-h3/`)

- **Model**: MiniMax-H3 FL2VA (joint video+audio DiT, t2v + i2v via first-frame keyframes). Repack: `Comfy-Org/MiniMax-H3`; workflow mirrors the official Comfy-Org "Image to Video (MiniMax H3)" template.
- **Models downloaded at cold start (~42.5 GB total)**: `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (20.97 GB), `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (15.69 GB, no Blackwell needed), `minimax_h3_video_vae_fp16.safetensors` (5.21 GB), `minimax_h3_audio_vae_fp32.safetensors` (0.61 GB).
- **ComfyUI**: cloned from official `Comfy-Org/ComfyUI` master — the H3 core nodes (`MiniMaxH3ImageToVideo`, `VAEDecodeAudio`, ...) landed there with the H3 release. Bump `CACHE_DATE` in the Dockerfile to refresh the checkout.
- **GPU**: 80GB+ tiers (`AMPERE_80,ADA_80_PRO,HOPPER_141,BLACKWELL_96,BLACKWELL_180`). The int8/nvfp4 quantizations also fit 48GB tiers for short generations — untested, start on 80GB.
- **Audio**: H3 always generates a synchronized soundtrack — no `audio_prompt` needed; the mp4 comes out with muxed audio.

## The Golden Rule

> **deploy.sh is the ONLY way to deploy.** It builds, tags, pushes, AND updates the template atomically.
> Never manually `docker push` without updating the template. RunPod uses explicit dated tags, not `:latest`.
