# RunPod MiniMax-H3 Worker

Serverless ComfyUI worker for **MiniMax-H3** video generation (t2v + i2v).
MiniMax-H3 is a joint video+audio DiT: every generation produces a video
**with a synchronized soundtrack** — no separate audio step needed.

| | |
|---|---|
| Image | `ghcr.io/m0nklabs/oelala-minimax-h3-worker` |
| Target GPU | 80 GB+ (A100/H100/B200/GB200); int8/nvfp4 quants also fit 48 GB tiers (untested) |
| Cold-start download | ~42.5 GB |
| Container disk | 100 GB |

## Models (downloaded at cold start from HuggingFace)

From [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) (repack of
[`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)):

| File | Size | Role |
|---|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20.97 GB | FL2VA diffusion model (t2v + i2v keyframes) |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15.69 GB | Qwen3-VL-32B text encoder (nvfp4, no Blackwell needed) |
| `minimax_h3_video_vae_fp16.safetensors` | 5.21 GB | Video VAE |
| `minimax_h3_audio_vae_fp32.safetensors` | 0.61 GB | Audio VAE (H3 generates audio unconditionally) |

## How it works

The Oelala backend builds the ComfyUI workflow programmatically
(`build_cloud_minimax_h3_t2v_workflow` / `build_cloud_minimax_h3_i2v_workflow`
in `src/backend/comfyui_client.py`) and submits it to this endpoint. The
workflow mirrors Comfy-Org's official "Image to Video (MiniMax H3)" template:

```
UNETLoader (fl2va int8)
CLIPLoader (qwen3vl, type=minimax)
VAELoader (video) + VAELoader (audio)
  → MiniMaxH3ImageToVideo (prompt + optional first/last frame keyframes)
  → BasicGuider (no negative prompt)
  → KSamplerSelect(res_multistep) + BasicScheduler(simple, 20 steps)
  → SamplerCustomAdvanced
  → VAEDecode (video frames) + VAEDecodeAudio (soundtrack)
  → VHS_VideoCombine (mp4 with muxed audio)
```

Key facts:
- **24 fps**, frame count snaps to the model's `17k+5` grid (124 ≈ 5s; trained range ~124–362).
- Canvas: 768 short edge, `768×1344` area cap.
- No negative prompts, no CFG — `BasicGuider`.
- Sigma shift video 12.0 / audio 3.0 is baked into the checkpoint config
  (`supported_models.MiniMaxH3.sampling_settings`) — no extra node needed.
- ComfyUI is cloned from **official `Comfy-Org/ComfyUI` master**: the H3 core
  nodes live in `comfy_extras/nodes_minimax_h3.py` (landed with the H3 release,
  July 2026). Bump `CACHE_DATE` in the Dockerfile to refresh the checkout.

## Best settings — resolutie & duur (officiële bronnen)

Onderzoek: HuggingFace model card, ComfyUI docs
(`tutorials/video/minimax/minimax-h3`), officiële workflow templates, en de
node-bron (`nodes_minimax_h3.py`, `nodes_resolution.py`).

- **Canvas**: 768px short edge, cap `768×1344`, alles veelvoud van 32.
- **Outputgrootte** loopt via *megapixels* — de officiële `ResolutionSelector`
  formule (`target = MP × 1024²` px op de gekozen aspect ratio, afgerond op 32).
  De backend accepteert `megapixels` in het generation-request; de frontend
  biedt dit aan als "Kwaliteit (megapixels)".

| MP (16:9) | Dimensies | Opmerking |
|---|---|---|
| 0.4 | 864×480 | Template default — snel itereren |
| 0.6 | 1056×608 | Balans |
| 0.98 | 1344×768 | **Native canvas — aanbevolen full quality** |
| 2.0 | 1920×1088 | 2K — maximaal, langzaamst |

Andere aspect ratios schalen mee (0.98 MP @9:16 → 768×1344, @1:1 → 768×768 …).

- **Duur**: max ~15 s per clip (362 frames @24 fps = getraind bereik 124–362).
  De frame count wordt automatisch op de 17k+5 grid gezet: 3 s → 73 f, 5 s →
  124 f, 8 s → 192 f, 10 s → 243 f, 15 s → 362 f.
- **Sampling**: 20 stappen (`simple` schedule), sampler `res_multistep`,
  `BasicGuider` (geen CFG / negative prompt).
- **Snelheid** (optioneel, later): turbo LoRA's van
  [`ModelTC/Minimax-H3-Turbo`](https://github.com/ModelTC/Minimax-H3-Turbo) —
  4-step 768p (getraind op 1344×768, shift 6/3, 4 NFE) en 8-step v1.0
  (544p, 8 of 4 NFE). Niet in de image gebakken.

## Deploy

```bash
cd deploy/runpod-minimax-h3
./deploy.sh              # build + push + update template
./deploy.sh --skip-build # push existing image + update template
./deploy.sh --dry-run    # show what would happen
```

Before the **first** deploy, create the template + endpoint on RunPod and set
the IDs (see `docs/RUNPOD_NEW_ENDPOINT_CHECKLIST.md` for the GraphQL snippets
and common mistakes):

```
RUNPOD_MINIMAX_H3_TEMPLATE_ID=<template-id>
RUNPOD_MINIMAX_H3_ENDPOINT_ID=<endpoint-id>
```

Endpoint defaults (`src/backend/runpod_defaults.py`, profile `minimax_h3`):
`workersMin=0`, `workersMax=2`, `idleTimeout=120`, `QUEUE_DELAY:1`,
`executionTimeout=45min`, `ttl=2h`, GPU tiers
`AMPERE_80,ADA_80_PRO,HOPPER_141,BLACKWELL_96,BLACKWELL_180`.

## Backend wiring

- `.env`: `RUNPOD_MINIMAX_H3_ENDPOINT_ID` + `RUNPOD_MINIMAX_H3_TEMPLATE_ID`
- Adapters: `MiniMaxH3CloudT2VAdapter` / `MiniMaxH3CloudI2VAdapter`
  (`src/backend/generation/adapters/cloud/minimax_h3_{t2v,i2v}.py`)
- Restart the backend after changing `.env`:
  `sudo systemctl restart oelala-backend`

## Optional: turbo LoRAs

Comfy-Org publishes 4-step / 8-step turbo LoRAs for faster generations
(`loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors`,
`loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`, ~2 GB each).
They are not baked into the image; add them to the `MINIMAX_H3_MODELS` list in
`handler.py` (or push them to the LoRA network volume) and load them with a
`LoraLoaderModelOnly` node when you need them.
