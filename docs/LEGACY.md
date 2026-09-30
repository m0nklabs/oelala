# Legacy inventory and cleanup

> What the platform no longer serves, what was removed, what is parked, and where a
> parked file can be fetched again. Hot rules live in `AGENTS.md`; current state in
> `docs/HANDOFF.md`. Numbers measured 2026-09-30.

## Policy

Oelala serves **cutting edge only**. A model family leaves the product when it is no
longer the best option for its slot, and legacy weights leave the disk once nothing in
the product references them. Weights that are still useful as *LoRA* material for a
possible successor may stay parked, clearly marked.

## Video model slots

| Slot | Model | Where it runs |
|---|---|---|
| Leading | **MiniMax-H3** (joint video+audio, 24 fps, 4-15 s) | RunPod `oelala-minimax-h3`, on-demand variants from the public mirror |
| Second | **LTX-2.3** (22B distilled) | RunPod `oelala-ltx23` (cloud-only; local copies were removed earlier) |
| Retired | **Wan 2.2** | endpoint `oelala-wan22` deleted, adapters/UI/workflows removed |

### Decided 2026-09-30 after a model-landscape review

- **H3 stays the leading model.** It is still MiniMax's newest open video model, ComfyUI core
  keeps shipping H3 work (v0.38.0), and its ecosystem (turbo distills, ControlNet-Union,
  community finetunes) keeps growing. Nothing open challenges it.
- **Upgrade the second slot LTX-2.3 → LTX-2.5** (released 2026-08-11, weights 2026-09-01).
  Same vendor and license family, ComfyUI-native (`ComfyUI-LTXVideo` ships 2.5 workflows),
  int8 path ≈ 40 GB and fits an A40-48GB with text-encoder offload. Code v1.4.x adds chunked
  long-video generation, keyframe-aware decode and multi-GPU runners; the nvfp4 checkpoint
  (18.7 GB) is the Blackwell fit. The HF repo is **gated** — the license must be accepted on
  the HuggingFace account before a worker can pull it.
- **Wan 2.2 stays retired.** There is no open-weight successor: Wan 2.5 / 2.6 / 2.7 / 3.0 are
  API-only, and the newest open Wan models are 2.2 derivatives (Wan-Animate-2, Wan-Dancer).
  That also means the parked Wan 2.2 LoRAs cannot be validated against a future Wan 3 — keep
  them only if a specific Wan-LoRA look is wanted, and drop them if disk is needed.
- **Do not add** HunyuanVideo-1.5 (official repo blocks EU/UK/KR gated access, no audio,
  older than LTX-2.5) or Ovi / LingBot (no LoRA ecosystem).


## Wan 2.2 — retired (measured 2026-09-30)

Local disk before cleanup: **43.4 GB in 36 files**.

| Item | Size | Re-fetch source |
|---|---|---|
| `unet_gguf/Wan2.2-I2V-A14B-{High,Low}Noise-Q8_0.gguf` | 30.8 GB | `QuantStack/Wan2.2-I2V-A14B-GGUF` (`HighNoise/`, `LowNoise/` — identical filenames) |
| `clip/umt5-xxl-enc-bf16.safetensors` | 11.4 GB | `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` (also on the public flat dump) |
| `vae/Wan2.1_VAE.safetensors`, `vae/wan_2.1_vae.safetensors` | 0.5 GB | `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` |
| Wan2.2 LoRAs (36 files, incl. T2V/I2V high+low pairs, lightning 4-step) | ~13 GB | **Parked** — kept for a possible open-weight Wan 3, see below |

Catalog totals for the family were higher (288.5 GB over 41 entries) because the catalog
also counts the cloud worker's copies and the T2V/quants that only ever existed there.

**Parked LoRAs.** The Wan 2.2 LoRA ecosystem is large and mostly NSFW-finetuned motion
LoRAs. They are kept on `/mnt/ssd/loras` and their `docs/lora_registry.yaml` entries stay,
marked as legacy: they are only worth loading if a future Wan release keeps the same
architecture (DiT + umt5 text encoder + Wan VAE). Verify that before reusing them — a new
text encoder or VAE breaks every LoRA regardless of the DiT.

## Other families (for later cleanup decisions)

| Family | Entries | GB (catalog) | Note |
|---|---|---|---|
| wan22 | 41 | 288.5 | retired; local runtime weights deleted, LoRAs parked |
| unknown | 123 | 236.6 | mostly LoRAs and local-only files without a recorded upstream |
| minimax_h3 | 13 | 144.1 | leading model, includes the mirrored finetunes |
| ltx23 | 18 | 80.0 | second model; 8.3 GB of that is local LoRAs |
| flux / flux2 | 7 | 79.7 | image models, still in use |
| qwen_image_edit | 4 | 28.3 | image editing |
| sdxl / krea2 | 21 | 20.5 | check whether these still have a product slot |
| wan21 | 1 | 0.2 | shared Wan VAE only |

The 123 `unknown` entries are the biggest open item: local files whose upstream is not
recorded, so they are not reproducible on a fresh install. `scripts/build_model_catalog.py`
lists them; recording a source per file is the fix.

## Cleanup log

- 2026-09-30 — Wan 2.2 retired from the product (adapters, UI, workflows, RunPod
  endpoint) and its local runtime weights deleted; LoRAs parked. See
  `changelog/20261001-retire-wan22.md`.
