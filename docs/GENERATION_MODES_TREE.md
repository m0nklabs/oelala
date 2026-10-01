# Generation Modes Tree

> **⚠️ HOLY TREE - SINGLE SOURCE OF TRUTH**
>
> Dit document bevat ALLE geteste en werkende generation modes per tool.
> **MOET worden bijgewerkt na elke succesvolle test van een nieuwe mode/model combo!**
>
> Zie ook: [GENERATION_MODES.md](GENERATION_MODES.md) voor gedetailleerde specs.

Visual tree structure of all generation modes per tool type.

---

## Tool Status Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│ 🛠️ PRIMARY TOOLS (Standalone generation)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ ✅ PRODUCTION                                                        │
│   ├── 🖼️ TextToImage (T2I)      → Generate images from text        │
│   ├── 🎬 ImageToVideo (I2V)     → Animate images into video        │
│   └── 🎥 TextToVideo (T2V)      → Direct text to video             │
│                                                                      │
│ 🔨 IN DEVELOPMENT                                                    │
│   ├── 🖼️ ImageToImage (I2I)     → Style transfer, inpainting       │
│   ├── 📝 ImageToText (caption)  → Generate descriptions            │
│   └── 🎵 TextToAudio (T2A)      → Generate audio/music             │
│                                                                      │
│ 📋 PLANNED                                                           │
│   ├── 🔊 TextToSpeech (TTS)     → Voice synthesis                   │
│   └── 🎭 FaceSwap               → Face replacement                  │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ ⚙️ POST-PROCESSING (Unified System - 2026-01-17)                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ ✅ INLINE OPTIONS (I2V/T2V checkboxes):                             │
│   ├── 📈 Upscale              → 2x/4x Real-ESRGAN (+5 credits)     │
│   ├── 🔄 Frame Interpolation  → 30/48/60 fps RIFE (+3 credits)     │
│   └── 🔊 Add Audio            → Attach audio track (I2V only)       │
│                                                                      │
│ ✅ STANDALONE TOOL (Advanced → Post-Processing):                    │
│   ├── 📈 Upscale Mode         → Process existing videos            │
│   ├── 🔄 Interpolate Mode     → Increase FPS of existing videos    │
│   └── 🔗 Concat Mode          → Join multiple videos together      │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ 🔗 PIPELINES (Tool combinations)                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ├── 🗣️ SpeechToVideo        → TTS + I2V + LipSync                │
│   ├── 📺 VideoToVideo         → Extract frames + I2V + Stitch      │
│   ├── 🎬 T2I→I2V Pipeline     → T2I + I2V (MiniMax-H3 I2V)         │
│   └── 🎵 Video+Audio          → I2V/T2V + Audio generation         │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ 💡 FUTURE (No workflow yet)                                         │
├─────────────────────────────────────────────────────────────────────┤
│   ├── 🎓 LoRATraining         → Train custom models                │
│   ├── 🔲 Reframe              → Change aspect ratio intelligently  │
│   └── 🧠 PromptGenerator      → AI-assisted prompt creation        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🎬 Image-to-Video (I2V)

### Maximum Duration Settings (Tested 2026-01-17)

| Resolution | Model | Max Duration | Max Frames | VRAM Usage |
|------------|-------|--------------|------------|------------|
| 768p | MiniMax-H3 (cloud, leading) | ~15 sec | 362 | RunPod 80GB+ |
| **480p** | LTX-2 | **12 sec** | 97 | ~18GB |
| 576p | LTX-2 | 8 sec | 97 | ~20GB |
| 720p | LTX-2 | 5 sec | 65 | ~22GB |

MiniMax-H3 (leading I2V/T2V model) also runs on the local Windows-PC ComfyUI
(16GB GPU); A40 duration benchmarks are in the T2V section below.

```
I2V Generation Modes
│
├── 📦 minimax_h3 (default, leading)
│   │   "MiniMax-H3 FL2VA 22B (joint video+audio)"
│   │   Single model • Native stereo audio • Cloud + local Windows-PC
│   │
│   ├── ☁️ Cloud (RunPod 80GB+, nvfp4 text encoder)
│   │   ├── minimax_h3_fl2va_pruned_int8_convrot.safetensors  [20.97GB]
│   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors      [15.69GB]
│   │
│   ├── 🪟 Local (Windows-PC ComfyUI, int8_convrot set, 16GB GPU)
│   │   ├── minimax_h3_fl2va_pruned_int8_convrot.safetensors
│   │   ├── qwen3vl_32b_minimax_h3_int8_convrot.safetensors
│   │   ├── minimax_h3_video_vae_fp16.safetensors
│   │   └── minimax_h3_audio_vae_fp32.safetensors
│   │
│   └── ✨ LoRA Support
│       └── Single-stage (MiniMax-H3 section in docs/lora_registry.yaml)
│
└── 📦 ltx2 (second choice - cloud-only)
    │   "LTX-2.3 (RunPod)"
    │   Single model • Faster inference • Uses Gemma encoder
    │
    └── ☁️ Cloud-only — local LTX-2 19B files were removed (see T2V section)
```

### I2V Model Comparison

Main UI modes are intentionally simplified. See `docs/LOCAL_GENERATION_DEFAULTS.md`
for the product-facing local-vs-cloud mode policy.

| Feature | MiniMax-H3 Cloud | MiniMax-H3 Local (Windows-PC) | LTX-2.3 Cloud |
|---------|------------------|-------------------------------|---------------|
| Pass Type | Single (joint video+audio) | Single (joint video+audio) | Single |
| Default Steps | 20 full / 8 standard / 4 draft | 20 full / 8 standard / 4 draft | 8 |
| Default CFG | — (turbo presets use shift) | — (turbo presets use shift) | 1.0 |
| Text Encoder | Qwen3-VL 32B (nvfp4 AWQ) | Qwen3-VL 32B (int8 convrot) | Gemma 3 |
| Best For | Leading quality, native audio | Local generation, no cloud cost | Fast cloud iterations |
| LoRA Support | ✅ Single-stage | ✅ Single-stage | ✅ Single-stage |
| Default Worker | RunPod 80GB+ | Windows-PC ComfyUI (16GB GPU) | RunPod 80GB+ |

The promoted I2V modes are **MiniMax-H3** (leading/default — cloud and local
Windows-PC) with **LTX-2.3** (cloud) as the alternative.

---

## 🎥 Text-to-Video (T2V)

```
T2V Generation Modes
│
├── 📦 minimax_h3 (default, leading)
    │   "MiniMax-H3 FL2VA 22B (joint video+audio) — cloud + local"
    │   Max frames: 362 | Default: 124 | Fixed 24fps
    │
    ├── ☁️ Cloud (RunPod 80GB+, nvfp4 text encoder)
    │   ├── minimax_h3_fl2va_pruned_int8_convrot.safetensors   [20.97GB]  official
    │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors      [15.69GB]
    │
    │   Model variants (model_variant + quality_mode, downloaded on demand):
    │   ├── eros         h3ErosMax_beta5_3185144   [14.00GB] turbo fp8/w4a8  → draft 4 / standard 8
    │   ├── eros_int8_turbo h3ErosMax_beta5_3178732 [20.97GB] turbo int8     → draft 4 / standard 8
    │   ├── eros_int8    h3ErosMax_beta5_3185154   [20.97GB] non-turbo       → full 20
    │   ├── dasiwa_turbo DasiwaMinimaxH3_dasiwaHybridTurboV2_3203135 [20.97GB] → draft 4 / standard 8
    │   └── dasiwa       DasiwaMinimaxH3_dasiwaHybridV2_3203130      [20.97GB] → full 20
    │   Mirrors: bomehika/oelala-models (public) — Civitai URL is the fallback.
    │
    └── 🪟 Local (Windows PC ComfyUI, int8_convrot set, 16GB GPU)
        ├── minimax_h3_fl2va_pruned_int8_convrot.safetensors
        ├── qwen3vl_32b_minimax_h3_int8_convrot.safetensors
        ├── minimax_h3_video_vae_fp16.safetensors
        └── minimax_h3_audio_vae_fp32.safetensors
│
└── 📦 ltx2 (second choice - cloud-only)
    │   "LTX-2.3 (RunPod)"
    │
    └── ☁️ Cloud-only — local LTX-2 19B files were removed
```

> Measured on an A40 (768×1344, 124 frames): official full 20 steps ≈ 689 s,
> official standard (8 steps + turbo LoRA) ≈ 327 s, eros draft 4 steps ≈ 205 s,
> eros standard 8 steps ≈ 325 s. The turbo-LoRA route also came out soft with
> near-silent audio; the Eros turbo files keep audio at a normal level.


### T2V Alternative Models (Swappable)

```
# Lokale LTX-2 19B-modellen zijn verwijderd (LTX-2.3 draait nu cloud-only).
# Verwijderd: ltx-2-19b-dev-Q4_K_M.gguf, ltx-2-19b-distilled_Q4_K_M.gguf,
# ltx-2-19b-distilled-fp8.safetensors, LTX-2-dev-Q2_K.gguf,
# ltx-2-19b-embeddings_connector_bf16.safetensors, LTX2_video_vae_bf16.safetensors
```

---

## 🖼️ Text-to-Image (T2I)

> **⚠️ IMAGE GENERATION IS FAST** - Unlike video, images take seconds not minutes!
> All benchmarks tested 2026-01-16 on RTX 5060 Ti 16GB.

```
T2I Model Categories (Kept: Flux + Flux 2 + SDXL-Pony + Krea 2)
│
├── ⚡ KREA 2 TURBO (snel, expressief — verified 2026-09-24)
│   │   "Krea 2 Turbo INT8 ConvRot - distilled, 8 steps, CFG 1.0"
│   │   832×1216 @ 8 steps: ~80-120s | dynamische VRAM-loading (aimdo/DisTorch)
│   │   Let op: guardian-LLM wordt vóór de run uit VRAM gezet (queue_prompt)
│   │
│   └── krea2_turbo_int8_convrot.safetensors
│       ├── Text encoder: qwen3vl_4b_bf16.safetensors (CLIPLoader type=krea2)
│       ├── VAE: qwen_image_vae.safetensors
│       ├── Sampler: euler + simple, CFG 1.0 (distilled — hoger = slechter)
│       ├── Steps: 4-20 (default 8)
│       └── LoRA: 1 slot (LoraLoaderModelOnly, /mnt/ssd/loras/krea2/)
│           ├── krea2/snofs_krea_v1_4.safetensors  [1.5GB] SNOFS v1.4 — GPU-verified
│           └── krea2/KNP_000003000.safetensors    [436MB] Krea 2 NSFW V4 — GPU-verified
│
├── 👑 MAX QUALITY (200-300 sec, multi-GPU)
│   │
│   └── 📦 flux2_dev
│       │   "Flux.2 Dev 32B (GGUF Q4) - nieuwste, meest kwaliteit"
│       │   1024×1024 @ 20 steps: ~297s | 2x GPU + CPU-offload
│       │   DisTorch2: cuda:1,8gb;cuda:0,4gb;cpu,*
│       │
│       └── flux2-dev-Q4_K_M.gguf  [19.9GB] ★ (nieuw)
│           ├── Sampler: euler + simple (via Flux2Scheduler)
│           ├── Steps: 20
│           ├── Guidance: 4.0 (géén negatieve prompt / CFG)
│           ├── Text encoder: mistral_3_small_flux2_fp8 (18GB, cpu)
│           └── Nodig: flux2-vae.safetensors (Mage one-step VAE)
│
├── 🐢 QUALITY (60-120 sec)
│   │
│   ├── 📦 flux_dev
│   │   │   "Flux.1 Dev FP8 - Highest Quality"
│   │   │   1024×1024 @ 20 steps: ~119s | VRAM: ~17GB
│   │   │
│   │   └── flux1-dev-fp8.safetensors                     [17GB]
│   │       ├── Sampler: euler + simple
│   │       ├── Steps: 20
│   │       ├── CFG: 3.5
│   │       └── Best for: Final renders, marketing
│   │
│   └── 📦 flux_nsfw
│       │   "Flux NSFW Variants (beste fotorealistische NSFW)"
│       │   1024×1024 @ 20 steps: ~75s | VRAM: ~12GB
│       │
│       ├── fluxedUpFluxNSFW_51FP8.safetensors            [12GB]
│       │   ├── 1024×1024 @ 20 steps: ~75s
│       │   └── CFG: 1.0 (classifier-free)
│       │
│       └── persephoneFluxNSFWSFW_11FP8.safetensors       [12GB]
│           └── NSFW/SFW dual mode
│
└── 🏃 FAST (15-30 sec)
    │
    └── 📦 sdxl_pony
        │   "SDXL-Pony (NSFW-specialist, grootste LoRA-ecosysteem)"
        │   1024×1024 @ 25 steps: ~18-28s | VRAM: ~8GB
        │
        ├── CyberRealistic_Pony_v14.1_FP16.safetensors    [6.5GB] ★
        │   ├── 1024×1024 @ 25 steps: ~18s
        │   ├── Category: Realistic/Pony (photoreal NSFW)
        │   ├── Sampler: dpmpp_2m + karras
        │   ├── CFG: 7.0
        │   └── Prompt: score_9, score_8_up prefixes
        │
        ├── ponyDiffusionV6XL_v6StartWithThisOne.safetensors [6.5GB]
        │   └── Category: Pony base (anime/stylized NSFW)
        │
        └── reapony_v90.safetensors                       [6.5GB]
            ├── Category: Realistic/Pony
            └── Best for: NSFW realistic content

### T2I Resolution Limits (SDXL)

```
SDXL Tested Resolutions (DreamShaper Lightning)
│
├── 1024×1024 (1:1 square): ✅ ~8s @ 8 steps
├── 1024×1536 (2:3 portrait): ✅ Scales linearly
├── 1536×1024 (3:2 landscape): ✅ Scales linearly
├── 832×1216 (SDXL native portrait): ✅ Optimal
├── 1216×832 (SDXL native landscape): ✅ Optimal
├── 2048×2048 (2K): ✅ ~4x time, ~16GB VRAM
└── 4096×4096 (4K): ⚠️ May OOM, use tiled VAE

Flux Tested Resolutions
│
├── Flux.1 Dev FP8
│   ├── 1024×1024: ✅ ~119s
│   ├── 1536×1536: ✅ ~4x time
│   └── 2048×2048: ⚠️ May need fp16 offload
│
└── Flux.2 Dev (GGUF Q4, multi-GPU)
    ├── 768×768: ✅ ~215s
    └── 1024×1024: ✅ ~297s (compute-kaart niet >75% vullen → OOM)
```

### T2I Optimal Settings Quick Reference

```
┌─────────────────────────────────────────────────────────────────────┐
│ SDXL-Pony (balanced speed/quality)                                    │
│ Model: CyberRealistic_Pony_v14.1_FP16.safetensors                   │
│ Steps: 25-30 | CFG: 7.0 | Sampler: dpmpp_2m + karras               │
│ Result: 1024×1024 in ~18-25 seconds                                 │
├─────────────────────────────────────────────────────────────────────┤
│ QUALITY PRIORITY (final renders)                                    │
│ Model: CyberRealistic_Pony_v14.1_FP16.safetensors                   │
│ Steps: 25-30 | CFG: 7.0 | Sampler: dpmpp_2m + karras               │
│ Result: 1024×1024 in ~18-25 seconds                                 │
├─────────────────────────────────────────────────────────────────────┤
│ BEST QUALITY (marketing, hero images)                               │
│ Model: flux1-dev-fp8.safetensors                                    │
│ Steps: 20 | CFG: 3.5 | Sampler: euler + simple                     │
│ Result: 1024×1024 in ~120 seconds                                   │
├─────────────────────────────────────────────────────────────────────┤
│ NSFW CONTENT                                                        │
│ SDXL: reapony_v90 or CyberRealistic_Pony (score_ prompts)          │
│ Flux: fluxedUpFluxNSFW_51FP8 (CFG 1.0)                             │
│ Prompt: Use Pony score tags for SDXL models                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Sub-Models Reference Tree

```
Sub-Models (Shared Components)
│
├── 📝 Text Encoders
│   │
│   ├── UMT5 Family (deleted 2026-10-01 — Wan 2.2 retired; the text encoder went with the family)
│   │   ├── umt5-xxl-enc-bf16.safetensors                   [11GB] ★ Primary
│   │   ├── umt5_xxl_fp8_e4m3fn.safetensors                 [5.7GB] Low VRAM
│   │   └── umt5_xxl_fp8_e4m3fn_scaled.safetensors          [6.7GB]
│   │
│   ├── Gemma Family (LTX-2)
│   │   ├── gemma-3-12b-it-qat-q4_0-unquantized/            [8GB] ★ Primary
│   │   ├── gemma-3-12b-it-q4_0.gguf                        [8GB] Alt GGUF
│   │   └── gemma_3_12B_it_nvfp4.safetensors                [8.3GB]
│   │
│   └── Qwen Family (Vision/General)
│       └── qwen_2.5_vl_7b_fp8_scaled.safetensors           [9.4GB]
│
├── 🎨 VAE Models
│   │
│   ├── wan_2.1_vae.safetensors                             [deleted 2026-10-01 — Wan 2.2 retired, weights removed]
│   ├── sdxl_vae.safetensors                                [335MB] → SDXL
│   ├── ae.safetensors                                      [335MB] → Flux
│   └── qwen_image_vae.safetensors                          [254MB] → Qwen
│
├── 👁️ CLIP Vision
│   │
│   └── clip_vision/
│       ├── clip_vision_h.safetensors                       [2.5GB] ★ Primary
│       └── SigLIP variants                                 (Alternative)
│
└── 🔗 Connectors
    │
    └── (LTX-2 lokale connector verwijderd — LTX-2.3 draait cloud-only)
```

---

## 🖥️ DisTorch2 GPU Distribution Tree

> **⚠️ PyTorch CUDA indices differ from nvidia-smi!**
> See [DISTORCH2_MULTI_GPU_SETTINGS.md](DISTORCH2_MULTI_GPU_SETTINGS.md) for full documentation.

```
Multi-GPU Distribution (28GB Total)
│
├── 🎮 cuda:1 (RTX 5060 Ti 16GB) ← PyTorch primary
│   │   nvidia-smi shows as GPU 1
│   │
│   ├── Typical Load: 12-16GB
│   │
│   └── Best for:
│       ├── Activations/KV-cache (needs contiguous memory)
│       ├── Main compute (100% GPU utilization)
│       └── Small model portion (when 3060 holds most)
│
├── 🎮 cuda:0 (RTX 3060 12GB) ← PyTorch secondary
│   │   nvidia-smi shows as GPU 0
│   │
│   ├── Typical Load: 10-12GB
│   │
│   └── Best for:
│       ├── Model weight storage (97% of model)
│       ├── Freeing cuda:1 for activations
│       └── PUT THIS FIRST in allocation for long videos
│
└── 💾 cpu (System RAM - Emergency)
    │
    └── Spillover for:
        ├── Overflow when >28GB needed
        └── Very high resolution/frames
```

### Optimal Allocation Strings

```
ALLOCATION STRING FORMAT: device,size;device,size;cpu,*

CRITICAL: Order determines which GPU gets model FIRST!

┌─────────────────────────────────────────────────────────────────────┐
│ MAX VIDEO LENGTH (480p Portrait, ~22 sec)                           │
│ cuda:0,11gb;cuda:1,15gb;cpu,*                                       │
│   → 3060 holds 97% of model (~11GB)                                 │
│   → 5060 Ti has 15GB free for activations                           │
│   → Max: 353-355 frames                                             │
├─────────────────────────────────────────────────────────────────────┤
│ BALANCED (faster, 10 sec max)                                       │
│ cuda:1,10gb;cuda:0,4gb;cpu,*                                        │
│   → 5060 Ti holds most of model                                     │
│   → Faster due to less memory transfers                             │
│   → Max: ~161 frames                                                │
├─────────────────────────────────────────────────────────────────────┤
│ CPU OFFLOAD (longest, slowest)                                      │
│ cuda:0,8gb;cuda:1,12gb;cpu,*                                        │
│   → Part of model on CPU RAM                                        │
│   → Slowest due to PCIe transfers                                   │
│   → For 400+ frames if needed                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### VRAM Budget by Resolution (tested 2026-01-16)

```
480 × 848 (Portrait) - RECOMMENDED FOR LONG VIDEOS
│
├── cuda:0,11gb;cuda:1,15gb;cpu,*
│   ├── 161 frames (~10 sec): ✅ ~22GB SAFE
│   ├── 241 frames (~15 sec): ✅ ~24GB SAFE
│   ├── 321 frames (~20 sec): ✅ ~26GB SAFE ← RECOMMENDED MAX
│   ├── 341 frames (~21 sec): ✅ ~27GB TIGHT
│   ├── 351-355 frames: ⚠️ Works sometimes, OOM risk
│   └── 357+ frames: ❌ OOM
│
└── cuda:1,10gb;cuda:0,4gb;cpu,* (balanced)
    └── 161 frames max: ✅ ~20GB

576 × 1024 (Standard Portrait)
│
├── 🟢 81 frames (~5 sec): ✅ ~24GB
├── 🟡 121 frames: ⚠️ Tight
└── 🔴 161+ frames: ❌ OOM

720 × 1280 (HD Portrait)
│
├── 41 frames with CPU offload: ✅
└── 81+ frames: ❌ OOM

RECOMMENDED SETTINGS FOR PRODUCTION:
├── Maximum video length: 321 frames (~20 sec)
├── Allocation: cuda:0,11gb;cuda:1,15gb;cpu,*
└── Headroom: 1-2GB for stability

GENERATION TIMES (6 steps, uni_pc sampler):
├── 81 frames:  ~50-60s/step  → ~5-6 min total
├── 161 frames: ~110-120s/step → ~12 min total
├── 321 frames: ~227s/step    → ~23 min total
└── Scaling: ~linear with frame count
```

---

## 📁 File Location Tree

```
ComfyUI/models/
│
├── unet/
│   └── (Wan 2.2 files removed — model retired from the product)
│
├── diffusion_models/
│   └── flux2-dev-Q4_K_M.gguf  (MiniMax-H3 local modellen draaien op de Windows-PC server)
│
├── text_encoders/
│   ├── umt5-xxl-enc-bf16.safetensors  (parked — Wan 2.2 retired; kept for a possible future Wan 3)
│   ├── gemma-3-12b-it-qat-q4_0-unquantized/
│   ├── qwen3vl_4b_bf16.safetensors
│   └── ...
│
├── vae/
│   ├── wan_2.1_vae.safetensors  (deleted 2026-10-01 — Wan 2.2 retired, weights removed)
│   ├── sdxl_vae.safetensors
│   ├── ae.safetensors
│   └── flux2-vae.safetensors
│
├── checkpoints/
│   ├── flux1-dev-fp8.safetensors
│   ├── CyberRealistic_Pony_v14.1_FP16.safetensors
│   ├── ponyDiffusionV6XL_v6StartWithThisOne.safetensors
│   ├── reapony_v90.safetensors
│   └── ...
│
├── clip_vision/
│   └── clip_vision_h.safetensors
│
└── loras/
    └── [LoRA files]
```

---

## 🎵 Audio Generation (LTX-2 Audio)

```
Audio Generation Modes
│
└── 📦 ltx2_audio (experimental)
    │   # REMOVED in cleanup — lokale LTX-2 draait niet meer (LTX-2.3 cloud-only).
    │   # MiniMax-H3 FL2VA genereert nu native stereo audio (geen aparte audio-stap).
```

---

## ⚙️ POST-PROCESSING OPTIONS

> **UNIFIED POST-PROCESSING SYSTEM (2026-01-17)**
>
> Post-processing is now available in TWO ways:
> 1. **Inline on I2V/T2V tools** - Checkbox options that run automatically after generation
> 2. **Standalone Post-Processing tool** - Under Advanced, for existing/uploaded media

### 🔄 Inline Post-Processing (I2V/T2V)

```
Inline Options (Chained Jobs)
│
│   These checkboxes appear on I2V and T2V tools.
│   They run AUTOMATICALLY after generation completes.
│
├── ☑️ Upscale Video (Real-ESRGAN)
│   ├── 2x upscale (default)
│   └── 4x upscale
│   💰 +5 credits
│
├── ☑️ Frame Interpolation (RIFE)
│   ├── 30 fps
│   ├── 48 fps
│   └── 60 fps (default)
│   💰 +3 credits
│
└── ☑️ Add Audio Track (I2V only)
    └── Upload audio file to attach
    💰 +0 credits (included)
```

### 🛠️ Standalone Post-Processing Tool

```
Post-Processing Tool (Advanced → Post-Processing)
│
│   Process EXISTING or UPLOADED media without regeneration.
│   Location: Advanced section in navigation
│
├── 📈 Upscale Mode
│   │   "Real-ESRGAN video upscaling"
│   │   Input: Single video
│   │
│   ├── Models:
│   │   └── realesrgan-x4plus.pth
│   ├── Scale options: 2x, 4x
│   └── 💰 5 credits
│
├── 🔄 Interpolate Mode
│   │   "RIFE frame interpolation"
│   │   Input: Single video
│   │
│   ├── Model: rife_v4.6
│   ├── Target FPS: 30, 48, 60
│   └── 💰 3 credits
│
└── 🔗 Concat Mode
    │   "Join multiple videos"
    │   Input: 2+ videos
    │
    ├── Preserves resolution of first video
    └── 💰 2 credits
```

### ComfyUI Workflow Builders

```
Backend Implementation
│
├── build_video_upscale_workflow()
│   └── Real-ESRGAN frame-by-frame upscaling
│
├── build_rife_workflow()
│   └── RIFE v4.6 interpolation
│
└── build_video_concat_workflow()
    └── FFmpeg-based video concatenation
```

### 📈 Upscaling (Legacy Reference)

```
Upscaling Options
│
├── 🖼️ Image Upscaling
│   │
│   ├── 📦 realesrgan_image
│   │   ├── RealESRGAN_x4plus.pth            → General 4x upscale
│   │   └── RealESRGAN_x4plus_anime_6B.pth   → Anime optimized
│   │
│   └── 📦 face_restore
│       ├── GFPGANv1.4.pth                   → Face enhancement
│       └── CodeFormer                        → Alternative face fix
│
└── 🎬 Video Upscaling
    │
    └── 📦 realesrgan_video
        │   "Frame-by-frame upscaling"
        │   Status: ✅ PRODUCTION (via Post-Processing tool)
        │
        └── ⚠️ Note: Slow - upscales each frame individually
```

### 🔄 Frame Interpolation (Legacy Reference)

```
Frame Interpolation Options
│
└── 📦 rife_v4
    │   "RIFE v4.6 - Increase FPS / Smooth motion"
    │   Status: ✅ PRODUCTION (via Post-Processing tool)
    │
    ├── 🧠 Model: rife_v4.6 (ComfyUI-Frame-Interpolation)
    │
    ├── Use cases:
    │   ├── 16fps → 32fps (2x interpolation)
    │   ├── 16fps → 48fps (3x interpolation)
    │   └── Slow motion effects
    │
    └── ⚠️ Note: Does NOT increase video length, only smoothness
```

### 💋 Audio Sync / LipSync

```
Audio Sync Options
│
├── 📦 wav2lip
│   │   "Wav2Lip - Sync lips to audio"
│   │   Status: 📋 PLANNED
│   │
│   ├── 🧠 Model: wav2lip_gan.pth
│   └── Use case: Make video character "speak" audio
│
└── 📦 audio_attach
    │   "Simple audio track attachment"
    │   Status: ✅ PRODUCTION (via I2V inline option)
    │
    └── Use case: Add background music/sfx to video
```

---

## 🔗 PIPELINES (Tool Combinations)

> **Pipelines combine multiple tools** into a single workflow.

### 🗣️ Speech-to-Video Pipeline

```
SpeechToVideo Pipeline
│
│   Input: Text script + Reference image
│   Output: Video of character speaking the text
│
├── Step 1: TextToSpeech (TTS)
│   └── Generate audio from text script
│
├── Step 2: ImageToVideo (I2V)
│   └── Animate the reference image
│
└── Step 3: LipSync (Post-process)
    └── Sync lips to generated audio

Status: 📋 PLANNED - Requires TTS + LipSync integration
```

### 📺 Video-to-Video Pipeline

```
VideoToVideo Pipeline
│
│   Input: Source video + Style prompt
│   Output: Restyled video
│
├── Step 1: Extract frames from source video
│
├── Step 2: I2V on keyframes OR I2I on each frame
│
└── Step 3: Stitch frames back together

Status: 📋 PLANNED
Workflow: Uses I2V pipeline with frame extraction
```

### 🎬 T2I→I2V Pipeline (MiniMax-H3 I2V)

```
T2I→I2V Pipeline (PRODUCTION)
│
│   Input: Text prompt
│   Output: Video
│
├── Step 1: TextToImage (T2I)
│   └── Flux/SDXL generates starting frame
│
└── Step 2: ImageToVideo (I2V)
    └── MiniMax-H3 (leading/default) or LTX-2.3 animates the image

Status: ✅ PRODUCTION
Workflow: T2I workflow → I2V mode (minimax_h3 cloud/local, ltx2 cloud)
```

---

## 📝 Image-to-Text (Captioning)

```
Image-to-Text Modes
│
├── 📦 florence2 (planned default)
│   │   "Florence-2 Vision Captioning"
│   │   Status: 📋 PLANNED
│   │
│   └── 🧠 Model
│       └── microsoft/Florence-2-large
│
└── 📦 llava
    │   "LLaVA Vision Language Model"
    │   Status: 📋 PLANNED
    │
    └── 🧠 Model
        └── llava-1.5-7b-hf
```

---

## 🖼️ Image-to-Image (Style Transfer)

```
Image-to-Image Modes
│
├── 📦 img2img_sdxl
│   │   "SDXL Image-to-Image"
│   │   Status: 🔨 IN DEVELOPMENT
│   │
│   ├── 📂 Workflow: workflows/ImageToImage/
│   └── 🧠 Uses T2I checkpoints with denoise control
│
└── 📦 controlnet
    │   "ControlNet Guided Generation"
    │   Status: 📋 PLANNED
    │
    └── ControlNet models (canny, depth, pose, etc.)
```

---

## 🎭 FaceSwap

```
FaceSwap Modes
│
└── 📦 insightface (planned)
    │   "InsightFace/ReActor"
    │   Status: 📋 PLANNED
    │
    ├── 📂 Workflow: [TBD]
    │
    └── 🧠 Models
        ├── inswapper_128.onnx
        └── GFPGANv1.4.pth (face restore)
```

---

## 🔊 Text-to-Speech (TTS)

```
TTS Modes
│
├── 📦 xtts (planned)
│   │   "Coqui XTTS v2 - Voice Cloning"
│   │   Status: 📋 PLANNED
│   │
│   └── 🧠 Model
│       └── XTTS-v2/
│
└── 📦 elevenlabs (external)
    │   "ElevenLabs API"
    │   Status: 💡 FUTURE
    │
    └── Requires API key
```

---

## Tested Configurations Log

> **⚠️ ADD ENTRIES HERE AFTER EVERY SUCCESSFUL TEST!**
> This is the MOST IMPORTANT section - proves what actually works.

```
┌─────────────────────────────────────────────────────────────────────┐
│ ✅ PRODUCTION-READY CONFIGURATIONS (Copy-paste ready!)              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ ═══════════════════════════════════════════════════════════════════ │
│ 🎬 VIDEO (I2V/T2V) - MINIMAX-H3 FL2VA (LEADING - CLOUD + LOCAL)    │
│ ═══════════════════════════════════════════════════════════════════ │
│                                                                      │
│ 768×1344 Cloud Benchmark (A40)                                       │
│ ─────────────────────────────────────────────────────────────────── │
│   Frames: 124 (~5 sec @ 24fps) | Max: 362 frames                    │
│   Models:                                                            │
│     - minimax_h3_fl2va_pruned_int8_convrot.safetensors              │
│     - qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors                  │
│   Timing: official 20 steps ≈ 689s | standard 8 ≈ 327s |            │
│   eros draft 4 ≈ 205s (variant-dependent, see T2V section)          │
│   Worker: RunPod 80GB+ (local: Windows-PC ComfyUI, 16GB GPU)        │
│   Tested: ✅ (A40 cloud benchmark)                                   │
│                                                                      │
│ ═══════════════════════════════════════════════════════════════════ │
│ 🎥 TEXT-TO-VIDEO (T2V) - LTX-2 19B (LOKALE MODUS VERWIJDERD — zie hieronder)│
│ ═══════════════════════════════════════════════════════════════════ │
│  # Historische benchmark (2026-01-12). De lokale LTX-2 19B-modellen │
│  # (ltx-2-19b-distilled_Q4_K_M.gguf, LTX2_video_vae_bf16,           │
│  # ltx-2-19b-embeddings_connector_bf16) en workflow ltx2_distorch2_ │
│  # multigpu_api.json zijn verwijderd — LTX-2.3 draait nu cloud-only.│
│                                                                      │
│ 768×512 Landscape                                                    │
│ ─────────────────────────────────────────────────────────────────── │
│   Frames: 97 (~6 sec) | VRAM: ~22GB | Time: ~8 min                  │
│   Model: ltx-2-19b-distilled_Q4_K_M.gguf                            │
│   CLIP: gemma-3-12b-it-qat-q4_0-unquantized/                        │
│   VAE: LTX2_video_vae_bf16.safetensors                              │
│   Connector: ltx-2-19b-embeddings_connector_bf16.safetensors        │
│   Workflow: ltx2_distorch2_multigpu_api.json                        │
│   Tested: 2026-01-12 ✅                                              │
│                                                                      │
│ ═══════════════════════════════════════════════════════════════════ │
│ 🖼️ TEXT-TO-IMAGE (T2I) - FLUX (Benchmarked 2026-01-16)              │
│ ═══════════════════════════════════════════════════════════════════ │
│                                                                      │
│ Flux Dev FP8 - 1024×1024                                             │
│ ─────────────────────────────────────────────────────────────────── │
│   Model: flux1-dev-fp8.safetensors [17GB]                           │
│   VRAM: ~17GB | Time: 119s (2 min)                                  │
│   Steps: 20 | CFG: 3.5 | Sampler: euler + simple                    │
│   Quality: HIGHEST - use for final renders                          │
│   Tested: 2026-01-16 ✅                                              │
│                                                                      │
│ FluxedUp NSFW - 1024×1024                                            │
│ ─────────────────────────────────────────────────────────────────── │
│   Model: fluxedUpFluxNSFW_51FP8.safetensors [12GB]                  │
│   VRAM: ~12GB | Time: 75s                                           │
│   Steps: 20 | CFG: 1.0 | Sampler: euler + simple                    │
│   Quality: HIGH - faster than flux dev                              │
│   Tested: 2026-01-16 ✅                                              │
│                                                                      │
│ ═══════════════════════════════════════════════════════════════════ │
│ 🖼️ TEXT-TO-IMAGE (T2I) - SDXL (Benchmarked 2026-01-16)              │
│ ═══════════════════════════════════════════════════════════════════ │
│                                                                      │
│ Pony Diffusion V6 - 1024×1024                                         │
│ ─────────────────────────────────────────────────────────────────── │
│   Model: ponyDiffusionV6XL_v6StartWithThisOne.safetensors [6.5GB]   │
│   Steps: 28 | CFG: 7.0 | Sampler: dpmpp_2m + karras                 │
│   Prompt: score_9, score_8_up prefixes                              │
│   Best for: anime/stylized NSFW, booru tags                         │
│   Tested: 2026-01-16 ✅                                              │
│                                                                      │
│ CyberRealistic Pony - 1024×1024 (FASTEST STANDARD)                   │
│ ─────────────────────────────────────────────────────────────────── │
│   Model: CyberRealistic_Pony_v14.1_FP16.safetensors [6.5GB]         │
│   VRAM: ~8GB | Time: 18s @ 25 steps                                 │
│   Steps: 25 | CFG: 7.0 | Sampler: dpmpp_2m + karras                 │
│   Prompt: score_9, score_8_up prefixes                              │
│   Quality: HIGH realistic                                           │
│   Tested: 2026-01-16 ✅                                              │
│                                                                      │
│ Reapony - 1024×1024                                                  │
│ ─────────────────────────────────────────────────────────────────── │
│   Model: reapony_v90.safetensors [6.5GB]                            │
│   VRAM: ~8GB | Time: 27s @ 25 steps                                 │
│   Steps: 25 | CFG: 7.0 | Sampler: dpmpp_2m + karras                 │
│   Best for: NSFW realistic content                                  │
│   Tested: 2026-01-16 ✅                                              │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ 🔬 EXPERIMENTAL / NEEDS MORE TESTING                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ 576×1024 Extended (121+ frames)                                      │
│ ─────────────────────────────────────────────────────────────────── │
│   Status: 🔨 NEEDS TESTING - expected ~27GB VRAM                     │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ ❌ KNOWN FAILURES / DO NOT USE                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ 480×848 @ 357+ frames: OOM even with optimal allocation              │
│ 576×1024 @ 161+ frames: OOM                                          │
│ 720×1280 @ 81+ frames: OOM                                           │
│ realvisxlV50_v50Bakedvae.safetensors: REMOVED - do not reference     │
│ umt5-xxl-enc-bf16-uncensored.safetensors: RENAMED to umt5-xxl-enc-bf16│
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Quick Copy-Paste: Optimal I2V Settings

```json
{
  "expert_mode_allocations": "cuda:0,11gb;cuda:1,15gb;cpu,*",
  "compute_device": "cuda:1",
  "donor_device": "cuda:0",
  "virtual_vram_gb": 16,
  "eject_models": true
}
```

### Quick Reference: What Works Right Now

> **⚠️ VIDEO: MINIMUM 81 FRAMES (5 sec) - Anything less is UNACCEPTABLE!**

#### 🎬 Video Generation

| Tool | Mode | Resolution | Frames | Duration | Time | Status |
|------|------|------------|--------|----------|------|--------|
| I2V/T2V | minimax_h3 (cloud, leading) | 768×1344 | 124 | **~5 sec** | ~3.5-11.5 min | ✅ DEFAULT |
| I2V/T2V | minimax_h3 (local Windows-PC) | — | — | — | — | ✅ |
| I2V/T2V | ltx2 (cloud, second choice) | — | — | — | — | ✅ |

MiniMax-H3 cloud timing (A40): draft 4 ≈ 205s, standard 8 ≈ 327s,
official 20 ≈ 689s — see the T2V section and Tested Configurations Log.

#### 🖼️ Image Generation (Benchmarked 2026-01-16)

| Model | Resolution | Steps | Time | VRAM | Use Case |
|-------|------------|-------|------|------|----------|
| DreamShaper Lightning | 1024×1024 | 8 | **8s** | 8GB | ⚡ Fastest previews |
| CyberRealistic Pony | 1024×1024 | 25 | **18s** | 8GB | 🎯 Best quality/speed |
| Reapony | 1024×1024 | 25 | **27s** | 8GB | 🔞 NSFW realistic |
| Juggernaut | 1024×1024 | 25 | **28s** | 8GB | 🎨 Artistic |
| Nova Anime | 1024×1024 | 25 | **28s** | 8GB | 🎌 Anime |
| FluxedUp NSFW | 1024×1024 | 20 | **75s** | 12GB | 🔥 Quality NSFW |
| Flux Dev FP8 | 1024×1024 | 20 | **119s** | 17GB | 👑 HIGHEST quality |
| Flux.2 Dev GGUF | 1024×1024 | 20 | **297s** | 19GB+2GPU | 🌌 Nieuwste 32B (nieuw) |

#### ⏱️ T2I Speed Tiers

```
⚡ LIGHTNING (5-10s)     → DreamShaper Lightning @ 8 steps
🏃 FAST (15-30s)         → SDXL models @ 25 steps
🐢 QUALITY (60-120s)     → Flux models @ 20 steps
🐌 MAX (200-300s)        → Flux.2 Dev 32B (multi-GPU)
```

### ❌ REMOVED - Under 5 seconds (UNACCEPTABLE)

| Resolution | Frames | Duration | Reason |
|------------|--------|----------|--------|
| 720×1280 | 41 | 2.5 sec | Too short, useless |
| 848×480 T2V | 41 | 2.5 sec | Too short, useless |
| Any | <81 | <5 sec | **NO PRODUCTION USE** |

---

**Legend:**
- 📦 = Generation Mode
- 🧠 = Diffusion Model
- 📝 = Text Encoder
- 🎨 = VAE
- 👁️ = CLIP Vision
- ✨ = LoRA
- ★ = Primary/Recommended
- [SIZE] = File size / VRAM requirement
- ✅ = Production ready
- 🔨 = In development
- 📋 = Planned
- 💡 = Future

---

## Maintenance Instructions

### When to Update This Document

1. **After successful ComfyUI generation** - Add to "Tested Configurations Log"
2. **After adding new model** - Add to appropriate tool section
3. **After testing new resolution/frame combo** - Update VRAM budgets
4. **After workflow changes** - Update workflow file references

### How to Add New Generation Mode

```markdown
## [Tool Name]

\```
[Tool] Generation Modes
│
└── 📦 [mode_id] ([variant])
    │   "[Display Name]"
    │   Status: [✅|🔨|📋|💡] [STATUS]
    │
    ├── 📂 Workflow: workflows/[path]
    │
    ├── 🧠 Diffusion Model
    │   └── [model_file.gguf]
    │
    ├── 📝 Text Encoder
    │   └── [encoder_file]
    │
    └── 🎨 VAE
        └── [vae_file]
\```
```

---

**Last Updated**: 2026-09-30
**Maintainer**: @copilot (auto-update on generation complete - FUTURE)
