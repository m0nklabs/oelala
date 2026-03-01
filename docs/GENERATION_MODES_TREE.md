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
│   ├── 🎬 T2I→I2V Pipeline     → T2I + I2V (Wan2.2 T2V method)      │
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
| **480p** | Wan2.2 | **20 sec** | 321 | ~26GB |
| 576p | Wan2.2 | 7 sec | 113 | ~24GB |
| 720p | Wan2.2 | 4 sec | 65 | ~27GB |
| **480p** | LTX-2 | **12 sec** | 97 | ~18GB |
| 576p | LTX-2 | 8 sec | 97 | ~20GB |
| 720p | LTX-2 | 5 sec | 65 | ~22GB |

```
I2V Generation Modes
│
├── 📦 wan2.2 (default)
│   │   "Wan2.2 14B Q6_K DisTorch2"
│   │   Dual-pass (high/low noise) • Best quality
│   │
│   ├── 🧠 Diffusion Models (Dual-Pass)
│   │   ├── wan2.2_i2v_high_noise_14B_Q6_K.gguf  [12GB]
│   │   └── wan2.2_i2v_low_noise_14B_Q6_K.gguf   [12GB]
│   │
│   ├── 📝 Text Encoder
│   │   └── umt5-xxl-enc-bf16.safetensors        [11GB]
│   │
│   ├── 👁️ CLIP Vision
│   │   └── clip_vision_h.safetensors            [2.5GB]
│   │
│   └── 🎨 VAE
│       └── wan_2.1_vae.safetensors              [242MB]
│
├── 📦 ltx2 (NEW - 2026-01-17)
│   │   "LTX-2 19B Q4_K_M"
│   │   Single model • Faster inference • Uses Gemma encoder
│   │
│   ├── 🧠 Diffusion Model (Single)
│   │   └── ltx-2-19b-dev-Q4_K_M.gguf            [12GB]
│   │       OR ltx-2-19b-distilled_Q4_K_M.gguf   [12GB] (faster)
│   │
│   ├── 📝 Text Encoder (Gemma 3)
│   │   ├── gemma_3_12B_it_nvfp4.safetensors     [8GB]
│   │   └── ltx-2-19b-embeddings_connector_bf16  [2.9GB]
│   │
│   └── 🎨 VAE
│       └── LTX2_video_vae_bf16.safetensors      [2.5GB]
│
└── 📦 nsfw_lora
    │   "Wan2.2 Enhanced NSFW with LoRAs"
    │
    ├── 🧠 Diffusion Models (Dual-Pass)
    │   ├── wan2.2_i2v_high_noise_14B_Q6_K.gguf  [12GB]
    │   └── wan2.2_i2v_low_noise_14B_Q6_K.gguf   [12GB]
    │
    ├── 📝 Text Encoder
    │   └── umt5-xxl-enc-bf16.safetensors        [11GB]
    │
    ├── 👁️ CLIP Vision
    │   └── clip_vision_h.safetensors            [2.5GB]
    │
    ├── 🎨 VAE
    │   └── wan_2.1_vae.safetensors              [242MB]
    │
    └── ✨ LoRAs (pre-configured)
        └── [NSFW LoRAs as defined in workflow]
```

### I2V Model Comparison

| Feature | Wan2.2 | LTX-2 |
|---------|--------|-------|
| Pass Type | Dual (high/low noise) | Single |
| Default Steps | 6 | 20 |
| Default CFG | 1.0 | 3.0 |
| Text Encoder | UMT5-XXL | Gemma 3 |
| Best For | Quality, longer clips | Speed, shorter clips |
| LoRA Support | ✅ Yes | ❌ No (native) |
| Max Duration (480p) | 20 sec | 12 sec |

### I2V Alternative Models (Swappable - Wan2.2 only)

```
Alternative High/Low Noise Pairs
│
├── ⚡ Lightning (faster inference)
│   ├── Wan22-I2V_A14B-Lightning-H-Q6_K.gguf
│   └── Wan22-I2V_A14B-Lightning-L-Q6_K.gguf
│
├── 🎭 SmoothMix (community merge)
│   ├── smoothMixWan22GGUF_highQ6K.gguf
│   └── smoothMixWan22GGUF_lowQ6K.gguf
│
├── 🔞 Enhanced NSFW V2
│   ├── wan22EnhancedNSFW_V2_Q6K_HIGH.gguf
│   └── wan22EnhancedNSFW_V2_Q6K_LOW.gguf
│
└── 📷 Enhanced Camera Motion
    ├── wan22EnhancedNSFWCameraPrompt_nsfwV2Q6KH.gguf
    └── wan22EnhancedNSFWCameraPrompt_nsfwV2Q6KL.gguf
```

---

## 🎥 Text-to-Video (T2V)

```
T2V Generation Modes
│
├── 📦 wan22 (default)
│   │   "Wan2.2 14B T2V (T2I → I2V pipeline)"
│   │   Max frames: 81 | Default: 41
│   │
│   ├── 🧠 Diffusion Models (FP8)
│   │   ├── wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors  [14GB]
│   │   └── wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors   [14GB]
│   │
│   ├── 📝 Text Encoder
│   │   └── umt5-xxl-enc-bf16.safetensors                     [11GB]
│   │
│   └── 🎨 VAE
│       └── wan_2.1_vae.safetensors                           [242MB]
│
└── 📦 ltx2
    │   "LTX-2 19B Distilled (Direct T2V)"
    │   Max frames: 97 | Default: 25
    │
    ├── 🧠 Diffusion Model (GGUF)
    │   └── ltx-2-19b-distilled_Q4_K_M.gguf                   [12GB]
    │
    ├── 📝 Text Encoder (Gemma)
    │   ├── gemma-3-12b-it-qat-q4_0-unquantized/              [8GB]
    │   └── ltx-2-19b-embeddings_connector_bf16.safetensors   [2.9GB]
    │
    └── 🎨 VAE
        └── LTX2_video_vae_bf16.safetensors                   [2.5GB]
```

### T2V Alternative Models (Swappable)

```
Alternative LTX-2 Variants
│
├── 🔬 LTX-2 Dev (potentially higher quality)
│   └── ltx-2-19b-dev-Q4_K_M.gguf                            [12GB]
│
├── 💾 LTX-2 Dev Q2 (lower VRAM)
│   └── LTX-2-dev-Q2_K.gguf                                  [7.5GB]
│
└── 🎯 LTX-2 FP8 Full (highest quality)
    └── ltx-2-19b-distilled-fp8.safetensors                  [27GB]
```

---

## 🖼️ Text-to-Image (T2I)

> **⚠️ IMAGE GENERATION IS FAST** - Unlike video, images take seconds not minutes!
> All benchmarks tested 2026-01-16 on RTX 5060 Ti 16GB.

```
T2I Model Categories (Speed Tiers)
│
├── ⚡ LIGHTNING FAST (5-15 sec)
│   │
│   ├── 📦 sdxl_lightning
│   │   │   "DreamShaper XL Lightning - FASTEST SDXL"
│   │   │   1024×1024 @ 8 steps: ~8s | VRAM: ~8GB
│   │   │
│   │   └── dreamshaperXL_lightningDPMSDE.safetensors     [6.5GB]
│   │       ├── Sampler: dpmpp_sde + karras
│   │       ├── Steps: 8 (optimized for lightning)
│   │       ├── CFG: 2.0
│   │       └── Best for: Quick iterations, previews
│   │
│   └── 📦 z_image_turbo
│       │   "Z-Image Turbo - Experimental Fast Model"
│       │   Status: 🔨 NEEDS TESTING
│       │
│       └── z_image_turbo_bf16.safetensors                [12GB]
│
├── 🏃 FAST (15-30 sec)
│   │
│   └── 📦 sdxl_standard
│       │   "SDXL Standard Checkpoints"
│       │   1024×1024 @ 25 steps: ~18-28s | VRAM: ~8GB
│       │
│       ├── CyberRealistic_Pony_v14.1_FP16.safetensors    [6.5GB] ★ FASTEST
│       │   ├── 1024×1024 @ 25 steps: ~18s
│       │   ├── Category: Realistic/Pony
│       │   ├── Sampler: dpmpp_2m + karras
│       │   ├── CFG: 7.0
│       │   └── Prompt: score_9, score_8_up prefixes
│       │
│       ├── reapony_v90.safetensors                       [6.5GB]
│       │   ├── 1024×1024 @ 25 steps: ~27s
│       │   ├── Category: Realistic/Pony
│       │   └── Best for: NSFW realistic content
│       │
│       ├── juggernautXL_ragnarok.safetensors             [7GB]
│       │   ├── 1024×1024 @ 25 steps: ~28s
│       │   └── Category: General/Artistic
│       │
│       ├── novaAnimeXL_ilV150.safetensors                [6.5GB]
│       │   ├── 1024×1024 @ 25 steps: ~28s
│       │   └── Category: Anime
│       │
│       ├── illustriousRealismBy_v10VAE.safetensors       [6.5GB]
│       │   └── Category: Realistic
│       │
│       ├── ponyDiffusionV6XL_v6StartWithThisOne.safetensors [6.5GB]
│       │   └── Category: Pony base
│       │
│       ├── ultraRealisticByStable_v20FP16.safetensors    [6.5GB]
│       │   └── Category: Photorealistic
│       │
│       └── waiIllustriousSDXL_v160.safetensors           [6.5GB]
│           └── Category: Anime/Illustrious
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
│   ├── 📦 flux_nsfw
│   │   │   "Flux NSFW Variants"
│   │   │   1024×1024 @ 20 steps: ~75s | VRAM: ~12GB
│   │   │
│   │   ├── fluxedUpFluxNSFW_51FP8.safetensors            [12GB]
│   │   │   ├── 1024×1024 @ 20 steps: ~75s
│   │   │   └── CFG: 1.0 (classifier-free)
│   │   │
│   │   └── persephoneFluxNSFWSFW_11FP8.safetensors       [12GB]
│   │       └── NSFW/SFW dual mode
│   │
│   └── 📦 wan22_t2i
│       │   "Wan2.2 T2I (video-compatible images)"
│       │   Status: 🔨 NEEDS TESTING
│       │
│       └── Uses T2V models for T2I
│           └── Best for: Images that will become video
│
└── 📦 sd15
    │   "SD 1.5 (legacy, fast)"
    │   512×512 @ 25 steps: ~5s | VRAM: ~4GB
    │
    └── Realistic_Vision_V5.1.safetensors                 [4GB]
        └── Best for: Quick low-res previews
```

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
├── 1024×1024: ✅ ~119s
├── 1536×1536: ✅ ~4x time
└── 2048×2048: ⚠️ May need fp16 offload
```

### T2I Optimal Settings Quick Reference

```
┌─────────────────────────────────────────────────────────────────────┐
│ SPEED PRIORITY (previews, iterations)                               │
│ Model: dreamshaperXL_lightningDPMSDE.safetensors                    │
│ Steps: 8 | CFG: 2.0 | Sampler: dpmpp_sde + karras                  │
│ Result: 1024×1024 in ~8 seconds                                     │
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
│   ├── UMT5 Family (Wan2.2)
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
│       ├── qwen_2.5_vl_7b_fp8_scaled.safetensors           [9.4GB]
│       └── qwen_3_4b.safetensors                           [8GB]
│
├── 🎨 VAE Models
│   │
│   ├── wan_2.1_vae.safetensors                             [242MB] → Wan2.2
│   ├── LTX2_video_vae_bf16.safetensors                     [2.5GB] → LTX-2
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
    └── ltx-2-19b-embeddings_connector_bf16.safetensors     [2.9GB] → LTX-2
```

---

## 🖥️ DisTorch2 GPU Distribution Tree

> **⚠️ PyTorch CUDA indices differ from nvidia-smi!**
> See [DISTORCH2_MULTI_GPU_SETTINGS.md](DISTORCH2_MULTI_GPU_SETTINGS.md) for full documentation.

```
Multi-GPU Distribution (28GB Total)
│
├── 🎮 cuda:0 (RTX 5060 Ti 16GB) ← PyTorch primary
│   │   nvidia-smi shows as GPU 1
│   │
│   ├── Typical Load: 12-16GB
│   │
│   └── Best for:
│       ├── Activations/KV-cache (needs contiguous memory)
│       ├── Main compute (100% GPU utilization)
│       └── Small model portion (when 3060 holds most)
│
├── 🎮 cuda:1 (RTX 3060 12GB) ← PyTorch secondary
│   │   nvidia-smi shows as GPU 0
│   │
│   ├── Typical Load: 10-12GB
│   │
│   └── Best for:
│       ├── Model weight storage (97% of model)
│       ├── Freeing cuda:0 for activations
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
│ cuda:1,11gb;cuda:0,15gb;cpu,*                                       │
│   → 3060 holds 97% of model (~11GB)                                 │
│   → 5060 Ti has 15GB free for activations                           │
│   → Max: 353-355 frames                                             │
├─────────────────────────────────────────────────────────────────────┤
│ BALANCED (faster, 10 sec max)                                       │
│ cuda:0,10gb;cuda:1,4gb;cpu,*                                        │
│   → 5060 Ti holds most of model                                     │
│   → Faster due to less memory transfers                             │
│   → Max: ~161 frames                                                │
├─────────────────────────────────────────────────────────────────────┤
│ CPU OFFLOAD (longest, slowest)                                      │
│ cuda:1,8gb;cuda:0,12gb;cpu,*                                        │
│   → Part of model on CPU RAM                                        │
│   → Slowest due to PCIe transfers                                   │
│   → For 400+ frames if needed                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### VRAM Budget by Resolution (tested 2026-01-16)

```
480 × 848 (Portrait) - RECOMMENDED FOR LONG VIDEOS
│
├── cuda:1,11gb;cuda:0,15gb;cpu,*
│   ├── 161 frames (~10 sec): ✅ ~22GB SAFE
│   ├── 241 frames (~15 sec): ✅ ~24GB SAFE
│   ├── 321 frames (~20 sec): ✅ ~26GB SAFE ← RECOMMENDED MAX
│   ├── 341 frames (~21 sec): ✅ ~27GB TIGHT
│   ├── 351-355 frames: ⚠️ Works sometimes, OOM risk
│   └── 357+ frames: ❌ OOM
│
└── cuda:0,10gb;cuda:1,4gb;cpu,* (balanced)
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
├── Allocation: cuda:1,11gb;cuda:0,15gb;cpu,*
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
│   ├── wan2.2_i2v_high_noise_14B_Q6_K.gguf
│   ├── wan2.2_i2v_low_noise_14B_Q6_K.gguf
│   ├── wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors
│   ├── wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors
│   ├── Wan22-I2V_A14B-Lightning-H-Q6_K.gguf
│   ├── Wan22-I2V_A14B-Lightning-L-Q6_K.gguf
│   ├── smoothMixWan22GGUF_highQ6K.gguf
│   ├── smoothMixWan22GGUF_lowQ6K.gguf
│   ├── wan22EnhancedNSFW_V2_Q6K_HIGH.gguf
│   ├── wan22EnhancedNSFW_V2_Q6K_LOW.gguf
│   └── ...
│
├── diffusion_models/
│   ├── ltx-2-19b-distilled_Q4_K_M.gguf
│   └── ltx-2-19b-dev-Q4_K_M.gguf
│
├── text_encoders/
│   ├── umt5-xxl-enc-bf16.safetensors
│   ├── gemma-3-12b-it-qat-q4_0-unquantized/
│   ├── ltx-2-19b-embeddings_connector_bf16.safetensors
│   └── ...
│
├── vae/
│   ├── wan_2.1_vae.safetensors
│   ├── LTX2_video_vae_bf16.safetensors
│   ├── sdxl_vae.safetensors
│   └── ae.safetensors
│
├── checkpoints/
│   ├── flux1-dev-fp8.safetensors
│   ├── CyberRealistic_Pony_v14.1_FP16.safetensors
│   ├── dreamshaperXL_lightningDPMSDE.safetensors
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
    │   "LTX-2 with Audio Track"
    │   Status: 🔨 IN DEVELOPMENT
    │
    ├── 📂 Workflow: workflows/ltx2_audio_t2v_api.json
    │
    ├── 🧠 Diffusion Model
    │   └── ltx-2-19b-distilled_Q4_K_M.gguf
    │
    ├── 📝 Text Encoder (Gemma)
    │   └── gemma-3-12b-it-qat-q4_0-unquantized/
    │
    ├── 🎨 VAE
    │   └── LTX2_video_vae_bf16.safetensors
    │
    └── 🔊 Audio Model
        └── [TBD - MMAudio integration]
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

### 🎬 T2I→I2V Pipeline (Wan2.2 T2V Method)

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
    └── Wan2.2 animates the image

Status: ✅ PRODUCTION
Workflow: reapony_t2i_wan22_i2v_multigpu.json
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
│ 🎬 IMAGE-TO-VIDEO (I2V) - WAN2.2 14B GGUF                          │
│ ═══════════════════════════════════════════════════════════════════ │
│                                                                      │
│ [RECOMMENDED] 480×848 Portrait - MAX LENGTH                          │
│ ─────────────────────────────────────────────────────────────────── │
│   Frames: 321 (~20 sec) | VRAM: ~26GB | Time: ~23 min               │
│   Allocation: cuda:1,11gb;cuda:0,15gb;cpu,*                         │
│   Models:                                                            │
│     - wan2.2_i2v_high_noise_14B_Q6_K.gguf                           │
│     - wan2.2_i2v_low_noise_14B_Q6_K.gguf                            │
│   CLIP: umt5-xxl-enc-bf16.safetensors                               │
│   VAE: wan_2.1_vae.safetensors                                      │
│   Workflow: WAN22-I2V-DISTORCH2-LATEST-api.json                     │
│   Sampler: uni_pc | Steps: 6 | CFG: 1.0                             │
│   Tested: 2026-01-16 ✅                                              │
│                                                                      │
│ 480×848 Portrait - SAFE PRODUCTION                                   │
│ ─────────────────────────────────────────────────────────────────── │
│   Frames: 161 (~10 sec) | VRAM: ~22GB | Time: ~12 min               │
│   Allocation: cuda:1,11gb;cuda:0,15gb;cpu,*                         │
│   Same models as above                                               │
│   Tested: 2026-01-16 ✅                                              │
│                                                                      │
│ 576×1024 Standard Portrait                                           │
│ ─────────────────────────────────────────────────────────────────── │
│   Frames: 81 (~5 sec) | VRAM: ~24GB | Time: ~6 min                  │
│   Allocation: cuda:1,11gb;cuda:0,15gb;cpu,*                         │
│   Same models as above                                               │
│   Tested: 2026-01-12 ✅                                              │
│                                                                      │
│ 720×1280 HD Portrait - ❌ UNACCEPTABLE (under 5 sec)                 │
│ ─────────────────────────────────────────────────────────────────── │
│   Max possible: 41 frames (~2.5 sec) - TOO SHORT FOR PRODUCTION     │
│   DO NOT USE for video generation - use 480p or 576p instead        │
│                                                                      │
│ ═══════════════════════════════════════════════════════════════════ │
│ � IMAGE-TO-VIDEO (I2V) - WAN2.2 14B Q8_0 GGUF (DisTorch2)        │
│ ═══════════════════════════════════════════════════════════════════ │
│                                                                      │
│ 480×848 Portrait - Q8 EXPERIMENTAL                                   │
│ ─────────────────────────────────────────────────────────────────── │
│   Frames: 81 (~5 sec) | VRAM: ~26GB | Time: ~24 min                 │
│   Allocation: cuda:1,11gb;cuda:0,14.5gb;cpu,*                       │
│   Models:                                                            │
│     - wan2.2_i2v_high_noise_14B_Q8_0.gguf                           │
│     - wan2.2_i2v_low_noise_14B_Q8_0.gguf                            │
│   CLIP: umt5_xxl_fp8_e4m3fn_scaled.safetensors                      │
│   VAE: wan_2.1_vae.safetensors                                      │
│   Workflow: wan22_i2v_distorch2_q8_api.json                         │
│   Sampler: uni_pc | Steps: 8 | CFG: 1.0                             │
│   ⚠️ CLIP must use device=cuda:0 (5060 Ti), NOT cuda:1!             │
│   Previously OOM'd due to reversed CLIP device/donor (fixed 2026-03) │
│   Tested: 2026-03-01 ✅                                              │
│                                                                      │
│ ═══════════════════════════════════════════════════════════════════ │
│ �🎥 TEXT-TO-VIDEO (T2V) - LTX-2 19B                                  │
│ ═══════════════════════════════════════════════════════════════════ │
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
│ 🎥 TEXT-TO-VIDEO (T2V) - WAN2.2 (T2I→I2V Pipeline)                  │
│ ═══════════════════════════════════════════════════════════════════ │
│                                                                      │
│ 848×480 Landscape                                                    │
│ ─────────────────────────────────────────────────────────────────── │
│   Frames: 41 (~2.5 sec) | VRAM: ~20GB | Time: ~8 min                │
│   Pipeline: Flux T2I → Wan2.2 I2V                                   │
│   Tested: 2026-01-10 ✅                                              │
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
│ ⚡ DreamShaper Lightning - 1024×1024 (FASTEST)                       │
│ ─────────────────────────────────────────────────────────────────── │
│   Model: dreamshaperXL_lightningDPMSDE.safetensors [6.5GB]          │
│   VRAM: ~8GB | Time: 8-33s (depends on model warmup)                │
│   Steps: 8 | CFG: 2.0 | Sampler: dpmpp_sde + karras                 │
│   Best for: Quick iterations, previews                              │
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
│ Juggernaut Ragnarok - 1024×1024                                      │
│ ─────────────────────────────────────────────────────────────────── │
│   Model: juggernautXL_ragnarok.safetensors [7GB]                    │
│   VRAM: ~8GB | Time: 28s @ 25 steps                                 │
│   Best for: General/artistic                                        │
│   Tested: 2026-01-16 ✅                                              │
│                                                                      │
│ Nova Anime - 1024×1024                                               │
│ ─────────────────────────────────────────────────────────────────── │
│   Model: novaAnimeXL_ilV150.safetensors [6.5GB]                     │
│   VRAM: ~8GB | Time: 28s @ 25 steps                                 │
│   Best for: Anime style                                             │
│   Tested: 2026-01-16 ✅                                              │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ 🔬 EXPERIMENTAL / NEEDS MORE TESTING                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ I2V Lightning Models (faster but lower quality)                      │
│ ─────────────────────────────────────────────────────────────────── │
│   Models: Wan22-I2V_A14B-Lightning-H/L-Q6_K.gguf                    │
│   Status: 🔨 Untested with optimal allocation                        │
│                                                                      │
│ I2V Enhanced NSFW V2                                                 │
│ ─────────────────────────────────────────────────────────────────── │
│   Models: wan22EnhancedNSFW_V2_Q6K_HIGH/LOW.gguf                    │
│   Status: 🔨 Untested with optimal allocation                        │
│                                                                      │
│ LTX-2 with Audio                                                     │
│ ─────────────────────────────────────────────────────────────────── │
│   Workflow: wan22_i2v_ltx2_audio_api.json                           │
│   Status: 🔨 Audio sync issues                                       │
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
  "expert_mode_allocations": "cuda:1,11gb;cuda:0,15gb;cpu,*",
  "compute_device": "cuda:0",
  "donor_device": "cuda:1",
  "virtual_vram_gb": 16,
  "eject_models": true
}
```

### Quick Reference: What Works Right Now

> **⚠️ VIDEO: MINIMUM 81 FRAMES (5 sec) - Anything less is UNACCEPTABLE!**

#### 🎬 Video Generation

| Tool | Mode | Resolution | Frames | Duration | Time | Status |
|------|------|------------|--------|----------|------|--------|
| I2V | wan22 | 480×848 | 321 | **20 sec** | 23min | ✅ BEST |
| I2V | wan22 | 480×848 | 161 | **10 sec** | 12min | ✅ SAFE |
| I2V | wan22 | 480×848 | 81 | **5 sec** | 6min | ✅ MIN |
| I2V | wan22 | 576×1024 | 81 | **5 sec** | 6min | ✅ |
| T2V | ltx2 | 768×512 | 97 | **6 sec** | 8min | ✅ |

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

#### ⏱️ T2I Speed Tiers

```
⚡ LIGHTNING (5-10s)     → DreamShaper Lightning @ 8 steps
🏃 FAST (15-30s)         → SDXL models @ 25 steps
🐢 QUALITY (60-120s)     → Flux models @ 20 steps
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

**Last Updated**: 2026-01-16
**Maintainer**: @copilot (auto-update on generation complete - FUTURE)
