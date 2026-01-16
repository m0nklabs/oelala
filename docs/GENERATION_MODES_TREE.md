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
Tool Implementation Status
│
├── ✅ PRODUCTION (Fully tested, working)
│   ├── 🎬 ImageToVideo (I2V)
│   ├── 🎥 TextToVideo (T2V)
│   └── 🖼️ TextToImage (T2I)
│
├── 🔨 IN DEVELOPMENT (Partially working)
│   ├── 🎵 AudioGeneration (LTX-2 audio)
│   ├── 🔄 FrameInterpolation
│   ├── 📈 VideoUpscaler
│   └── 🖼️ ImageToImage (style transfer)
│
├── 📋 PLANNED (Workflow exists, not integrated)
│   ├── 📝 ImageToText (captioning)
│   ├── 🎭 FaceSwap
│   ├── 💋 LipSync
│   ├── 🔊 VoiceCloning
│   ├── 📺 VideoToVideo
│   └── 🗣️ SpeechToVideo
│
└── 💡 FUTURE (No workflow yet)
    ├── 🎓 LoRATraining
    ├── 🔲 Reframe
    └── 🧠 PromptGenerator
```

---

## 🎬 Image-to-Video (I2V)

```
I2V Generation Modes
│
├── 📦 standard (default)
│   │   "Wan2.2 14B Q6_K DisTorch2"
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

### I2V Alternative Models (Swappable)

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

```
T2I Model Categories
│
├── 📦 wan22
│   │   "Wan2.2 T2I Multi-GPU (video-ready images)"
│   │
│   └── wan2.2-t2i
│       ├── 🧠 Model: Built-in T2I workflow
│       └── 🎨 VAE: wan_2.1_vae.safetensors
│
├── 📦 flux
│   │   "Flux.1 (high quality artistic)"
│   │
│   └── flux1-dev-fp8
│       ├── 🧠 Model: flux1-dev-fp8.safetensors              [17GB]
│       ├── 📝 T5: (built-in)
│       └── 🎨 VAE: ae.safetensors                           [335MB]
│
├── 📦 sdxl
│   │   "SDXL Checkpoints (general purpose)"
│   │
│   ├── CyberRealistic_Pony_v14.1_FP16.safetensors          [6.5GB]
│   │   └── Category: Realistic/Pony
│   │
│   ├── dreamshaperXL_lightningDPMSDE.safetensors           [6.5GB]
│   │   └── Category: General
│   │
│   ├── illustriousRealismBy_v10VAE.safetensors             [6.5GB]
│   │   └── Category: Realistic
│   │
│   ├── juggernautXL_ragnarok.safetensors                   [7GB]
│   │   └── Category: General
│   │
│   ├── novaAnimeXL_ilV150.safetensors                      [6.5GB]
│   │   └── Category: Anime
│   │
│   ├── ponyDiffusionV6XL_v6StartWithThisOne.safetensors    [6.5GB]
│   │   └── Category: Pony
│   │
│   ├── reapony_v90.safetensors                             [6.5GB]
│   │   └── Category: Realistic/Pony
│   │
│   ├── ultraRealisticByStable_v20FP16.safetensors          [6.5GB]
│   │   └── Category: Realistic
│   │
│   └── waiIllustriousSDXL_v160.safetensors                 [6.5GB]
│       └── Category: Anime
│
├── 📦 sd15
│   │   "SD 1.5 (photorealistic)"
│   │
│   └── Realistic_Vision_V5.1.safetensors                   [4GB]
│
└── 📦 diffusers
    │   "Diffusers Models (experimental)"
    │
    └── sd3.5-large-int8
        └── (Requires diffusers pipeline)
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

## 🔄 Frame Interpolation (RIFE)

```
Frame Interpolation Modes
│
└── 📦 rife_v4 (default)
    │   "RIFE v4.6 Frame Interpolation"
    │   Status: 🔨 IN DEVELOPMENT
    │
    ├── 📂 Workflow: workflows/FrameInterpolation/
    │
    └── 🧠 Model
        └── rife_v4.6 (built into ComfyUI-Frame-Interpolation)
```

---

## 📈 Video Upscaler

```
Video Upscaler Modes
│
├── 📦 realesrgan (default)
│   │   "RealESRGAN 4x Video Upscaler"
│   │   Status: 🔨 IN DEVELOPMENT
│   │
│   ├── 📂 Workflow: workflows/VideoUpscale/
│   │
│   └── 🧠 Model
│       ├── RealESRGAN_x4plus.pth
│       └── RealESRGAN_x4plus_anime_6B.pth (anime variant)
│
└── 📦 topaz (external)
    │   "Topaz Video AI (if available)"
    │   Status: 💡 FUTURE
    │
    └── [External integration required]
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
│   │
│   └── 🧠 Uses T2I checkpoints with denoise control
│
└── 📦 controlnet
    │   "ControlNet Guided Generation"
    │   Status: 📋 PLANNED
    │
    └── ControlNet models (canny, depth, pose, etc.)
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

## 💋 LipSync

```
LipSync Modes
│
└── 📦 wav2lip (planned)
    │   "Wav2Lip Audio-Driven LipSync"
    │   Status: 📋 PLANNED
    │
    └── 🧠 Model
        └── wav2lip_gan.pth
```

---

## 🔊 Voice Cloning

```
Voice Cloning Modes
│
└── 📦 xtts (planned)
    │   "Coqui XTTS v2"
    │   Status: 📋 PLANNED
    │
    └── 🧠 Model
        └── XTTS-v2/
```

---

## 📺 Video-to-Video

```
Video-to-Video Modes
│
└── 📦 vid2vid_wan
    │   "Wan2.2 Video-to-Video (restyle)"
    │   Status: 📋 PLANNED
    │
    └── Uses I2V pipeline with frame extraction
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
│ 🎥 TEXT-TO-VIDEO (T2V) - LTX-2 19B                                  │
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
│ 🖼️ TEXT-TO-IMAGE (T2I) - FLUX                                       │
│ ═══════════════════════════════════════════════════════════════════ │
│                                                                      │
│ 1024×1024 Square                                                     │
│ ─────────────────────────────────────────────────────────────────── │
│   Model: flux1-dev-fp8.safetensors                                  │
│   VRAM: ~17GB | Time: ~30 sec                                       │
│   Steps: 20 | CFG: 3.5                                              │
│   Tested: 2026-01-08 ✅                                              │
│                                                                      │
│ ═══════════════════════════════════════════════════════════════════ │
│ 🖼️ TEXT-TO-IMAGE (T2I) - SDXL                                       │
│ ═══════════════════════════════════════════════════════════════════ │
│                                                                      │
│ 1024×1024 with CyberRealistic Pony                                   │
│ ─────────────────────────────────────────────────────────────────── │
│   Model: CyberRealistic_Pony_v14.1_FP16.safetensors                 │
│   VRAM: ~8GB | Time: ~15 sec                                        │
│   Steps: 25 | CFG: 7                                                │
│   Tested: 2026-01-05 ✅                                              │
│                                                                      │
│ 1024×1024 with DreamShaper XL                                        │
│ ─────────────────────────────────────────────────────────────────── │
│   Model: dreamshaperXL_lightningDPMSDE.safetensors                  │
│   VRAM: ~8GB | Time: ~8 sec (lightning)                             │
│   Steps: 8 | CFG: 2                                                 │
│   Tested: 2026-01-05 ✅                                              │
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

> **⚠️ MINIMUM 81 FRAMES (5 sec) - Anything less is UNACCEPTABLE!**

| Tool | Mode | Resolution | Frames | Duration | Time | Status |
|------|------|------------|--------|----------|------|--------|
| I2V | wan22 | 480×848 | 321 | **20 sec** | 23min | ✅ BEST |
| I2V | wan22 | 480×848 | 161 | **10 sec** | 12min | ✅ SAFE |
| I2V | wan22 | 480×848 | 81 | **5 sec** | 6min | ✅ MIN |
| I2V | wan22 | 576×1024 | 81 | **5 sec** | 6min | ✅ |
| T2V | ltx2 | 768×512 | 97 | **6 sec** | 8min | ✅ |
| T2I | flux | 1024×1024 | - | - | 30s | ✅ |
| T2I | sdxl | 1024×1024 | - | - | 15s | ✅ |

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
