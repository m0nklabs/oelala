# ComfyUI Server Inventory

Complete inventory of available resources on the self-hosted GPU runner.
**Last updated**: 2026-08-23

> **Compute backends** — which generation model families run on which server is now
> configurable via the **Compute Backend Inventory** (`src/backend/generation/compute_backends.py`
> + `compute_backends.json`), editable in the Admin panel → "Compute". Sections below document the
> known servers (ai-kvm2 default, Windows-PC, RunPod) that the inventory currently defines.

---

## 🎮 Hardware

| Component | Specs | CUDA Device |
|-----------|-------|-------------|
| GPU 1 | RTX 3060 12GB | `cuda:1` |
| GPU 2 | RTX 5060 Ti 16GB | `cuda:0` |
| **Total VRAM** | **28GB** | |
| RAM | 128GB DDR4 | |
| Storage | NVMe SSD | |

### DisTorch2 Allocation
```
cuda:1,11gb;cuda:0,15gb;cpu,2gb
```
Safe allocation with 1GB buffer per GPU + 2GB CPU fallback for edge cases.

---

## 📊 VRAM Limits & Guidelines

### DisTorch2 Multi-GPU Distribution
DisTorch2 automatically distributes model layers across both GPUs. Use these nodes:
- `UnetLoaderGGUFAdvancedDisTorch2MultiGPU`
- `VAELoaderDisTorch2MultiGPU`
- `CLIPLoaderDisTorch2MultiGPU`

**All loader nodes MUST include `expert_mode_allocations: "cuda:1,11gb;cuda:0,15gb;cpu,2gb"`**

### Video Generation (Wan 2.2 14B Q6_K)

| Resolution | Max Frames | Duration (16fps) | VRAM Usage |
|------------|------------|------------------|------------|
| 480p (848x480) | 81 | ~5 sec | ~24GB |
| 720p (1280x720) | 41 | ~2.5 sec | ~26GB |
| 1080p (1920x1080) | 17-25 | ~1-1.5 sec | ~27GB |

**Best Practice**: Start with 480p/41 frames for testing, scale up for production.

### Image Generation (SDXL/Pony)

| Task | Model | Max Resolution | VRAM Usage |
|------|-------|----------------|------------|
| T2I | SDXL Lightning | 1024x1024 | ~6GB |
| T2I | SDXL (full) | 1024x1024 | ~8GB |
| T2I | Flux FP8 | 1024x1024 | ~12GB |
| T2I | Krea 2 Turbo (INT8) | 1024x1024-2048x2048 | ~10-12GB |
| I2I | SDXL | 1536x1536 | ~10GB |
| Upscale | 4x (ESRGAN) / AI (SeedVR2) | 720p-2048p | ~10-16GB |

**Note**: Image gen fits on single GPU. Use `cuda:0` (16GB) for headroom.

### LTX-2 Video Generation (⚠️ Experimental)

> **Removed in cleanup** — de lokale LTX-2 19B-modellen zijn van de schijf
> verwijderd (LTX-2.3 draait nu **cloud-only** via RunPod). Verwijderde files:
> `ltx-2-19b-distilled-fp8.safetensors`, `ltx-2-19b-distilled_Q4_K_M.gguf`,
> `ltx-2-19b-dev-Q4_K_M.gguf`, `ltx-2-19b-embeddings_connector_bf16.safetensors`,
> `LTX2_video_vae_bf16.safetensors`, `ltx2_audio_vae.safetensors`.

### Audio Generation (MMAudio)

| Task | Model | VRAM Usage |
|------|-------|------------|
| Video-to-Audio | MMAudio Large 44k | ~8GB |
| Synchformer | Sync model | ~2GB |

**Note**: Run on `cuda:1` to keep `cuda:0` free for concurrent image/video.

### Text Encoding

| Encoder | VRAM Usage | Notes |
|---------|------------|-------|
| CLIP-L | ~1GB | SDXL text encoder |
| T5-XXL FP8 | ~8GB | Wan/Flux text encoder |
| UMT5-XXL | ~10GB | Wan 2.2 multilingual |

**Tip**: T5/UMT5 can be offloaded after encoding to free VRAM for generation.

### Concurrent Operations

With 28GB total VRAM, you can run:
- ✅ Image gen + Audio gen (different GPUs)
- ✅ Multiple image gens (batch on cuda:0)
- ⚠️ Video gen + anything else (tight, may OOM)
- ❌ Two video gens simultaneously

---

## 📦 Checkpoints (SDXL/Pony/Flux)

| Model | Category | Notes |
|-------|----------|-------|
| `CyberRealistic_Pony_v14.1_FP16.safetensors` | Realistic/Pony | High quality |
| `flux1-dev-fp8.safetensors` | Flux | FP8 quantized |
| `ponyDiffusionV6XL_v6StartWithThisOne.safetensors` | Pony | Base Pony model |
| `reapony_v90.safetensors` | Realistic/Pony | |

---

## 🖼️ Krea 2 (Flux2-familie, ComfyUI ≥ 0.27)

| Model | File | Size | Notes |
|-------|------|------|-------|
| Diffusion | `diffusion_models/krea2_turbo_int8_convrot.safetensors` | 13.5GB INT8 | Turbo = 8-step distilled |
| Text Encoder | `text_encoders/qwen3vl_4b_bf16.safetensors` | 8.9GB | CLIPLoader type `krea2` |
| VAE | `vae/qwen_image_vae.safetensors` | 254MB | Gedeeld met Qwen Image |

**Pipeline**: `UNETLoader(weight_dtype=default)` → `CLIPLoader(type=krea2)` → `KSampler(8 steps, CFG 1.0, euler/simple)` → `VAEDecode`
**Getest**: 2026-08-21 ✅ (1024×1024, ~80s eerste load)
**NSFW**: via CivitAI LoRA's (bijv. krea-2-nsfw-v2, 12 steps)
**License**: Krea 2 Community License (gratis < $1M omzet / 50 seats)

---

## 🌌 Flux 2 Dev (32B, multi-GPU, ComfyUI ≥ 0.31)

| Model | File | Size | Notes |
|-------|------|------|-------|
| Diffusion (GGUF) | `unet/flux2-dev-Q4_K_M.gguf` | 19.9GB Q4_K_M | unsloth Dynamic 2.0; 32B rectified-flow |
| Text Encoder | `text_encoders/mistral_3_small_flux2_fp8.safetensors` | 18GB FP8 | Mistral3-small, CLIPLoader type `flux2`, device=cpu |
| VAE | `vae/flux2-vae.safetensors` | 336MB | Flux 2 (Mage) one-step VAE |

**Pipeline**: `UnetLoaderGGUFDisTorch2MultiGPU(flux2 Q4, multi-GPU)` → `CLIPLoader(type=flux2, mistral3, cpu)` → `FluxGuidance` + `Flux2Scheduler` + `EmptyFlux2LatentImage` → `SamplerCustomAdvanced(euler/simple)` → `VAEDecode`

**Multi-GPU allocatie (DisTorch2, getest op RTX 3060 12GB + RTX 5060 Ti 16GB)**:
```
cuda:1,8gb;cuda:0,4gb;cpu,*
```
Compute op `cuda:1`, ~7.2GB model naar CPU-offload. De compute-kaart NIET >75% vullen — anders OOM op activations.

**Getest**: 2026-08-21 ✅ (1024×1024, ~297s; 768px ~215s). Eerste pogingen OOM'den op hogere allocatie.
**NSFW**: BFL heeft expliciete NSFW-mitigaties; community fine-tunes (bijv. AniEdit Flux.2 Klein) bestaan maar zijn schaarser dan Flux.1.
**License**: FLUX [dev] Non-Commercial License.

> **Opmerking**: `mistral_3_small_flux2_bf16` (35GB) en `flux2_dev_fp8mixed` (35GB) zijn ook op HF beschikbaar, maar we gebruiken alleen de fp8-Te + Q4-diffusie versies vanwege VRAM.

---

## 🎬 Video Models (Wan 2.2)

### UNET Models (GGUF - for DisTorch2)

| Model | Type | Quality |
|-------|------|---------|
| `wan2.2_i2v_high_noise_14B_Q6_K.gguf` | I2V High | Standard |
| `wan2.2_i2v_low_noise_14B_Q6_K.gguf` | I2V Low | Standard |
| `Wan22-I2V_A14B-Lightning-H-Q6_K.gguf` | I2V High | **Lightning (fast)** |
| `Wan22-I2V_A14B-Lightning-L-Q6_K.gguf` | I2V Low | **Lightning (fast)** |

> **Removed in cleanup**: `smoothMixWan22GGUF_high/lowQ6K.gguf`,
> `wan22EnhancedNSFWCameraPrompt_nsfwV2Q6KH/L.gguf` (ongebruikt/duplicaat).

> **Note**: Additional specialized UNET variants available on the server.

### UNET Models (FP8 - for T2V)

| Model | Type |
|-------|------|
| `Wan2_2-T2V-A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors` | T2V High |
| `Wan2_2-T2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors` | T2V Low |
| `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` | T2V High |
| `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` | T2V Low |

---

## 🖥️ Tweede ComfyUI-server (remote compute node — lokale MiniMax-H3)

Naast de ComfyUI op **ai-kvm2** (`localhost:8188`, systemd `comfyui`) kan er een **tweede
ComfyUI-server** zijn (bijv. de Windows-PC van de user) waarop lokale MiniMax-H3 draait. Die
wordt, net als elke andere `comfyui` backend, geconfigureerd via de compute backend inventory
(Admin panel → Compute) of de `COMPUTE_NODE_*` env-fallback — er is geen aparte server-client
meer.

| Eigenschap | Waarde |
|-----------|--------|
| Host | `base_url` in `compute_backends.json` / `COMPUTE_NODE_{n}_HOST` (fallback) |
| Poort (default) | `8188` |
| Installatie | ComfyUI portable (`C:\PROGRAMME\ComfyUI_windows_portable`) |
| Draaiend via | Taak `ComfyUIServer` bij inloggen (`start_comfy_server.bat`, log `comfy_server.log`) |
| Backend-client | `get_comfyui_client_for_backend()` (inventory) in `comfyui_client.py` |

Lokale MiniMax-H3 adapters (`minimax-h3-local-t2v` / `-i2v`) worden alleen geregistreerd
wanneer er een enabled `comfyui` backend is die `minimax_h3` declareert. De I2V-variant
uploadt z'n input-image zelf naar deze server (`handles_own_image_upload=True`).

### MiniMax-H3 local modellen (Windows PC)

| Component | File | Subdir |
|-----------|------|--------|
| Diffusion (int8 pruned) | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `diffusion_models/` |
| Text encoder (int8 convrot) | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | `text_encoders/` |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | `vae/` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | `vae/` |

Intro/uitgelegd in `README_MiniMax_H3_workflow.md`; downloadscripts `download_minimax_h3.cmd` / `.ps1`.
FL2VA genereert altijd een synchrone soundtrack (24 fps, 17k+5 frame grid, geen negative prompt / CFG).

---

## 🎨 LoRAs

### Wan 2.2 Video LoRAs

**90+ motion and action LoRAs available** in dual high/low noise variants.

All video LoRAs require BOTH versions loaded simultaneously:
- High noise → affects early diffusion steps
- Low noise → affects late diffusion steps

Available categories:
- Motion & animation effects
- Camera movements
- Action sequences
- Style transfers

> **Note**: Full LoRA list available on the server filesystem at `/home/flip/oelala/ComfyUI/models/loras/`

### SDXL/Pony LoRAs (SFW)

| Name | File | Purpose |
|------|------|---------|
| Add Details | `Add_Details_v1.2.safetensors` | Detail enhancement |
| IG Baddie | `igbaddie-PN.safetensors` | Instagram style |
| Tattoo Girls | `tattoogirls-PN.safetensors` | Tattoo style |
| Incase Style | `incase_style_v3-1_ponyxl_ilff.safetensors` | Artist style |
| Real Skin | `RealSkin_xxXL_v1.safetensors` | Skin texture |
| Shiny Skin | `ShinySkinSlider3.0_800steps.safetensors` | Slider |
| Body Weight Slider | `body_weight_slider_v1.safetensors` | Slider |
| Pony Realism | `Pony Realism Slider.safetensors` | Slider |
| Dramatic Lighting | `Dramatic Lighting Slider.safetensors` | Slider |
| Realism Yogi V2 | `Realism_Lora_By_Stable_Yogi_Pony_V2.safetensors` | Realism |

---

## 🔧 VAE Models

| Model | Purpose |
|-------|---------|
| `ae.safetensors` | General |
| `sdxl_vae.safetensors` | SDXL |
| `wan_2.1_vae.safetensors` | Wan 2.1/2.2 video |
| `Wan2.1_VAE.safetensors` | Wan 2.1/2.2 video |
| `qwen_image_vae.safetensors` | QwenVL / Krea 2 |
| `flux2-vae.safetensors` | Flux2 |

---

## 📝 CLIP/Text Encoders

| Model | Purpose |
|-------|---------|
| `clip_l.safetensors` | CLIP-L (SDXL) |
| `t5xxl_fp8_e4m3fn.safetensors` | T5-XXL FP8 (Wan/Flux) |
| `umt5-xxl-enc-bf16.safetensors` | UMT5-XXL (Wan 2.2) |
| `qwen3vl_4b_bf16.safetensors` | Qwen3-VL-4B (Krea 2, type krea2) |

---

## 🔊 Audio Models (MMAudio)

| Model | Purpose |
|-------|---------|
| `mmaudio_large_44k_v2_fp16.safetensors` | Main audio model |
| `mmaudio_vae_44k_fp16.safetensors` | Audio VAE |
| `mmaudio_synchformer_fp16.safetensors` | Sync model |
| `apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors` | CLIP for audio |

---

## 🧩 Custom Nodes

### Video Generation
- `ComfyUI-WanVideoWrapper` - Wan 2.2 workflow wrapper
- `ComfyUI-MultiGPU` - DisTorch2 multi-GPU distribution
- `ComfyUI-GGUF` - GGUF model loading
- `ComfyUI-gguf-vae` - GGUF VAE support
- `ComfyUI-Frame-Interpolation` - RIFE interpolation
- `ComfyUI-VFI` - Video frame interpolation
- `ComfyUI-GIMM-VFI` - GIMM-VFI
- `ComfyUI-VideoHelperSuite` - Video utilities
- `ComfyUI-FramePackWrapper` - Frame packing
- `ComfyUI-PainterI2V` - Painter I2V
- `ComfyUI-PainterLongVideo` - Long video generation
- `ComfyUI-LatentSyncWrapper` - LatentSync wrapper
- `ComfyUI-SeedVR2_VideoUpscaler` - Video upscaling

### Audio
- `ComfyUI-MMAudio` - MMAudio integration
- `ComfyUI-F5-TTS` - Text-to-speech
- `ComfyUI-SoundFlow` - Audio flow
- `TTS-Audio-Suite` - TTS utilities
- `ComfyUI-MelBandRoFormer` - Audio separation

### Image Processing
- `ComfyUI-Florence2` - Florence2 vision
- `ComfyUI-QwenVL` - QwenVL vision
- `ComfyUI_QwenImageEdit` - Qwen image editing
- `ComfyUI-JoyCaption` - Image captioning
- `ComfyUI-Image-Filters` - Image filters
- `ComfyUI-RMBG` - Background removal
- `ComfyUI-ColorCorrection` - Color correction
- `Comfyui-ColorMatchNodes` - Color matching
- `ComfyUI-WarperNodes` - Image warping

### Face Processing (Added 2026-03-02)
- `ComfyUI-Impact-Pack` - Detection, masking, FaceDetailer
- `ComfyUI-Impact-Subpack` - UltralyticsDetectorProvider (face_yolov8m.pt)
- **IP-Adapter FaceID Plus V2 SDXL** (`ComfyUI/models/ipadapter/ip-adapter-faceid-plusv2_sdxl.bin`)
- **CLIP Vision ViT-H** (`ComfyUI/models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`)
- **InsightFace buffalo_l** (`ComfyUI/models/insightface/models/buffalo_l/`)
- **GFPGAN v1.4** (`ComfyUI/models/facerestore_models/GFPGANv1.4.pth`)
- **SAM ViT-B** (`ComfyUI/models/sams/sam_vit_b_01ec64.pth`)
- **face_yolov8m.pt** (`ComfyUI/models/ultralytics/bbox/face_yolov8m.pt`)

### Utilities
- `ComfyUI-Inspire-Pack` - Inspiration tools
- `ComfyUI-KJNodes` - KJ utilities
- `ComfyUI_essentials` - Essential nodes
- `ComfyUI-Easy-Use` - Simplified nodes
- `ComfyUI-Crystools` - Crystal tools
- `ComfyUI-Custom-Scripts` - Custom scripts
- `was-node-suite-comfyui` - WAS node suite
- `rgthree-comfy` - rgthree nodes
- `efficiency-nodes-comfyui` - Efficiency nodes
- `ComfyUI_Comfyroll_CustomNodes` - Comfyroll
- `ComfyUI_tinyterraNodes` - TinyTerra
- `Derfuu_ComfyUI_ModdedNodes` - Derfuu nodes

### ControlNet & Preprocessing
- `comfyui_controlnet_aux` - ControlNet preprocessors
  - **OpenPose Preprocessor** - Pose detection (classic)
  - **DWPose Preprocessor** - Pose detection (recommended, better accuracy)
  - **Pose Keypoint Postprocess** - Keypoint processing
  - Canny, Depth, Lineart, etc.
- `ComfyUI-Addoor` - Addoor nodes

### Prompt & Text
- `comfyui-dynamicprompts` - Dynamic prompts
- `comfyui-portrait-master` - Portrait prompts
- `ComfyUI-Ollama-Describer` - Ollama integration
- `ComfyUI-Miaoshouai-Tagger` - Tagging

### Workflow Control
- `ComfyUI-AsyncPause` - Async pause
- `ComfyUI-pause` - Pause nodes
- `cg-use-everywhere` - Use everywhere
- `ComfyUI-TripleKSampler` - Triple KSampler
- `ComfyUI-SamplerSchedulerSelector` - Sampler selection
- `sigmas_tools_and_the_golden_scheduler` - Sigma tools
- `ComfyUI-WanMoeKSampler` - MoE KSampler
- `ComfyUI-TeaCache` - TeaCache acceleration

---

## 🚫 NOT Available

- `realvisxlV50_v50Bakedvae.safetensors` - **Removed**
- ControlNet models - **Not installed**

---

## 🔍 Upscale Models

### ESRGAN (Image/Video per-frame)
| Model | Size | Purpose |
|-------|------|------|
| `RealESRGAN_x4plus.pth` | 64MB | General purpose 4x upscale |
| `RealESRGAN_x4plus_anime_6B.pth` | 18MB | Anime/illustration optimized |
| `4x-UltraSharp.pth` | 64MB | High detail sharpening |
| `4x_foolhardy_Remacri.pth` | 64MB | Alternative 4x (good detail) |

Path: `ComfyUI/models/upscale_models/`

### SeedVR2 (AI Video Upscaler)
| Model | Size | Purpose |
|-------|------|------|
| `seedvr2_ema_3b_fp8_e4m3fn.safetensors` | 3.2GB | DiT 3B model (fp8 quantized) |
| `ema_vae_fp16.safetensors` | 479MB | VAE model (fp16) |

Path: `ComfyUI/models/SEEDVR2/`

**Tested Settings (2026-03-05):**
- DiT on cuda:1 (5060 Ti), VAE on cuda:0 (3060)
- BlockSwap: 28 blocks + swap_io to cuda:0
- Tiled VAE: 512x512 tiles, 64px overlap
- Max output: 720p (1280 max_resolution safety cap)
- Input 480x480 41 frames → 720x720 in ~100s

---

## 📁 Key Paths

```
/home/flip/oelala/ComfyUI/models/
├── checkpoints/       # SDXL/Pony/Flux checkpoints
├── unet/              # Wan 2.2 GGUF/FP8 models
├── loras/             # All LoRAs
├── vae/               # VAE models
├── clip/              # CLIP/T5 text encoders
├── mmaudio/           # MMAudio models
├── upscale_models/    # RealESRGAN_x4plus, anime_6B, UltraSharp, Remacri
├── SEEDVR2/           # SeedVR2 DiT 3B fp8 + VAE fp16
└── controlnet/        # Empty (not installed)

/home/flip/oelala/ComfyUI/custom_nodes/  # All custom nodes
/home/flip/venvs/gpu/                     # GPU Python environment
```

---

## 🔗 References

- [MULTI_GPU_SETUP.md](./MULTI_GPU_SETUP.md) - DisTorch2 configuration
- [copilot-instructions.md](../.github/copilot-instructions.md) - Agent instructions
- [gpu-tests.yml](../.github/workflows/gpu-tests.yml) - GPU test workflow
