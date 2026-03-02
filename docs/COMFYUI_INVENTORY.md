# ComfyUI Server Inventory

Complete inventory of available resources on the self-hosted GPU runner.
**Last updated**: 2026-03-02

---

## 🎮 Hardware

| Component | Specs | CUDA Device |
|-----------|-------|-------------|
| GPU 1 | RTX 3060 12GB | `cuda:0` |
| GPU 2 | RTX 5060 Ti 16GB | `cuda:1` |
| **Total VRAM** | **28GB** | |
| RAM | 128GB DDR4 | |
| Storage | NVMe SSD | |

### DisTorch2 Allocation
```
cuda:0,11gb;cuda:1,15gb;cpu,2gb
```
Safe allocation with 1GB buffer per GPU + 2GB CPU fallback for edge cases.

---

## 📊 VRAM Limits & Guidelines

### DisTorch2 Multi-GPU Distribution
DisTorch2 automatically distributes model layers across both GPUs. Use these nodes:
- `UnetLoaderGGUFAdvancedDisTorch2MultiGPU`
- `VAELoaderDisTorch2MultiGPU`
- `CLIPLoaderDisTorch2MultiGPU`

**All loader nodes MUST include `expert_mode_allocations: "cuda:0,11gb;cuda:1,15gb;cpu,2gb"`**

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
| I2I | SDXL | 1536x1536 | ~10GB |
| Upscale | 4x | 2048x2048 max | ~8GB |

**Note**: Image gen fits on single GPU. Use `cuda:1` (16GB) for headroom.

### LTX-2 Video Generation (⚠️ Experimental)

| Model | Size | Text Encoder | Total VRAM |
|-------|------|--------------|------------|
| `ltx-2-19b-distilled-fp8.safetensors` | 26GB | UMT5 FP8 (5.6GB) | ~32GB |
| `ltx-2-19b-distilled-fp8.safetensors` | 26GB | Gemma 3 12B (23GB) | ~49GB ❌ |

**Current Status**: LTX-2 requires ~32GB VRAM minimum with UMT5, or ~49GB with Gemma 3 12B.
With 28GB total VRAM, LTX-2 is **not yet practical** on this setup.

**Blockers**:
1. Native ComfyUI loaders expect `spiece_model` tensor embedded in safetensors
2. HuggingFace Gemma 3 format uses separate `tokenizer.model` file
3. ComfyUI-LTXVideo custom nodes work but conflict with ComfyUI memory management when using `device_map="auto"`

**Future Options**:
- Wait for smaller quantized LTX-2 models (FP4)
- Use diffusers directly instead of ComfyUI nodes
- Use UMT5 with manual memory management
- Upgrade to 32GB+ GPU

### Audio Generation (MMAudio)

| Task | Model | VRAM Usage |
|------|-------|------------|
| Video-to-Audio | MMAudio Large 44k | ~8GB |
| Synchformer | Sync model | ~2GB |

**Note**: Run on `cuda:0` to keep `cuda:1` free for concurrent image/video.

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
- ✅ Multiple image gens (batch on cuda:1)
- ⚠️ Video gen + anything else (tight, may OOM)
- ❌ Two video gens simultaneously

---

## 📦 Checkpoints (SDXL/Pony/Flux)

| Model | Category | Notes |
|-------|----------|-------|
| `CyberRealistic_Pony_v14.1_FP16.safetensors` | Realistic/Pony | High quality |
| `dreamshaperXL_lightningDPMSDE.safetensors` | SDXL | **Fast** - recommended for T2I |
| `flux1-dev-fp8.safetensors` | Flux | FP8 quantized |
| `illustriousRealismBy_v10VAE.safetensors` | Realistic | Built-in VAE |
| `juggernautXL_ragnarok.safetensors` | SDXL | Popular |
| `novaAnimeXL_ilV150.safetensors` | Anime | SDXL |
| `ponyDiffusionV6XL_v6StartWithThisOne.safetensors` | Pony | Base Pony model |
| `Realistic_Vision_V5.1.safetensors` | SD1.5 | Legacy support |
| `reapony_v90.safetensors` | Realistic/Pony | |
| `ultraRealisticByStable_v20FP16.safetensors` | Realistic | FP16 |
| `waiIllustriousSDXL_v160.safetensors` | Anime | Illustrious-based |

---

## 🎬 Video Models (Wan 2.2)

### UNET Models (GGUF - for DisTorch2)

| Model | Type | Quality |
|-------|------|---------|
| `wan2.2_i2v_high_noise_14B_Q6_K.gguf` | I2V High | Standard |
| `wan2.2_i2v_low_noise_14B_Q6_K.gguf` | I2V Low | Standard |
| `Wan22-I2V_A14B-Lightning-H-Q6_K.gguf` | I2V High | **Lightning (fast)** |
| `Wan22-I2V_A14B-Lightning-L-Q6_K.gguf` | I2V Low | **Lightning (fast)** |
| `smoothMixWan22GGUF_highQ6K.gguf` | I2V High | Smooth motion |
| `smoothMixWan22GGUF_lowQ6K.gguf` | I2V Low | Smooth motion |

> **Note**: Additional specialized UNET variants available on the server.

### UNET Models (FP8 - for T2V)

| Model | Type |
|-------|------|
| `Wan2_2-T2V-A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors` | T2V High |
| `Wan2_2-T2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors` | T2V Low |
| `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` | T2V High |
| `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` | T2V Low |

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
| `qwen_image_vae.safetensors` | QwenVL |

---

## 📝 CLIP/Text Encoders

| Model | Purpose |
|-------|---------|
| `clip_l.safetensors` | CLIP-L (SDXL) |
| `t5xxl_fp8_e4m3fn.safetensors` | T5-XXL FP8 (Wan/Flux) |
| `umt5-xxl-enc-bf16.safetensors` | UMT5-XXL (Wan 2.2) |

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
- ESRGAN upscale models - **Not installed** (empty folder)
- ControlNet models - **Not installed**

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
├── upscale_models/    # Empty (not installed)
└── controlnet/        # Empty (not installed)

/home/flip/oelala/ComfyUI/custom_nodes/  # All custom nodes
/home/flip/venvs/gpu/                     # GPU Python environment
```

---

## 🔗 References

- [MULTI_GPU_SETUP.md](./MULTI_GPU_SETUP.md) - DisTorch2 configuration
- [copilot-instructions.md](../.github/copilot-instructions.md) - Agent instructions
- [gpu-tests.yml](../.github/workflows/gpu-tests.yml) - GPU test workflow
