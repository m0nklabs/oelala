# 🔥 OELALA - Advanced ComfyUI NSFW Setup

A curated ComfyUI configuration for NSFW image and video generation, featuring:
- **WAN 2.2 14B** video generation (I2V, T2V) with multi-GPU support
- **Flux NSFW** models (FluxedUp, Persephone, Z-Image Turbo)
- **SmolLM2** AI-powered prompt generation
- **JoyCaption** uncensored image captioning
- **QwenVL** video captioning with NSFW support

## 🚀 Quick Start (Windows)

```batch
git clone https://github.com/m0nklabs/oelala.git
cd oelala
scripts\install_windows.bat
```

Then download models (selectively):
```batch
scripts\download_base_models.bat      # Required (~30GB)
scripts\download_nsfw_models.bat      # NSFW checkpoints (~65GB)
scripts\download_wan22_models.bat     # Video generation (~72GB)
scripts\download_llm_models.bat       # AI captioning (~16GB)
scripts\download_zimage_model.bat     # Fast generation (~12GB)
```

## 📁 Repository Structure

```
oelala/
├── ComfyUI/
│   ├── custom_nodes/          # Pre-configured custom nodes
│   ├── user/default/workflows/  # Ready-to-use workflows
│   └── models/                # Model directories (empty, use download scripts)
├── scripts/
│   ├── install_windows.bat    # Windows installation
│   ├── download_*.bat         # Model download scripts
│   └── install_linux.sh       # Linux installation
└── docs/
    ├── MODELS.md              # Model documentation
    └── WORKFLOWS.md           # Workflow documentation
```

## 🎨 Included Workflows

### Image Generation (T2I)
| Workflow | Model | Description |
|----------|-------|-------------|
| `flux_dev_t2i_smollm2.json` | Flux Dev | High quality images with AI prompts |
| `fluxed_up_nsfw_t2i_smollm2.json` | FluxedUp | NSFW image generation |
| `z_image_turbo_t2i_smollm2.json` | Z-Image Turbo | Fast 1-step generation |
| `persephone_nsfw_t2i_smollm2.json` | Persephone | SFW/NSFW toggle |

### Video Generation (I2V/T2V)
| Workflow | Model | Description |
|----------|-------|-------------|
| `wan22_EnhancedNSFW_Q6K_HIGH_only_i2v_smollm2.json` | WAN 2.2 NSFW | Image to NSFW video |
| `wan22_i2v_14b_Q6K_multigpu_smollm2.json` | WAN 2.2 14B | Multi-GPU video generation |
| `wan22_t2i_gguf_multigpu_smollm2.json` | WAN 2.2 GGUF | Text to video |

### Captioning
| Workflow | Model | Description |
|----------|-------|-------------|
| `joycaption_i2t_nsfw.json` | JoyCaption | Uncensored image captioning |
| `joycaption_batch_i2t.json` | JoyCaption | Batch dataset captioning |

### Image-to-Image
| Workflow | Model | Description |
|----------|-------|-------------|
| `i2i_CyberRealistic_Pony_clothing_removal_smollm2.json` | CyberRealistic | Clothing removal |

## 🔧 Custom Nodes

All required custom nodes are automatically installed:

| Node | Purpose |
|------|---------|
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | GGUF model loading |
| [ComfyUI-MultiGPU](https://github.com/pollockj/ComfyUI-MultiGPU) | Multi-GPU distribution |
| [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) | WAN 2.2 video |
| [ComfyUI_LayerStyle_Advance](https://github.com/zombieyang/ComfyUI_LayerStyle_Advance) | SmolLM2 prompts |
| [ComfyUI-JoyCaption](https://github.com/1038lab/ComfyUI-JoyCaption) | Image captioning |
| [ComfyUI-QwenVL](https://github.com/1038lab/ComfyUI-QwenVL) | Video captioning |
| [comfyui-dynamicprompts](https://github.com/adieyal/comfyui-dynamicprompts) | Dynamic prompts |
| [comfyui-portrait-master](https://github.com/florestefano1975/comfyui-portrait-master) | Portrait generation |

## 💾 Model Downloads

Total model size: **~195GB** (download what you need)

| Category | Size | Script | Models |
|----------|------|--------|--------|
| Base | ~30GB | `download_base_models.bat` | Flux Dev, CLIP, VAE, Text Encoders |
| NSFW | ~65GB | `download_nsfw_models.bat` | CyberRealistic Pony, FluxedUp, Persephone, WAN NSFW |
| WAN 2.2 | ~72GB | `download_wan22_models.bat` | WAN 2.2 14B I2V/T2V, Seko LoRAs |
| LLM | ~16GB | `download_llm_models.bat` | SmolLM2, JoyCaption, QwenVL |
| Z-Image | ~12GB | `download_zimage_model.bat` | Z-Image Turbo BF16 |

## 🖥️ System Requirements

### Minimum (Basic workflows)
- **GPU**: NVIDIA RTX 3060 12GB
- **RAM**: 32GB
- **Storage**: 100GB SSD

### Recommended (WAN 2.2 14B)
- **GPU**: 2x NVIDIA RTX 4090 24GB
- **RAM**: 64GB
- **Storage**: 500GB NVMe SSD

### Linux Server Setup
For best performance, run ComfyUI on a Linux server and access via browser:
```
http://YOUR_SERVER_IP:8188
```

## 📝 License

This repository contains only configuration and scripts. Model licenses vary:
- Flux models: FLUX.1-dev license
- WAN models: Apache 2.0
- NSFW models: Various (check individual model pages)

## ⚠️ Disclaimer

This project is for **adult content generation**. Users are responsible for:
- Compliance with local laws
- Ethical use of AI generation
- Respecting model licenses
- Not generating illegal content

---

Created with ❤️ by m0nk111
