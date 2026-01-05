# Advanced Video Workflows

This directory contains ComfyUI workflow templates for advanced video processing capabilities.

## 🖥️ GPU Configuration (oelala-gpu)

**Hardware:**
- RTX 5060 Ti 16GB (cuda:1)
- RTX 3060 12GB (cuda:0)
- Total: 28GB VRAM

**DisTorch2 Allocation:**
```
expert_mode_allocations: "cuda:0,11gb;cuda:1,15gb;cpu,2gb"
```

This allocation distributes model weights across both GPUs with a small CPU fallback for overflow.
- 11GB on RTX 3060 (cuda:0)
- 15GB on RTX 5060 Ti (cuda:1)
- 2GB CPU fallback
- Total: 26GB allocated out of 28GB available (leaves 2GB headroom for VRAM overhead)

**Performance Benefits:**
- 2x faster than single GPU
- Enables longer videos (up to 241 frames @ 720×400)
- Better VRAM utilization
- SageAttention reduces VRAM by 15-20%

**See Also:**
- `docs/COMFYUI_INVENTORY.md` - Complete model & VRAM inventory
- `docs/HARDWARE_LIMITS.md` - Tested resolution/frame limits
- `docs/MULTI_GPU_SETUP.md` - DisTorch2 setup guide

## 📁 Directory Structure

```
workflows/
├── VideoUpscale/           # Video upscaling workflows
├── FrameInterpolation/     # Frame interpolation workflows
├── VideoExtension/         # Video extension workflows
├── ImageToVideo/           # Image-to-video workflows (existing)
└── VideoToVideo/           # Video-to-video workflows (existing)
```

## 🎬 Video Upscaling

### video_upscale_realesrgan.json
AI-enhanced video upscaling using Real-ESRGAN.

**Features:**
- Resolution presets: 480p → 720p → 1080p → 4K
- Quality vs speed presets (fast, balanced, quality)
- Batch processing for long videos
- AI-enhanced upscaling for better quality than traditional methods

**Required Nodes:**
- VHS (Video Helper Suite) - For video loading/saving
- ComfyUI-Upscale - Real-ESRGAN models

**Usage:**
```json
{
  "model": "RealESRGAN_x4plus.pth",
  "input": "input_video.mp4",
  "output": "upscaled_video.mp4"
}
```

## ⚡ Frame Interpolation

### rife_interpolation.json
Smooth frame interpolation using RIFE (Real-Time Intermediate Flow Estimation).

**Features:**
- FPS conversion: 15fps → 30fps → 60fps
- Slow motion creation (2x, 4x, 8x slower)
- Optical flow visualization (planned, not yet implemented)
- Fast & high quality

**Required Nodes:**
- VHS (Video Helper Suite)
- ComfyUI-Frame-Interpolation (RIFE/FILM)

**Usage:**
```json
{
  "model": "rife47.pth",
  "multiplier": 2,
  "target_fps": 60,
  "fast_mode": true
}
```

## 🎞️ Video Extension

### extend_video_wan22.json
Extend videos forwards or backwards using WAN 2.2 video generation with DisTorch2 multi-GPU support.

**Features:**
- Extend video forwards (continuation)
- Extend video backwards (prequel)
- Seamless loop creation
- Scene continuation with AI
- **DisTorch2 multi-GPU distribution** for faster processing

**Required Nodes:**
- VHS (Video Helper Suite)
- ComfyUI-WanVideoWrapper (WAN 2.2 models)
- ComfyUI-MultiGPU (DisTorch2 support)
- Video concatenation nodes

**DisTorch2 Configuration:**
```json
{
  "expert_mode_allocations": "cuda:0,11gb;cuda:1,15gb;cpu,2gb",
  "compute_device": "cuda:0",
  "donor_device": "cuda:1",
  "virtual_vram_gb": 16,
  "eject_models": true
}
```

**Nodes Used:**
- `UnetLoaderGGUFAdvancedDisTorch2MultiGPU` - Load WAN 2.2 GGUF model with GPU distribution
- `VAELoaderDisTorch2MultiGPU` - Load VAE with GPU distribution
- `CLIPLoaderDisTorch2MultiGPU` - Load T5 encoder with GPU distribution
- `PathchSageAttentionKJ` - Reduce VRAM usage by 15-20%

**Usage:**
```json
{
  "direction": "forward",
  "frames": 81,
  "prompt": "smooth camera motion, continuation of the scene"
}
```

## 🔧 Installation

### Required ComfyUI Custom Nodes

1. **Video Helper Suite (VHS)**
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
   ```

2. **Frame Interpolation (RIFE/FILM)**
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation
   ```

3. **Upscale Models**
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/city96/ComfyUI-GGUF
   ```

4. **WAN Video (already installed)**
   - ComfyUI-WanVideoWrapper

### Required Models

Download these models to `ComfyUI/models/`:

**Upscaling Models** (`upscale_models/`):
- RealESRGAN_x4plus.pth
- RealESRGAN_x4plus_anime_6B.pth
- 4x-UltraSharp.pth

**Interpolation Models** (`frame_interpolation/`):
- rife47.pth (RIFE)
- film_net_fp32.pt (FILM)

**Video Models** (`unet/` - already installed):
- wan2.2_i2v_high_noise_14B_Q6_K.gguf
- wan2.2_i2v_low_noise_14B_Q6_K.gguf

## 📊 Technical Notes

### Video Upscaling
- **Memory**: ~8-12GB VRAM for 1080p upscaling
- **Speed**: ~5-10 seconds per frame (depends on resolution)
- **Batch Size**: Higher = faster but more VRAM

### Frame Interpolation
- **Memory**: ~4-8GB VRAM
- **Speed**: RIFE is faster than FILM (~2-5 seconds per frame)
- **Quality**: RIFE 4.7 recommended for best quality/speed balance

### Video Extension
- **Memory**: Same as I2V generation (~12-24GB for 81 frames)
- **Speed**: ~2-5 minutes per extension clip
- **Continuity**: Works best with static camera or smooth motion

## 🎯 Frontend Integration

These workflows are integrated into the Oelala frontend via:

1. **VideoUpscalerTool.jsx** - `/upscale-video` endpoint
2. **FrameInterpolationTool.jsx** - `/interpolate-video` endpoint
3. **VideoToVideoTool.jsx** (enhanced) - `/extend-video` endpoint

## 🐛 Troubleshooting

### Video loading fails
- Ensure VHS is installed: `ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite`
- Check video format (MP4 H.264 recommended)

### Upscaling crashes
- Reduce batch size (lower VRAM usage)
- Use smaller resolution preset
- Check if upscale model is downloaded

### Interpolation artifacts
- Try different multiplier (2x is most stable)
- Reduce target FPS
- Use RIFE instead of FILM for action scenes

## 📝 License

These workflow templates are part of the Oelala project.
Models may have different licenses - check individual model documentation.
