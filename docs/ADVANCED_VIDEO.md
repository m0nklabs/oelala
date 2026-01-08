# Advanced Video Workflows

**Updated**: 2026-01-08
**Status**: ✅ Core Features Implemented

---

## Overview

Advanced video processing for Oelala: upscaling, frame interpolation, and I2V enhancements.

## ✅ Implemented Features

### 1. Image-to-Video (I2V)
- 16 camera motion presets (pan, tilt, zoom, dolly, orbit, crane, tracking, handheld)
- Duration: 3-15 seconds slider
- Full WAN 2.2 integration
- **Frontend**: `src/frontend/src/dashboard/tools/ImageToVideoTool.jsx`
- **Component**: `src/frontend/src/components/CameraMotionSelector.jsx`

### 2. Video Upscaling
- Resolution presets: 480p→720p, 480p→1080p, 720p→1080p, 720p→4K, 1080p→4K
- AI-enhanced: Real-ESRGAN Video
- Quality presets: Fast, Balanced, Quality
- **Frontend**: `src/frontend/src/dashboard/tools/VideoUpscalerTool.jsx`
- **Backend**: `/upscale-video` endpoint
- **Workflow**: `workflows/VideoUpscale/video_upscale_realesrgan.json`

### 3. Frame Interpolation
- FPS conversion: 15→30, 15→60, 24→30, 24→60, 30→60
- Slow motion: 2x, 4x, 8x
- Models: RIFE (recommended), FILM
- **Frontend**: `src/frontend/src/dashboard/tools/FrameInterpolationTool.jsx`
- **Backend**: `/interpolate-video` endpoint
- **Workflow**: `workflows/FrameInterpolation/rife_interpolation.json`

### 4. Video Extension (Template Only)
- Extend forwards/backwards
- Seamless loops
- **Workflow**: `workflows/VideoExtension/extend_video_wan22.json`

---

## Required ComfyUI Nodes

```bash
cd ComfyUI/custom_nodes

# Video Helper Suite (VHS)
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite

# Frame Interpolation
git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation
```

## Required Models

**Upscaling** (`ComfyUI/models/upscale_models/`):
- RealESRGAN_x4plus.pth
- RealESRGAN_x4plus_anime_6B.pth

**Interpolation** (`ComfyUI/models/frame_interpolation/`):
- rife47.pth

---

## API Examples

### Video Upscaling
```bash
curl -X POST http://localhost:7998/upscale-video \
  -F "file=@input.mp4" \
  -F "model=realesrgan-video" \
  -F "resolution_preset=720p → 1080p" \
  -F "quality_preset=balanced"
```

### Frame Interpolation
```bash
curl -X POST http://localhost:7998/interpolate-video \
  -F "file=@input.mp4" \
  -F "model=rife" \
  -F "target_fps=60"
```

---

## Performance

| Operation | VRAM | Speed |
|-----------|------|-------|
| Upscale 480p→720p | 8-12GB | ~5-10 sec/frame |
| Upscale 720p→1080p | 12-16GB | ~8-15 sec/frame |
| RIFE 2x | 4-8GB | ~2-5 sec/frame |

---

## File Structure

```
src/frontend/src/dashboard/tools/
├── VideoUpscalerTool.jsx       # Upscaling UI
├── FrameInterpolationTool.jsx  # Interpolation UI
└── ImageToVideoTool.jsx        # I2V with camera motion

workflows/
├── VideoUpscale/
├── FrameInterpolation/
└── VideoExtension/
```

See also: [ADVANCED_VIDEO_README](../workflows/ADVANCED_VIDEO_README.md)
