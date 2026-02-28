### Added
- Real-ESRGAN model support in `/upscale-video` endpoint for AI-enhanced video upscaling
- Quality presets (fast/balanced/quality) for video upscaling with configurable CRF encoding
- Scale factor selector (2x/4x) in VideoUpscalerTool frontend
- Video Upscaler tool accessible from sidebar navigation
- "Upscale Video" option in My Media "Send to Tool" menu for quick access from media gallery
- GPU integration tests for quality_preset parameter validation

### Changed
- Updated `/upscale-video` endpoint to accept `quality_preset` parameter (fast, balanced, quality)
- Updated `build_video_upscale_workflow()` in comfyui_client.py to support quality presets
- Updated VideoUpscalerTool.jsx with model selector, scale factor, and quality preset controls
