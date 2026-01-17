### Added

- **Unified Post-Processing System**: Complete overhaul of post-processing tools
  - Inline post-processing options added to Image-to-Video tool (upscale, frame interpolation, audio)
  - Inline post-processing options added to Text-to-Video tool (upscale, frame interpolation)
  - Chained job execution: post-processing runs automatically after generation completes

- **Standalone Post-Processing Tool** (`Advanced → Post-Processing`)
  - Process existing/uploaded media without regeneration
  - Three modes:
    - **Upscale**: Real-ESRGAN video upscaling (2x or 4x)
    - **Interpolate**: RIFE frame interpolation (30/48/60 fps)
    - **Concat**: Join multiple videos into one
  - MediaPicker integration for selecting existing gallery items
  - Multi-file upload support for concatenation

- **New Backend Endpoint** (`/post-process`)
  - Standalone post-processing for existing media
  - Accepts uploaded files or references to existing media
  - Supports upscale, interpolate, and concat modes
  - Credit-based pricing (5/3/2 credits respectively)

### Changed

- **Navigation restructure**: Removed separate "Video Upscaler" and "Frame Interpolation" tools
- Post-processing now unified under single tool in Advanced section
- I2V/T2V tools now show post-processing options as collapsible section

### Technical

- Added `pending_post_processing` dict for tracking chained jobs
- Added `trigger_post_processing_chain()` function for sequential job execution
- ComfyUI workflow builders: `build_video_upscale_workflow()`, `build_rife_workflow()`, `build_video_concat_workflow()`
- Lazy loading for PostProcessingTool component
