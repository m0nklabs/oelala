### Added

- **InfoTooltip component** (`src/frontend/src/components/InfoTooltip.jsx`): Reusable hover tooltip with ? icon, auto-flip for viewport overflow, arrow pointer, and CSS transition animations.
- **Info tooltips across all 7 generation tools** — added contextual help "wolkjes" (cloud bubbles with ? icons) to ~70+ controls:
  - **ImageToVideoTool**: ~20 tooltips (Generation Mode, CFG, Resolution, Upscale, Aspect Ratio, Duration, FPS, Model Version, Sampling Steps, High Noise Steps, Model Shift, NAG Scale, Seed, Florence2, RIFE, Cloud Max settings)
  - **TextToVideoTool**: 12 tooltips (Generation Mode, Resolution, Upscale, Aspect Ratio, Duration, FPS, Extend Duration, Video Steps, Seed, T2I Steps, T2I CFG)
  - **TextToImageTool**: 11 tooltips (Batch Count, Flux/Wan2.2/SDXL Steps, Guidance, Seed, CFG Scale, Sampler, Scheduler, LoRAs)
  - **VideoToVideoTool**: 7 tooltips (Transform Strength, Output FPS, Max Frames, Steps, CFG Scale, Seed, Negative Prompt)
  - **FaceSwapTool**: 9 tooltips (Source Profile, Name, Reference Photos, Person Name, Training Steps, Training Jobs, Trained LoRAs)
  - **UpscaleTool**: 5 tooltips (Scale Factor, Model, Face Enhancement, Quality Preset)
  - **ImageToImageTool**: 13 tooltips (Prompt, Denoise Strength, Model, Face Identity, Face Detailer, Face Restore, Negative Prompt, Steps, CFG Scale, Sampler, Scheduler, Seed)

### Changed

- **ImageToImageTool**: Replaced native `title` attributes on face processing toggles with proper InfoTooltip components for consistent UX.
- **ImageToVideoTool**: Refactored inline upscale tooltip (30+ lines) and CFG `title` span to use InfoTooltip component.
- **TextToVideoTool**: Replaced inline resolution note text with InfoTooltip.
