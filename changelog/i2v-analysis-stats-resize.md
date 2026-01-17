### Added

- **I2V Analyze & Generate**: New vision AI pipeline using Moondream + Gemma2
  - 🔮 SFW button: Analyze image and generate creative video prompts
  - 🔥 NSFW button: Generate adult/sensual video prompts (when enabled)
  - Vision model describes image, LLM creates compelling video scenarios
  - Image description shown in purple box for reference

- **Generation Stats Tracking**: Track duration and settings for all generations
  - Records: resolution, frames, fps, steps, cfg, model, duration, success/fail
  - New endpoint: `GET /api/generation-stats?limit=100&job_type=wan22`
  - Returns summary stats (avg/min/max duration, success rate) + records
  - Stats stored in `/home/flip/oelala/data/generation_stats.json`

- **I2V Prompt Strength Slider**: CFG control with visual feedback
  - Range 1.0-5.0 with labels: 🌊 Subtle / ⚡ Balanced / 🔥 Strong
  - Default changed from 1.0 to 3.0 for better prompt adherence

- **Admin AI Settings Panel**: Configurable LLM prompts via admin UI
  - Edit system prompt for prompt enhancement
  - Select Ollama model from available models
  - Reset to defaults button

- **Camera Position Selector**: New T2I component for shot composition
  - 20+ presets: Eye level, Low/High angle, Close-up, Wide shot, POV, etc.
  - Auto-prefixes prompt with camera/angle descriptors

### Fixed

- **I2V Edge Noise**: Added `ImageResize+` node to workflow
  - Input images now resized/cropped to exact target dimensions
  - Method: "fill / crop" with lanczos interpolation
  - Eliminates padding artifacts at video edges

- **Popup Backgrounds**: Fixed transparent dropdowns
  - QueueIndicator: `--bg-secondary` → `--bg-panel`
  - UserMenu: `--bg-card` → `--bg-panel`

- **Ollama VRAM Cleanup**: Added `keep_alive: "0"` to all API calls
  - Models unload immediately after use
  - Frees VRAM before video generation starts

### Changed

- T2I model selector reorganized with grouped categories
- Prompt enhance button can now generate random prompts when empty
- LLM temperature increased to 1.2 for more creative variety
