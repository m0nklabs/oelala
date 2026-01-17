# Changelog

All notable changes to the Oelala AI Video Generation Platform are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.9.0] - 2026-01-17

### Added

#### AI Generation
- **I2V Analyze & Generate**: Vision AI pipeline using Moondream + Gemma2
  - 🔮 SFW and 🔥 NSFW buttons for AI-powered prompt generation
  - Image description shown in purple info box for reference
- **Generation Stats Tracking**: Duration and settings recorded for all generations
  - `GET /api/generation-stats` endpoint with summary stats
  - Stats stored in `/data/generation_stats.json`
- **I2V Prompt Strength Slider**: CFG control 1.0-5.0 with visual feedback
  - Labels: 🌊 Subtle / ⚡ Balanced / 🔥 Strong
- **Camera Position Selector**: 20+ presets for T2I shot composition
- **LTX-2 I2V Mode**: Alternative to Wan2.2 with faster inference
- **T2V Generation Modes**: Wan2.2 14B and LTX-2 19B support
- **Video-to-Video Style Transfer**: `/api/v2v/generate` endpoint
- **Unified Post-Processing**: Upscale, interpolate, concat in I2V/T2V tools

#### Admin & Management
- **Admin AI Settings Panel**: Configurable LLM prompts via UI
- **Admin System Monitoring**: GPU, queue, logs, disk status
- **Admin User Management**: User list, credit adjustment, tier management
- **User Suspension System**: Admin can suspend/unsuspend users

#### API & Integration
- **Public REST API v1**: Programmatic access at `/api/v1/*`
- **API Key Management**: Create, list, enable/disable, revoke keys
- **Webhook Delivery System**: HMAC-signed webhooks with retry logic
- **MediaService**: Unified oelala-storage + Supabase sync

#### User Features
- **User Profile Page**: Edit username, bio, social links
- **Storage Quota Display**: Progress bar with upgrade CTA
- **Gallery Lazy Loading**: Media loads on scroll
- **WebSocket Progress**: Real-time job tracking with ETA

### Fixed
- **I2V Edge Noise**: Added `ImageResize+` node for exact dimensions
- **Popup Backgrounds**: Fixed transparent dropdowns
- **Ollama VRAM Cleanup**: `keep_alive: "0"` unloads models immediately

### Changed
- Default CFG changed from 1.0 to 3.0 for better prompt adherence
- T2I model selector reorganized with grouped categories
- LLM temperature increased to 1.2 for more creative variety
- Code splitting: 25+ tools lazy loaded with React.lazy()

---

## [0.8.0] - 2026-01-10

### Added
- **DisTorch2 Multi-GPU Support**: Optimal dual-GPU allocation
  - `cuda:1,11gb;cuda:0,15gb;cpu,*` configuration
  - Wan2.2 14B Q6_K across RTX 5060 Ti + RTX 3060
- **Sequential Clip Generation**: Extend mode for longer videos
- **LoRA Stacking**: Multiple LoRAs with individual strengths
- **Auto-Upload to Storage**: Generated content saved to user bucket

### Fixed
- Video duration slider respects resolution-based VRAM limits
- ComfyUI output paths unified

---

## [0.7.0] - 2026-01-05

### Added
- **Supabase Database**: Complete migration system (001-007)
- **User Credits System**: Balance, transactions, tier management
- **Gallery System**: Publish, like, view generated content
- **Signed URLs**: Secure media access with expiration

### Changed
- Frontend migrated to React with Vite
- Backend consolidated to single FastAPI app

---

## [0.6.0] - 2025-12-28

### Added
- **Wan2.2 Image-to-Video**: First working I2V pipeline
- **Text-to-Image**: SDXL, Flux, SD1.5 support via ComfyUI
- **ComfyUI Integration**: Queue management, progress tracking

---

## [0.5.0] - 2025-12-20

### Added
- Initial project setup
- Basic FastAPI backend
- Simple frontend interface
- ComfyUI connection prototype

---

[0.9.0]: https://github.com/m0nklabs/oelala/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/m0nklabs/oelala/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/m0nklabs/oelala/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/m0nklabs/oelala/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/m0nklabs/oelala/releases/tag/v0.5.0
