# Changelog

All notable changes to the Oelala AI Video Generation Platform are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.12.0] - 2026-04-15

### Changed
- **Storage Migration: oelala-storage → MinIO** (PR #128)
  - Replaced custom Go-based oelala-storage service with MinIO S3-compatible object storage
  - `storage_client.py` now uses `minio` Python SDK directly (no REST wrapper)
  - Media, user files, and ComfyUI output served via MinIO with streaming + range requests
  - Bucket mapping: `oelala-generated`, `oelala-comfyui`, `oelala-avatars`, `oelala-users`
  - Cloudflare tunnel updated: `storage.oelala.xyz` → MinIO S3 API, `storage-main.oelala.xyz` → MinIO Console
  - 1.1 GiB / 1,059 objects migrated from oelala-storage to MinIO
  - Old oelala-storage + node-01 systemd services stopped and disabled

### Added
- **Dual-node CDN distribution**: Public media (generated, comfyui) served via round-robin across `storage.oelala.xyz` (node 1) + `storage2.oelala.xyz` (node 2)
  - `StorageClient.public_url()` generates direct CDN URLs with round-robin distribution
  - `STORAGE_PUBLIC_NODES` env var configures public CDN endpoints
  - Private buckets (users, avatars) still use presigned URLs via backend proxy
- **Backblaze B2 offsite backup**: All 4 MinIO buckets mirrored to B2 `oelala-media-eu` via `mc mirror`
  - Backup script `scripts/minio-backup-mirror.sh` syncs to both node 2 and B2 every 15 min

### Removed
- **Backblaze B2 as primary storage**: B2 env vars removed from `.env` (B2 now backup-only)
- **oelala-storage dependency**: Go storage service no longer required; replaced by MinIO
- **`storage-node-01.oelala.xyz`** tunnel route (defunct)

---

## [0.11.0] - 2026-04-14

### Added
- **Qwen Image Edit 2511 RunPod Worker**: Full cloud infrastructure for instruction-based image editing
  - `deploy/runpod-qwen/` — handler, Dockerfile, deploy script
  - RunPod serverless endpoint `oelala-i2i` (ID: `8djiexluyybooj`, template: `ed2614hd8k`)
  - Backend endpoint `/generate-qwen-edit` with width/height/steps/cfg/seed/lightning/lora_configs params
  - ComfyUI workflow builder `_build_qwen_edit_workflow()` based on official v2509 reference
- **I2I Edit Mode Resolution Controls**: Full resolution + aspect ratio picker
  - 4 resolution tiers: 768 (Fast), 1024 (Standard), 1280 (High), 1536 (Ultra)
  - 7 aspect ratios with dynamic pixel dimension calculation (clamped to multiples of 16)
  - Settings persistence via `useToolSettings`

### Changed
- **"Use in tool" smart filtering**: Media type-aware tool selection in MyMedia
  - Each tool tagged with `accepts: ['image']` / `['video']` filter
  - Auto-sends directly when only 1 applicable tool (no popup needed)
  - Dropdown only shown when multiple tools apply

### Fixed
- **Qwen Edit workflow validation**: Fixed 5 structural bugs in ComfyUI node graph
- **Double output filter**: Handler returns only the edit result, not reference reconstruction

---

## [0.10.0] - 2026-03-02

### Added
- **I2I Face Processing Pipeline**: Enhanced Image-to-Image with IP-Adapter FaceID, FaceDetailer (YOLO + SAM), and GFPGAN face restore
  - 4 presets: Fast, Balanced, Face Preserve, Custom
  - `/i2i/presets` endpoint for preset discovery
  - Dynamic ComfyUI workflow builder (`_build_i2i_workflow()`)
- **ComfyUI Impact-Subpack**: Installed for UltralyticsDetectorProvider
  - face_yolov8m.pt YOLO model for face detection

### Changed
- **CreationsPickerModal**: Refactored from full-screen overlay to inline expandable panel
  - CSS slide-in animation, compact header with X close
  - Applied to all 7 tool files (I2I, V2V, I2T, Inpaint, FaceSwap, Reframe, V2T)
- **CORS Configuration**: Fixed spec violation (`allow_origins=["*"]` + credentials)
  - Explicit origins list for credentialed requests
  - `/comfyui/output/` endpoint adds `Vary: Origin` for Cloudflare CDN cache correctness
  - `apiFetch` adds `?_cors=1` cache-bust for stale CF entries
- **Image loading**: All 7 tools migrated from raw `fetch()` to `apiFetch()` for authenticated My Creations loading

### Fixed
- **CORS on media files**: Browser rejected `Access-Control-Allow-Origin: *` with `Authorization` header
- **Cloudflare cache**: Old responses without `Vary: Origin` caused CORS failures for new requests

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

[0.10.0]: https://github.com/m0nklabs/oelala/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/m0nklabs/oelala/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/m0nklabs/oelala/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/m0nklabs/oelala/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/m0nklabs/oelala/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/m0nklabs/oelala/releases/tag/v0.5.0
Updated models system to dynamic ComfyUI API
