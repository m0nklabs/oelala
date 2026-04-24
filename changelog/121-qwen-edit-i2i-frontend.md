### Added
- **Qwen Image Edit 2511 RunPod Worker**: Full cloud infrastructure for instruction-based image editing
  - `deploy/runpod-qwen/` — handler, Dockerfile, deploy script
  - RunPod serverless endpoint `oelala-i2i` (ID: `8djiexluyybooj`, template: `ed2614hd8k`)
  - Backend endpoint `/generate-qwen-edit` with width/height/steps/cfg/seed/lightning/lora_configs params
  - ComfyUI workflow builder `_build_qwen_edit_workflow()` based on official v2509 reference
- **I2I Edit Mode Resolution Controls**: Full resolution + aspect ratio picker in frontend
  - 4 resolution tiers: 768 (Fast), 1024 (Standard), 1280 (High), 1536 (Ultra)
  - 7 aspect ratios: 1:1, 16:9, 9:16, 4:3, 3:4, 2:3, 3:2
  - Dynamic pixel dimension calculation, clamped to multiples of 16
  - Settings persistence via `useToolSettings`

### Changed
- **"Use in tool" smart filtering**: Media type-aware tool selection in MyMedia
  - Each tool tagged with `accepts: ['image']` / `['video']` filter
  - Auto-sends directly when only 1 applicable tool for the media type (no popup)
  - Dropdown only shown when multiple tools apply; chevron arrow hidden for single option

### Fixed
- **Qwen Edit workflow validation**: Fixed 5 structural bugs in ComfyUI node graph
  - Two `TextEncodeQwenImageEditPlus` nodes (positive + negative with empty prompt)
  - `EmptySD3LatentImage` for 16-channel latent (was incorrectly using VAEEncode)
  - Correct input names (`prompt` not `positive`/`negative`)
- **Double output filter**: Handler now returns only the last image (edit result) when multiple outputs produced
