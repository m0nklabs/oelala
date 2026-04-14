### Added
- Qwen-Image-Edit-2511 instruction-based image editing via RunPod cloud
- AI Edit mode in Image to Image tool (mode selector: Transform vs AI Edit)
- Lightning LoRA support for 4-step fast generation in edit mode
- LoRA Stack panel for edit mode with per-slot strength controls
- `/generate-qwen-edit` backend endpoint with full workflow builder
- `/loras` endpoint integration for available LoRAs listing
- Example instruction chips for quick edit suggestions
- Cloud-powered info banner for edit mode

### Changed
- Image to Image tool now has dual-mode UI: Transform (local SDXL) and AI Edit (cloud Qwen)
- Unified settings persistence under single `image_to_image` key for both modes
- I2I generate handler branches by mode to appropriate backend endpoint
- Job metadata registration added to `/generate-i2i` for storage upload tracking

### Removed
- Standalone ImageEditTool (merged into ImageToImageTool as AI Edit mode)
- `IMAGE_EDIT` tool ID and navigation entry
