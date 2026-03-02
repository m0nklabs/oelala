### Added
- **I2I Face Processing Pipeline**: Enhanced Image-to-Image with IP-Adapter FaceID, FaceDetailer (YOLO + SAM), and GFPGAN face restore
  - 4 presets: Fast, Balanced, Face Preserve, Custom
  - Frontend UI with face ID, face detailer, and face restore toggles
  - `/i2i/presets` endpoint for preset discovery
- **ComfyUI Impact-Subpack**: Installed for UltralyticsDetectorProvider (face_yolov8m.pt)

### Changed
- **CreationsPickerModal**: Refactored from full-screen overlay to inline expandable panel
  - CSS slide-in animation, max-height 60vh scrollable content
  - Applied to all 7 tool files (I2I, V2V, I2T, Inpaint, FaceSwap, Reframe, V2T)
- **CORS Configuration**: Changed from `allow_origins=["*"]` to explicit origins list
  - Wildcard + credentials is invalid per CORS spec (browsers reject)
  - `/comfyui/output/` endpoint now sends `Vary: Origin` + explicit CORS headers
  - `apiFetch` adds `?_cors=1` cache-bust for Cloudflare stale cache invalidation
- **Image loading**: All 7 tools migrated from raw `fetch()` to `apiFetch()` for authenticated image loading from My Creations
