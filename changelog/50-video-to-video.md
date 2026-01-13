### Added

- **Video-to-Video (V2V) style transfer endpoint** (`/api/v2v/generate`)
  - Apply AI style transfer to existing videos while preserving motion
  - Three modes: `style_transfer`, `anime`, `enhance`
  - Configurable strength parameter (0.0-1.0)
  - Extracts first frame from input video
  - Uses Wan2.2 DisTorch2 I2V pipeline for generation
  - Full integration with credits system and media storage

- **V2V modes endpoint** (`/api/v2v-modes`)
  - Returns available V2V style transfer modes
  - Includes strength ranges and default values
  - Links to available I2V modes for advanced users

### Technical Details

- V2V uses existing DisTorch2 dual-GPU I2V workflow
- Frame extraction via OpenCV (cv2)
- Automatic resolution adjustment for video aspect ratios
- Full metadata tracking (original video info, generation params)
- Cleanup of temporary input files after processing

### Related

- Issue #50 - Video-to-Video style transfer
- Uses existing `generate_distorch2_video()` from comfyui_client
