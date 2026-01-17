### Added

- **LTX-2 Image-to-Video mode**: Added LTX-2 19B as alternative I2V generation model
  - Single model architecture (no high/low noise dual-pass)
  - Uses Gemma 3 text encoder instead of UMT5
  - Faster inference for shorter clips
  - Available via dropdown in Image-to-Video tool

- **Dynamic duration limits**: Duration slider now respects resolution-based VRAM limits
  - 480p: max 20s (Wan2.2) / 12s (LTX-2)
  - 576p: max 7s (Wan2.2) / 8s (LTX-2)
  - 720p: max 4s (Wan2.2) / 5s (LTX-2)

- **New backend endpoint**: `/generate-ltx2-i2v-async` for LTX-2 I2V generation

- **New workflow**: `workflows/ImageToVideo/ltx2_i2v_api.json` for LTX-2 I2V

### Changed

- Updated `GENERATION_MODES_TREE.md` with LTX-2 I2V documentation and max duration tables
- Model info badge now shows different information per model mode
- Unet model panel only shows for Wan2.2 (not applicable to LTX-2)
