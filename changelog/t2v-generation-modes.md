### Added

- **T2V Generation Modes**: Added support for multiple Text-to-Video base models
  - Wan2.2 14B: High quality T2V via T2I→I2V pipeline (default)
  - LTX-2 19B: Direct text-to-video, faster inference
- `/api/t2v-modes` endpoint to list available T2V models
- `T2V_GENERATION_MODES` config in comfyui_client.py
- `build_ltx2_t2v_workflow()` function for LTX-2 workflow building
- Frontend model selector in TextToVideoTool.jsx with dynamic frame limits

### Changed

- `/generate-text` endpoint now accepts `model_type` parameter (wan22 or ltx2)
- TextToVideoTool.jsx fetches available models from backend on mount
- Duration slider dynamically adjusts based on selected model's max frames
