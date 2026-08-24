### Added
- Local MiniMax-H3 generation (t2v + i2v) via the user's **Windows PC ComfyUI**
  - New `get_windows_comfyui_client()` factory in `comfyui_client.py` — a second
    ComfyUI server (default set via
    `COMFYUI_WINDOWS_HOST` / `COMFYUI_WINDOWS_PORT`) separate from the one on ai-kvm2
  - New local builders `build_local_minimax_h3_t2v_workflow()` /
    `build_local_minimax_h3_i2v_workflow()` — reuse the FL2VA cloud graph but with the
    **int8_convrot** model files on the Windows PC (checkpoint `minimax_h3_fl2va_pruned_int8_convrot.safetensors`,
    text encoder `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`, fp16/fp32 video/audio VAEs)
  - New adapters `MiniMaxH3LocalT2VAdapter` + `MiniMaxH3LocalI2VAdapter`
    (`minimax-h3-local-t2v` / `minimax-h3-local-i2v`); the I2V variant uploads its input
    image to the Windows server itself (`handles_own_image_upload=True`, router pre-upload skipped)
  - Registered in the adapter factory only when `COMFYUI_WINDOWS_HOST` is set
  - Frontend: new "🪟 MiniMax H3 — Lokaal (Windows PC)" mode in TextToVideo + ImageToVideo
    tools (local compute target, same FL2VA UI: MP-selector canvas, 24fps, no CFG/neg prompt)

### Changed
- Video-model cleanup: removed unused/duplicated Wan/LTX model files and dead workflows
  - Deleted orphaned physical models (freed ~98 GB): `smoothMixWan22` high/low,
    `wan22EnhancedNSFWCameraPrompt` high/low, `LTX-2-dev-Q2_K`, and the local LTX-2 19B set
    (`ltx-2-19b-dev-Q4_K_M`, `ltx-2-19b-distilled_Q4_K_M`, `ltx-2-19b-distilled-fp8`,
    `ltx-2-19b-embeddings_connector_bf16`, `LTX2_video_vae_bf16`, `ltx2_audio_vae`)
  - Removed 51 orphaned video workflow JSON's (ltx2 local T2V/I2V, dead wan22 experiments)
  - These were referenced by no active code path; cloud RunPod LTX/MiniMax generation is unaffected
