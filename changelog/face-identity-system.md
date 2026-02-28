### Added

- **`face_service.py`**: New backend service for face swap and face profile management
  - Direct insightface integration (insightface 0.7.3 + inswapper_128.onnx)
  - `detect_faces()` — high-accuracy face detection using buffalo_l (InsightFace)
  - `swap_faces()` / `swap_faces_to_bytes()` — direct Python face swap (no ComfyUI queue needed)
  - `create_face_profile()` — multi-image face profile with averaged embeddings
  - `list_face_profiles()`, `get_face_profile()`, `delete_face_profile()`
  - `swap_with_profile()` — face swap using a saved profile as source

- **Backend endpoints**:
  - `POST /detect-faces` — detect faces in image (now uses InsightFace, replaces OpenCV Haar cascade)
  - `POST /face-swap` — face swap returning PNG bytes directly (replaces ReActor/ComfyUI queue)
  - `POST /face-swap/profile` — face swap using a saved face profile
  - `GET /api/face-profiles` — list all saved face profiles
  - `GET /api/face-profiles/{id}` — get single profile
  - `POST /api/face-profiles` — create profile from 1+ reference photos
  - `DELETE /api/face-profiles/{id}` — delete profile and images

- **Lynx face identity for Wan2.2 video** (ComfyUI):
  - Installed `insightface` pip package (needed by WanVideoWrapper Lynx nodes)
  - Downloaded `lynx_lite_resampler_fp32.safetensors` (328MB) → `models/diffusion_models/WanVideo/lynx/`
  - Downloaded `Wan2_1-T2V-14B-Lynx_lite_ip_layers_fp16.safetensors` (839MB) → same dir
  - Lynx nodes: `LynxInsightFaceCrop`, `LynxEncodeFaceIP`, `WanVideoAddLynxEmbeds` (in WanVideoWrapper)

- **IP-Adapter FaceID for SDXL image generation** (ComfyUI):
  - Cloned `ComfyUI_IPAdapter_plus` custom node
  - Downloaded `ip-adapter-faceid-plusv2_sdxl.bin` (1.49GB) → `models/ipadapter/`
  - Downloaded `ip-adapter-faceid-plusv2_sdxl_lora.safetensors` (372MB) → `models/ipadapter/`

- **Face models downloaded**:
  - `inswapper_128.onnx` (554MB) → `models/insightface/` (face swap model)
  - `GFPGANv1.4.pth` (51MB) → `models/face_restore/` (face enhancement via comfy_mtb)
  - `buffalo_l` analyzer auto-downloads on first use to `models/insightface/models/`

### Changed

- `app.py`: Replaced ReActor-based face swap (GitHub-disabled) with direct insightface implementation
- `app.py`: `/detect-faces` now uses InsightFace buffalo_l (was naïve OpenCV Haar cascade)
- `app.py`: Added `import io` and `face_service` conditional import

### Notes

- Face LoRA training pipeline (from photos → identity LoRA) — deferred to next session
- Frontend face management UI — deferred to next session
- For Lynx video generation: use `LoadLynxResampler` + `LynxInsightFaceCrop` + `WanVideoAddLynxEmbeds` nodes in ComfyUI
- Face profiles stored in `data/face_profiles/` (not committed)
