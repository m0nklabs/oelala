### Added
- Created complete V2 payload structures for all primary frontend UI tools to utilize flat JSON representations via the `useGeneration` hook.

### Changed
- Refactored `ImageToVideoTool.jsx`, `TextToVideoTool.jsx`, `VideoToVideoTool.jsx`, `UpscaleTool.jsx`, and `ImageToImageTool.jsx` to completely replace legacy `postForm` usage with `useGeneration`.
- Enabled dynamic UI parameter mapping on the backend by changing `model_config = ConfigDict(extra="allow")` inside the V2 `GenerationRequest` base definition.
- Modified Pydantic generation adapters (e.g., `Wan22LocalI2VBase`, `ErnieLocalT2IAdapter`) to directly ingest arbitrary `req.model_extra` values down into the underlying workflow `kwargs` (e.g., `shift`, `high_noise_steps`, `enable_upscale`) when executing generations.

### Removed
- Removed deep dependencies on `FormData` encoding loops globally within the refactored tools, replacing them with `FileReader` routines to inject Base64 payloads directly into flat JSON arrays `input_images` and `input_audio`.
- Purged lingering unused `postForm` imports in modernized `.jsx` files.
