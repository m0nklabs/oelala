### Added

- Video upscaling with three quality tiers: fast (lanczos), balanced (RealESRGAN), quality (SeedVR2)
- Image upscaling endpoint with auth, credit deduction, and progress tracking
- Video upscaling endpoint with auth, credit deduction, job registration, and WebSocket progress
- RealESRGAN x4 AI upscale model integration for per-frame video enhancement
- Quality presets API (`/upscale/models`) returning available models and preset definitions
- Credit costs: image ESRGAN (5), video lanczos/bicubic (5), video RealESRGAN (15), video SeedVR2 (30)
- Frontend "Upscale" button on MediaDetailModal for video items (owner only)
- Job tracking integration for upscale jobs (WebSocket progress, queue management)

### Fixed

- `/upscale` and `/upscale-video` endpoints now require authentication (were previously unprotected)
- Proper error handling with HTTPException re-raise in upscale endpoints
