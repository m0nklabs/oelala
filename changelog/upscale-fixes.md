### Fixed
- SeedVR2 video upscale: corrected VAE model name and removed invalid DiT parameters
- SeedVR2 VRAM management: added tiled VAE encoding/decoding and increased BlockSwap to prevent OOM on 12GB GPU
- Removed non-existent upscale models from model list (RealESRGAN_x2plus, 4x_NMKD-Siax_200k)

### Added
- Downloaded missing ESRGAN models: RealESRGAN_x4plus_anime_6B, 4x-UltraSharp
- Downloaded SeedVR2 models: DiT 3B fp8 (3.2GB), VAE fp16 (479MB)
- All 3 video upscale presets now fully functional: fast (lanczos), balanced (ESRGAN), quality (SeedVR2)
