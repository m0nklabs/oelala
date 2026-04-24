### Fixed
- Fixed RunPod job queue timeout (`Read timed out. (read timeout=30)`) when pushing large Base64-heavy workflows into ComfyUI's `/prompt` endpoint by bumping `requests.post` timeout from 30s to 120s across all RunPod handlers.
