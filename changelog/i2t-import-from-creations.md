### Added
- **ImageToTextTool**: magic wand import-from-previous-generation button — select any image from My Media and load it directly into the Image to Text tool
- **MyMediaTool**: "📷 Image to Text" added to the "Use in tool" dropdown in lightbox

### Fixed
- `vite.config.js`: proxy `/comfyui` and `/media` routes to backend in dev mode, fixing CORS errors when fetching static media files via `fetch()`
