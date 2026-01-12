### Added

- **Background auto-upload for async video generation** ([#15](https://github.com/m0nklabs/oelala/issues/15))
  - Progress monitor now automatically uploads generated media when async jobs complete
  - Works for all async endpoints: `/generate-video`, `/generate-text-video`, `/generate-wan22-comfyui`
  - Uploads triggered via ComfyUI WebSocket `executing` event with `node_id=None`
  - Downloads output files from ComfyUI history and uploads to user storage
  - Runs in background thread to not block main event loop

### Fixed

- Media no longer lost when user disconnects before polling `/status` endpoint
- All video generations now reliably uploaded to user storage regardless of client polling behavior
