### Added

- Avatar upload endpoint `POST /api/profile/me/avatar` — accepts JPEG/PNG/WebP/GIF (max 5 MB), auto-crops to centered 256×256 JPEG and saves to `media/avatars/`
- Public static route `/avatars/{user_id}.jpg` for serving avatar images without auth
- ProfileTool: camera button now opens a file picker; upload progress shown via spinner; URL text input removed
