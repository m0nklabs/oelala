### Added

- New unified media API endpoint `/api/media/unified` that aggregates all media sources:
  - User's private storage (`users/{user_id}/`)
  - Generated media bucket (`generated/`) - admin only
  - ComfyUI local output (`comfyui-local/`) - admin only via symlink
- `listUnifiedMedia()` function in `api.js` for frontend consumption

### Changed

- **MyMediaTool.jsx**: Refactored `fetchMedia` from ~100 lines of chaos (3 separate API calls, manual merging) to ~40 lines using single unified API call
- Backend imports `check_admin` from `admin_api.py` for visibility control
- Created symlink `/home/flip/oelala/media/comfyui-local` → `ComfyUI/output/` for unified access

### Technical Details

- Unified endpoint respects admin status: regular users only see their own media
- Admin users see all three sources merged with `source` field identifying origin
- LazyMedia component with IntersectionObserver unchanged - works with new API structure
- Eliminates frontend complexity of coordinating multiple async fetches
