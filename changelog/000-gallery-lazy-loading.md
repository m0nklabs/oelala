### Added

- **Gallery lazy loading**: Media items only load when scrolled into viewport
  - Each gallery item has its own IntersectionObserver for precise viewport detection
  - Media loads 100px before entering viewport for smooth experience
  - Uses `React.memo` for performance optimization
  - Shows placeholder icons until media loads
  - Smooth fade-in transition when media loads

- **Public gallery media endpoint**: `GET /api/gallery/{media_id}/file`
  - Serves published media files without authentication
  - Streams content from owner's storage (not viewer's)
  - Includes proper content-type detection and caching headers
  - Enables gallery viewing for anonymous users

### Changed

- **Gallery media URLs**: Changed from `/user/media/{storage_path}` to `/api/gallery/{id}/file`
  - Fixes issue where gallery couldn't display media from other users
  - Enables proper streaming without requiring viewer authentication
- Lazy-load gallery media previews and defer heavy media loading until items enter the viewport
- Stream user media from storage to reduce first-paint latency for large files

### Fixed

- Gallery items now load progressively as user scrolls instead of all at once
- Fixed 404 errors when viewing gallery items from other users
- Reduced initial page load from ~17s to near-instant with progressive loading
