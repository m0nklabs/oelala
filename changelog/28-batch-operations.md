### Added

- Batch operations for media management (#28):
  - **Backend**: `POST /api/media/batch-download-zip` endpoint streams a ZIP archive of selected media files
    - Supports user storage items (`/user/media/`), generated media, ComfyUI output, and gallery items
    - Auth-protected, 50-item limit, skips unresolvable files gracefully
  - **MyMediaTool**: Replaced sequential download with proper ZIP download via the new endpoint
  - **Gallery**: Multi-select mode with "Select" toggle button (authenticated users only)
    - Click items to toggle selection, checkbox overlay shows selection state
    - Batch action bar: select all, clear selection, "Download ZIP (N)" button
    - Blue selection border highlights selected items
