### Added

- `getMediaUrl()` utility function in frontend config for consistent URL handling
- Support for signed URLs from oelala-storage in all media components

### Changed

- Updated `MyMediaTool.jsx` to use getMediaUrl helper (14 locations)
- Updated `VideoGenerator.jsx` for signed URL support in download and playback
- Updated `ImageToVideoTool.jsx` for signed URL support
- Updated `OutputPanel.jsx` for history video selection
- Updated `Dashboard.jsx` and `QueueIndicator.jsx` for job completion URLs
- All frontend media components now handle both relative paths and full signed URLs
