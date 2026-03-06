### Changed
- Migrated all raw `fetch()` calls to `apiFetch()` across 14 frontend tool files for consistent JWT auth headers and Cloudflare CORS cache-busting
- Files: APIKeysTool, AdminPanelTool, AdminAnalyticsTab, AdminModerationTab, AdminSystemTab, VideoToVideoTool, VideoToTextTool, ImageToTextTool, ReframeTool, ProfileTool, TextToVideoTool, TextToImageTool, MyMediaTool, ImageToVideoTool

### Security
- Intentionally kept raw `fetch()` for blob:/data: URL loads and user-entered external URLs to prevent JWT token leakage to third-party servers
