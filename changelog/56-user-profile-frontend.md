### Added

- **User Profile Page** (#56)
  - New "My Profile" option in sidebar Account section
  - `ProfileTool.jsx` component with profile editing UI
  - Profile form: username, display name, bio, avatar URL
  - Social links: Twitter, Instagram, YouTube, GitHub, Website
  - Public/private profile toggle
  - Profile stats display (media count, published, likes, followers)

### Frontend Changes

- Added `MY_PROFILE` to `TOOL_IDS` in nav.js
- Added "My Profile" menu item with 👤 emoji to Account nav group
- Created `ProfileTool` component with:
  - Two-column layout (avatar/stats + form)
  - Input validation and character limits
  - Save/loading states with feedback alerts
  - Responsive styling matching app theme

### Integration

- Uses existing `/api/profile/me` backend endpoint (GET/PUT)
- Uses existing `/api/profile/me/stats` for stats display
- No new backend changes required (profile_api.py already complete)

### Also Fixed

- StorageQuota.jsx: Fixed API_BASE_URL → BACKEND_BASE import
