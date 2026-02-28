### Added
- Following system: users can follow/unfollow other users (#57)
- User profile pages accessible from gallery creator info
- Backend endpoints: follow, unfollow, followers list, following list, is-following check
- Gallery items enriched with creator username, display name, and avatar
- MediaDetailModal shows clickable creator info
- ProfileTool shows following count in stats
- Database migration 009_follows.sql with auto-updating counts via triggers
