### Added

- **Tier-based media retention policy**: Media uploads now include `X-Expires-At` header
  - `free` tier: 30 days retention
  - `pro` tier: 90 days retention  
  - `vip` tier: 365 days retention

- `MediaService.get_user_tier()` method to fetch user tier from Supabase
- `MediaService.calculate_expires_at()` helper for tier-based expiration
- `TIER_RETENTION_DAYS` config constant in media_service.py
- Retention metadata stored in Supabase: `tier`, `retention_days`, `expires_at`

### Changed

- `MediaService.upload()` now accepts optional `user_tier` parameter
- Upload headers now include `X-Expires-At` for oelala-storage GC integration
- Improved upload logging to show tier and expiration date

### Technical Notes

This completes the storage retention integration:
1. Backend calculates `expires_at` based on user tier
2. `X-Expires-At` header is sent to oelala-storage on upload
3. oelala-storage GC uses this to clean up expired files
4. Metadata is also stored in Supabase for frontend display
