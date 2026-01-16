### Added

- **Storage quota tracking and display** (#33)
  - New `GET /user/storage-quota` endpoint returning quota usage
  - `MediaService.get_user_quota()` method to fetch from oelala-storage
  - `StorageQuota.jsx` component with usage progress bar
  - Warning indicator when usage > 80%
  - Upgrade CTA button when usage > 95%

### Frontend Changes

- Added `StorageQuota` component to user dropdown menu
- Shows: used bytes, quota limit, percentage, tier
- Color-coded progress bar: green (ok), orange (warning), red (critical)
- Human-readable sizes (e.g., "2.5 GB / 10 GB")

### Backend Changes

- `MediaService.get_user_quota(user_id)` calls `GET /buckets/{user_id}` on oelala-storage
- Returns enriched quota info with `warning`, `upgrade_needed`, and `human_*` fields
- Graceful fallback when bucket doesn't exist yet (returns 10GB default)

### Integration Notes

- Relies on oelala-storage bucket API (already implemented)
- Quota info includes: used_bytes, quota_bytes, file_count, tier
