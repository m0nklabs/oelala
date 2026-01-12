### Added

- **User suspension system**: Admins can now suspend/unsuspend users via `POST /api/admin/suspension/toggle`
- Suspended users receive HTTP 403 when attempting to generate content (via credits check)
- Suspension reason and timestamp tracked in database
- User list/detail endpoints now include `is_suspended`, `suspended_at`, `suspension_reason` fields
- Database migration `007_user_suspension.sql` adds suspension columns and RPC function
- Audit trail: suspension actions logged in `credit_transactions` table

### Technical Details

- New `SuspensionToggle` Pydantic model for API requests
- Suspension check integrated into `check_and_reserve()` in credits.py
- Admins cannot suspend themselves (safety check in RPC)
