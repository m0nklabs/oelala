# Admin Panel Deployment Checklist

Use this checklist when deploying the admin panel to production.

## Pre-Deployment

- [ ] Review all code changes in PR
- [ ] Verify frontend builds successfully
- [ ] Read ADMIN_MIGRATION_GUIDE.md
- [ ] Backup Supabase database
- [ ] Test in staging environment (if available)

## Database Migration

- [ ] Open Supabase SQL Editor
- [ ] Copy contents of `src/backend/migrations/003_admin_system.sql`
- [ ] Review migration SQL before running
- [ ] Run migration in SQL Editor
- [ ] Verify columns added: `tier`, `is_vip`, `is_admin`
- [ ] Verify functions created: `admin_grant_credits`, `admin_update_tier`, `admin_toggle_status`
- [ ] Verify RLS policies created (3 policies)

## Grant Initial Admin Access

- [ ] Identify user to grant admin access
- [ ] Get user ID from Supabase auth.users table
- [ ] Run SQL: `UPDATE user_credits SET is_admin = true WHERE user_id = 'USER_ID';`
- [ ] Verify: `SELECT is_admin FROM user_credits WHERE user_id = 'USER_ID';`
- [ ] Document admin user in secure location

## Backend Deployment

- [ ] Merge PR to main branch
- [ ] Deploy backend changes
- [ ] Restart backend service: `sudo systemctl restart oelala-api`
- [ ] Check backend logs: `journalctl -u oelala-api -f`
- [ ] Verify no errors in logs
- [ ] Test admin check endpoint: `curl -X GET /api/admin/check` (should return 401)

## Frontend Deployment

- [ ] Build frontend: `npm run build`
- [ ] Deploy frontend build
- [ ] Clear CDN cache (if applicable)
- [ ] Verify no console errors

## Verification

- [ ] Log in with admin account
- [ ] Verify "Admin" section appears in sidebar
- [ ] Click "Admin Panel" (crown icon)
- [ ] Verify admin panel loads without errors
- [ ] Check system statistics display correctly
- [ ] Verify user list loads with pagination

## Functional Testing

- [ ] **Search**: Test user search by email
- [ ] **Filter**: Test tier filter (free/pro/vip)
- [ ] **Pagination**: Navigate through user pages
- [ ] **Expand Details**: Click user row to expand
- [ ] **Credit Adjustment**: 
  - Click "Adjust Credits"
  - Add 10 credits with reason "Test"
  - Verify balance updates
  - Check transaction appears in history
- [ ] **Tier Change**:
  - Select different tier from dropdown
  - Verify tier badge updates
- [ ] **VIP Toggle**:
  - Click "Grant VIP"
  - Verify crown icon appears
  - Click "Remove VIP"
  - Verify crown icon disappears
- [ ] **Transaction History**: Verify transactions display correctly

## Security Testing

- [ ] **Non-Admin Access**:
  - Log in with non-admin account
  - Verify "Admin" section NOT in sidebar
  - Try to access `/api/admin/users` (should return 403)
- [ ] **Unauthenticated Access**:
  - Log out
  - Try to access `/api/admin/users` (should return 401)
- [ ] **Audit Trail**:
  - Make credit adjustment
  - Verify transaction logged in `credit_transactions`
  - Verify `type='admin'` and `reference_id` contains admin ID

## Documentation

- [ ] Update internal wiki with admin panel location
- [ ] Share ADMIN_QUICK_REFERENCE.md with admins
- [ ] Set up admin action monitoring (optional)
- [ ] Document admin users and their responsibilities

## Post-Deployment

- [ ] Monitor backend logs for 24 hours
- [ ] Check for any error reports from admins
- [ ] Create second admin user (backup)
- [ ] Set up alerts for admin actions (optional)

## Rollback Plan

If issues occur, rollback with:

```sql
-- Remove admin columns (WARNING: Deletes admin data!)
ALTER TABLE user_credits
DROP COLUMN tier,
DROP COLUMN is_vip,
DROP COLUMN is_admin;

-- Drop functions
DROP FUNCTION IF EXISTS admin_grant_credits;
DROP FUNCTION IF EXISTS admin_update_tier;
DROP FUNCTION IF EXISTS admin_toggle_status;

-- Drop policies
DROP POLICY IF EXISTS "Admins can view all user credits" ON user_credits;
DROP POLICY IF EXISTS "Admins can update any user credits" ON user_credits;
DROP POLICY IF EXISTS "Admins can view all transactions" ON credit_transactions;
```

Then redeploy previous backend/frontend versions.

## Support

- [ ] Admin Panel Docs: `docs/ADMIN_PANEL.md`
- [ ] Migration Guide: `docs/ADMIN_MIGRATION_GUIDE.md`
- [ ] Quick Reference: `docs/ADMIN_QUICK_REFERENCE.md`
- [ ] Backend Logs: `journalctl -u oelala-api -f`

## Sign-Off

- [ ] Deployment completed by: _______________
- [ ] Date: _______________
- [ ] Tested by: _______________
- [ ] Approved by: _______________

## Notes

_Add any deployment notes, issues encountered, or deviations from the plan:_

---

