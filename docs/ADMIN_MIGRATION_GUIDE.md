# Admin Panel Migration Guide

This guide walks you through deploying the admin panel to your Oelala instance.

## Prerequisites

- Access to Supabase SQL Editor
- Admin access to your Oelala backend
- A test user account for verification

## Step 1: Run Database Migration

1. Open your Supabase project at https://supabase.com/dashboard
2. Navigate to **SQL Editor**
3. Click **New Query**
4. Copy the contents of `src/backend/migrations/003_admin_system.sql`
5. Paste into the SQL Editor
6. Click **Run** (or press Ctrl+Enter)

Expected output:
```
Success. No rows returned.
```

### Verify Migration

Run this query to verify the columns were added:

```sql
SELECT column_name, data_type, column_default
FROM information_schema.columns
WHERE table_name = 'user_credits'
AND column_name IN ('tier', 'is_vip', 'is_admin');
```

You should see:
- `tier` - TEXT - 'free'
- `is_vip` - BOOLEAN - false
- `is_admin` - BOOLEAN - false

## Step 2: Grant Admin Access

### Option A: Direct SQL Update (Recommended for initial setup)

1. Get your user ID by logging into Oelala
2. Open browser console and run:
   ```javascript
   console.log(localStorage.getItem('supabase.auth.token'))
   ```
3. Copy the JWT token and decode it at https://jwt.io
4. Find the `sub` field (your user ID)

5. In Supabase SQL Editor, run:
   ```sql
   UPDATE user_credits
   SET is_admin = true
   WHERE user_id = 'YOUR_USER_ID_HERE';
   ```

6. Verify:
   ```sql
   SELECT user_id, tier, is_vip, is_admin
   FROM user_credits
   WHERE user_id = 'YOUR_USER_ID_HERE';
   ```

### Option B: Via Backend API (For ongoing admin grants)

Once you have at least one admin, they can grant admin to others via the admin panel UI:

1. Log in as an admin
2. Navigate to Admin Panel (crown icon in sidebar)
3. Find the user in the list
4. Click to expand their details
5. Click "Grant Admin" button

## Step 3: Restart Backend (Optional)

If your backend is already running, restart it to load the new admin_api module:

```bash
sudo systemctl restart oelala-api
```

Verify backend is running:
```bash
sudo systemctl status oelala-api
journalctl -u oelala-api -f
```

## Step 4: Verify Frontend

1. Clear browser cache (Ctrl+Shift+Delete)
2. Navigate to your Oelala instance
3. Log in with your admin account
4. You should see **Admin** section in the sidebar
5. Click **Admin Panel** (👑 icon)

Expected view:
- System statistics cards at the top
- User list with search/filter
- All user management controls

## Step 5: Test Admin Functions

### Test 1: View Users
1. Click Admin Panel
2. Verify user list loads
3. Try search functionality
4. Try tier filter dropdown

### Test 2: Adjust Credits
1. Expand a user row
2. Click "Adjust Credits"
3. Enter amount: `100`
4. Enter reason: `Testing admin panel`
5. Click Confirm
6. Verify balance updates

### Test 3: Change Tier
1. Expand a user row
2. Select different tier from dropdown
3. Verify tier badge updates

### Test 4: Toggle VIP
1. Expand a user row
2. Click "Grant VIP" button
3. Verify crown icon appears
4. Click "Remove VIP" to revert

### Test 5: View Transactions
1. Expand a user row
2. Scroll to "Recent Transactions"
3. Verify your credit adjustment appears
4. Check amount, type, and date

## Troubleshooting

### Issue: "Access Denied" in Admin Panel

**Solution 1:** Verify admin flag in database
```sql
SELECT is_admin FROM user_credits WHERE user_id = 'YOUR_USER_ID';
```

If `false`, run the UPDATE query from Step 2.

**Solution 2:** Check backend logs
```bash
journalctl -u oelala-api -f
```

Look for:
```
👑 ADMIN-API: User YOUR_USER_ID is not an admin
```

**Solution 3:** Verify RLS policies
```sql
SELECT * FROM pg_policies 
WHERE tablename = 'user_credits' 
AND policyname LIKE '%Admin%';
```

You should see:
- `Admins can view all user credits`
- `Admins can update any user credits`

### Issue: Admin Nav Not Showing

**Solution 1:** Check AuthContext
1. Open browser console
2. Run:
   ```javascript
   // Assuming you can access React DevTools
   // Look for AuthContext.isAdmin value
   ```

**Solution 2:** Test admin check endpoint
```bash
curl -X GET https://your-backend/api/admin/check \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

Expected: `{"is_admin": true}`

**Solution 3:** Hard refresh browser
- Press Ctrl+Shift+R (Windows/Linux)
- Press Cmd+Shift+R (Mac)

### Issue: Credit Adjustment Not Working

**Solution 1:** Check transaction log
```sql
SELECT * FROM credit_transactions 
WHERE user_id = 'TARGET_USER_ID' 
ORDER BY created_at DESC 
LIMIT 5;
```

**Solution 2:** Verify function exists
```sql
SELECT routine_name 
FROM information_schema.routines 
WHERE routine_name LIKE 'admin_%';
```

You should see:
- `admin_grant_credits`
- `admin_update_tier`
- `admin_toggle_status`

**Solution 3:** Test function directly
```sql
SELECT * FROM admin_grant_credits(
  'TARGET_USER_ID'::uuid,
  100,
  'Test adjustment',
  'YOUR_ADMIN_ID'::uuid
);
```

Expected: `(true, NEW_BALANCE, NULL)`

### Issue: Users Not Loading

**Solution 1:** Check backend endpoint
```bash
curl -X GET https://your-backend/api/admin/users \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

**Solution 2:** Check Supabase service key
```bash
# In backend .env file
echo $SUPABASE_SERVICE_KEY
```

Must be set to your service role key (starts with `eyJ...`).

**Solution 3:** Verify user_credits table has data
```sql
SELECT COUNT(*) FROM user_credits;
```

## Security Checklist

After deployment, verify these security measures:

- [ ] Admin endpoints return 401 for unauthenticated requests
- [ ] Admin endpoints return 403 for non-admin users
- [ ] RLS policies are enabled on user_credits table
- [ ] RLS policies are enabled on credit_transactions table
- [ ] Admin actions are logged in credit_transactions
- [ ] Admin panel not visible to non-admin users
- [ ] Admin nav group not visible to non-admin users

Test security:

```bash
# Test without auth (should fail with 401)
curl -X GET https://your-backend/api/admin/users

# Test with non-admin user (should fail with 403)
curl -X GET https://your-backend/api/admin/users \
  -H "Authorization: Bearer NON_ADMIN_JWT"
```

## Post-Migration Tasks

### 1. Document Your Admins

Keep a record of admin users:

```sql
SELECT 
  uc.user_id,
  au.email,
  uc.created_at as user_since,
  uc.is_vip
FROM user_credits uc
JOIN auth.users au ON au.id = uc.user_id
WHERE uc.is_admin = true
ORDER BY uc.created_at;
```

### 2. Set Up Monitoring

Monitor admin actions:

```sql
-- Admin credit adjustments in last 7 days
SELECT 
  ct.created_at,
  ct.user_id,
  ct.amount,
  ct.description,
  ct.reference_id as admin_id
FROM credit_transactions ct
WHERE ct.type = 'admin'
AND ct.created_at > NOW() - INTERVAL '7 days'
ORDER BY ct.created_at DESC;
```

### 3. Create Backup Admin

Always have at least 2 admins:

```sql
-- Grant admin to second user
UPDATE user_credits
SET is_admin = true
WHERE user_id = 'SECOND_ADMIN_USER_ID';
```

## Rollback Plan

If you need to rollback the migration:

```sql
-- Remove admin columns (WARNING: This deletes admin data!)
ALTER TABLE user_credits
DROP COLUMN tier,
DROP COLUMN is_vip,
DROP COLUMN is_admin;

-- Drop admin functions
DROP FUNCTION IF EXISTS admin_grant_credits;
DROP FUNCTION IF EXISTS admin_update_tier;
DROP FUNCTION IF EXISTS admin_toggle_status;

-- Remove admin policies
DROP POLICY IF EXISTS "Admins can view all user credits" ON user_credits;
DROP POLICY IF EXISTS "Admins can update any user credits" ON user_credits;
DROP POLICY IF EXISTS "Admins can view all transactions" ON credit_transactions;
```

## Support

If you encounter issues not covered here:

1. Check backend logs: `journalctl -u oelala-api -f`
2. Check browser console for errors
3. Review [ADMIN_PANEL.md](./ADMIN_PANEL.md) documentation
4. Check database with SQL queries above
5. Contact support with:
   - User ID experiencing the issue
   - Timestamp of the error
   - Backend logs excerpt
   - Browser console errors

## Next Steps

Once admin panel is working:

1. Explore system statistics
2. Familiarize yourself with user management
3. Test credit adjustments with small amounts
4. Review audit trail in transactions
5. Set up regular admin action monitoring
6. Consider implementing additional features from [ADMIN_PANEL.md](./ADMIN_PANEL.md#future-enhancements)

## Related Documentation

- [Admin Panel Overview](./ADMIN_PANEL.md)
- [Database Migrations](../src/backend/migrations/)
- [Security Best Practices](./SECURITY.md)
- [Credits System](./CREDITS.md)
