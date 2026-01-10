# Admin Panel & User Management System

This document describes the admin panel and user management system implementation for Oelala.

## Overview

The admin system allows authorized administrators to:
- View and manage all registered users
- Adjust user credit balances
- Change user tiers (free/pro/vip)
- Grant or revoke VIP status
- Grant or revoke admin status
- View credit transaction history
- Monitor system-wide statistics

## Database Schema

### Migration: `003_admin_system.sql`

Adds admin/tier columns to the existing `user_credits` table:

```sql
ALTER TABLE public.user_credits
ADD COLUMN tier TEXT DEFAULT 'free' CHECK (tier IN ('free', 'pro', 'vip')),
ADD COLUMN is_vip BOOLEAN DEFAULT false,
ADD COLUMN is_admin BOOLEAN DEFAULT false;
```

### Stored Functions

**`admin_grant_credits(user_id, amount, description, admin_id)`**
- Grants or subtracts credits from a user
- Requires caller to have `is_admin=true`
- Logs transaction with `type='admin'`
- Returns new balance on success

**`admin_update_tier(user_id, tier, admin_id)`**
- Updates user tier (free/pro/vip)
- Requires caller to have `is_admin=true`
- Validates tier value

**`admin_toggle_status(user_id, is_admin, is_vip)`**
- Toggles admin or VIP status
- Requires service role (backend only)
- Prevents privilege escalation

## Backend API

### Admin Endpoints (`/api/admin/*`)

All endpoints require authentication and admin privileges.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/admin/check` | GET | Check if current user is admin |
| `/api/admin/users` | GET | List users with pagination/filters |
| `/api/admin/users/{user_id}` | GET | Get specific user details |
| `/api/admin/credits/adjust` | POST | Adjust user credits |
| `/api/admin/tier/update` | POST | Update user tier |
| `/api/admin/status/toggle` | POST | Toggle admin/VIP status |
| `/api/admin/transactions/{user_id}` | GET | Get user transaction history |
| `/api/admin/stats` | GET | Get system statistics |

### Authentication Flow

1. User authenticates with JWT token
2. `get_admin_user` dependency checks `is_admin` flag in database
3. Returns 403 Forbidden if not admin
4. Executes admin action
5. Returns response

### Example: Adjust Credits

```python
POST /api/admin/credits/adjust
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "user_id": "uuid-here",
  "amount": 100,
  "reason": "Compensation for bug #123"
}
```

Response:
```json
{
  "success": true,
  "new_balance": 125,
  "message": "Credits adjusted by 100"
}
```

## Frontend Integration

### AuthContext

Added `isAdmin` state to track admin status:

```javascript
const { user, isAdmin } = useAuth()
```

On user login, calls `/api/admin/check` to determine admin status.

### Admin Panel Tool

**Location:** `src/frontend/src/dashboard/tools/AdminPanelTool.jsx`

**Features:**
- **User List**: Search, filter by tier, pagination
- **User Details**: Expandable rows with full user info
- **Credit Adjustment**: Modal for adding/subtracting credits
- **Tier Management**: Dropdown to change user tier
- **Status Toggles**: Buttons to grant/revoke VIP/Admin
- **Transaction History**: Recent transactions per user
- **System Stats**: Dashboard with key metrics

### Navigation

Admin panel appears in sidebar only when `isAdmin=true`:

```javascript
// nav.js
{
  id: 'admin',
  title: 'Admin',
  adminOnly: true,
  items: [
    { id: 'admin-panel', label: 'Admin Panel', status: 'new' },
  ],
}

// Sidebar.jsx
{NAV_GROUPS.map((group) => {
  if (group.adminOnly && !isAdmin) return null
  // ... render group
})}
```

### UI Components

**Stats Cards:**
- Total Users
- Total Credits Issued
- Total Credits Used
- VIP Users

**User Row:**
- Email, User ID
- Current balance
- Tier badge (color-coded)
- Admin/VIP icons
- Expandable details

**Credit Adjustment Modal:**
- Amount input (positive/negative)
- Reason field (required for audit)
- Confirmation/Cancel buttons

## Security

### Row Level Security (RLS)

Admin-specific policies in Supabase:

```sql
-- Admins can view all user credits
CREATE POLICY "Admins can view all user credits"
    ON public.user_credits FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM public.user_credits uc
        WHERE uc.user_id = auth.uid() AND uc.is_admin = true
    ));

-- Admins can update any user credits
CREATE POLICY "Admins can update any user credits"
    ON public.user_credits FOR UPDATE
    USING (EXISTS (
        SELECT 1 FROM public.user_credits uc
        WHERE uc.user_id = auth.uid() AND uc.is_admin = true
    ));
```

### Audit Trail

All admin actions are logged in `credit_transactions`:

```sql
INSERT INTO credit_transactions (user_id, amount, type, description, reference_id, metadata)
VALUES (
  target_user_id,
  amount,
  'admin',
  'Reason provided by admin',
  admin_user_id,
  '{"admin_id": "uuid"}'
);
```

This provides full traceability:
- Who performed the action (admin_id)
- What was changed (amount, type)
- When it happened (created_at)
- Why it was done (description)

### Permission Model

**Admin (`is_admin=true`):**
- View all users
- Adjust credits
- Change tiers
- Toggle VIP status
- Toggle admin status (via service role only)
- View all transactions
- Access admin panel

**VIP (`is_vip=true`):**
- Future: Priority queue
- Future: Exclusive models
- Future: Credit discounts

**Regular User:**
- View own data
- Generate content
- Purchase credits

## Granting Admin Access

### Method 1: Direct Database Update

Connect to Supabase SQL Editor and run:

```sql
UPDATE user_credits
SET is_admin = true
WHERE user_id = 'your-user-uuid';
```

### Method 2: Via Service Role API

Use the backend service role to call `admin_toggle_status`:

```bash
curl -X POST https://your-backend/api/admin/status/toggle \
  -H "Authorization: Bearer SERVICE_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "uuid-here",
    "is_admin": true
  }'
```

### Method 3: Initial Setup Script

Create a one-time setup script:

```python
from supabase import create_client

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Grant admin to first user
supabase.rpc('admin_toggle_status', {
    'p_user_id': 'first-user-uuid',
    'p_is_admin': True
}).execute()
```

## Testing

Run the test suite:

```bash
cd /home/runner/work/oelala/oelala
python3 -m pytest tests/test_admin_integration.py -v
```

Tests document:
- All admin endpoints
- Authorization flow
- Database schema
- Frontend integration
- Security model
- Audit trail

## Future Enhancements

### P1 - Content Moderation
- [ ] Review flagged content queue
- [ ] Remove published media
- [ ] Override NSFW classification
- [ ] User ban/suspension

### P2 - Tier Benefits
- [ ] Pro tier: 10% credit discount
- [ ] VIP tier: 20% credit discount, priority queue
- [ ] Custom tier limits per generation type

### P3 - Advanced Analytics
- [ ] User engagement metrics
- [ ] Revenue analytics
- [ ] Popular model tracking
- [ ] Generation success rates

### P4 - Bulk Operations
- [ ] Bulk credit grants
- [ ] Bulk tier changes
- [ ] CSV export/import
- [ ] Automated tier upgrades based on usage

## Troubleshooting

### "Access Denied" in Admin Panel

1. Verify user has `is_admin=true` in database:
   ```sql
   SELECT is_admin FROM user_credits WHERE user_id = 'your-uuid';
   ```

2. Check `/api/admin/check` returns `{"is_admin": true}`

3. Verify RLS policies are active:
   ```sql
   SELECT * FROM pg_policies WHERE tablename = 'user_credits';
   ```

### Credits Not Updating

1. Check transaction log:
   ```sql
   SELECT * FROM credit_transactions 
   WHERE user_id = 'target-user-uuid' 
   ORDER BY created_at DESC 
   LIMIT 5;
   ```

2. Verify `admin_grant_credits` function returns success
3. Check backend logs for errors

### Frontend Not Showing Admin Nav

1. Check AuthContext `isAdmin` state
2. Verify `/api/admin/check` endpoint is accessible
3. Check browser console for errors
4. Ensure user is logged in (JWT token present)

## Support

For issues or questions:
1. Check backend logs: `journalctl -u oelala-api -f`
2. Check browser console for frontend errors
3. Review audit trail in credit_transactions table
4. Contact support with user_id and timestamp

## Related Documentation

- [Credits System](../docs/CREDITS.md)
- [Database Migrations](../src/backend/migrations/)
- [API Documentation](../docs/API.md)
- [Security Best Practices](../docs/SECURITY.md)
