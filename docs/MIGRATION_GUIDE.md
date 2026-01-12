# Supabase Database Migration Guide

This guide walks you through setting up the complete Oelala database schema in Supabase.

## Prerequisites

- Supabase project created (cloud or self-hosted)
- Access to Supabase SQL Editor
- Database backup recommended before running migrations

## Migration Files Overview

| File | Description | Tables Created |
|------|-------------|----------------|
| `001_credits_system.sql` | Credit system & payments | user_credits, credit_transactions, credit_packages |
| `002_published_media.sql` | Published media & likes | published_media, published_media_likes |
| `003_admin_system.sql` | Admin permissions | Adds columns to user_credits |
| `004_api_keys.sql` | API key management | api_keys |
| `005_user_profiles.sql` | User profiles | profiles |
| `006_user_media.sql` | Media ownership & gallery | user_media, gallery |

## Migration Execution Order

⚠️ **CRITICAL**: Run migrations in numerical order (001 → 006)

### Step-by-Step Instructions

#### 1. Open Supabase SQL Editor

1. Go to your Supabase project dashboard
2. Navigate to **SQL Editor** in the left sidebar
3. Click **New Query**

#### 2. Run Migration 001 - Credits System

```bash
# Copy file contents
cat src/backend/migrations/001_credits_system.sql
```

1. Paste the entire SQL into the editor
2. Click **Run** or press `Ctrl+Enter`
3. Wait for "Success. No rows returned" message
4. Verify tables created:

```sql
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name IN ('user_credits', 'credit_transactions', 'credit_packages');
```

Expected output: 3 rows

#### 3. Run Migration 002 - Published Media

```bash
cat src/backend/migrations/002_published_media.sql
```

1. **New Query** in SQL Editor
2. Paste SQL and **Run**
3. Verify tables:

```sql
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name IN ('published_media', 'published_media_likes');
```

Expected output: 2 rows

#### 4. Run Migration 003 - Admin System

```bash
cat src/backend/migrations/003_admin_system.sql
```

1. **New Query** in SQL Editor
2. Paste SQL and **Run**
3. Verify columns added:

```sql
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'user_credits'
AND column_name IN ('tier', 'is_vip', 'is_admin');
```

Expected output: 3 rows

#### 5. Run Migration 004 - API Keys

```bash
cat src/backend/migrations/004_api_keys.sql
```

1. **New Query** in SQL Editor
2. Paste SQL and **Run**
3. Verify table:

```sql
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name = 'api_keys';
```

Expected output: 1 row

#### 6. Run Migration 005 - User Profiles

```bash
cat src/backend/migrations/005_user_profiles.sql
```

1. **New Query** in SQL Editor
2. Paste SQL and **Run**
3. Verify table:

```sql
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name = 'profiles';
```

Expected output: 1 row

#### 7. Run Migration 006 - User Media & Gallery

```bash
cat src/backend/migrations/006_user_media.sql
```

1. **New Query** in SQL Editor
2. Paste SQL and **Run**
3. Verify tables:

```sql
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name IN ('user_media', 'gallery');
```

Expected output: 2 rows

## Post-Migration Verification

### Check All Tables Created

```sql
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
```

Expected tables:
- api_keys
- credit_packages
- credit_transactions
- gallery
- profiles
- published_media
- published_media_likes
- user_credits
- user_media

### Check RLS Policies

```sql
SELECT schemaname, tablename, policyname
FROM pg_policies
WHERE schemaname = 'public'
ORDER BY tablename, policyname;
```

You should see multiple RLS policies for each table.

### Check Functions

```sql
SELECT routine_name
FROM information_schema.routines
WHERE routine_schema = 'public'
AND routine_type = 'FUNCTION'
ORDER BY routine_name;
```

Expected functions:
- add_credits
- admin_grant_credits
- admin_toggle_status
- admin_update_tier
- create_profile_on_signup
- create_user_credits_on_signup
- deduct_credits
- get_profile_by_username
- get_user_media
- increment_view_count
- publish_to_gallery
- toggle_like
- track_media_generation
- update_updated_at
- validate_api_key

### Check Triggers

```sql
SELECT trigger_name, event_object_table
FROM information_schema.triggers
WHERE trigger_schema = 'public'
ORDER BY event_object_table, trigger_name;
```

Expected triggers:
- on_auth_user_created_credits (on auth.users)
- on_auth_user_created_profile (on auth.users)
- update_profiles_updated_at (on profiles)
- update_published_media_updated_at (on published_media)
- update_user_credits_updated_at (on user_credits)
- update_user_media_updated_at (on user_media)

## Grant Initial Admin Access

After all migrations are complete, grant admin access to your primary user:

1. Find your user ID:

```sql
SELECT id, email FROM auth.users WHERE email = 'your-email@example.com';
```

2. Grant admin status:

```sql
UPDATE public.user_credits
SET is_admin = true
WHERE user_id = 'YOUR_USER_ID_HERE';
```

3. Verify:

```sql
SELECT u.email, uc.is_admin, uc.tier, uc.balance
FROM auth.users u
JOIN public.user_credits uc ON u.id = uc.user_id
WHERE uc.is_admin = true;
```

## Testing the Setup

### Test 1: User Credits Auto-Creation

Create a test user via Supabase Auth UI, then check:

```sql
SELECT uc.*, u.email
FROM public.user_credits uc
JOIN auth.users u ON uc.user_id = u.id
ORDER BY uc.created_at DESC
LIMIT 1;
```

Should show 25 welcome credits.

### Test 2: Profile Auto-Creation

```sql
SELECT p.*, u.email
FROM public.profiles p
JOIN auth.users u ON p.id = u.id
ORDER BY p.created_at DESC
LIMIT 1;
```

Should show auto-generated username and display name.

### Test 3: Credit Packages

```sql
SELECT * FROM public.credit_packages ORDER BY sort_order;
```

Should show 5 default packages (starter, basic, pro, studio, enterprise).

### Test 4: RLS Policies Work

Try accessing as a non-admin user (should fail):

```sql
-- This should return empty if executed as non-admin
SELECT * FROM public.user_credits WHERE user_id != auth.uid();
```

## Troubleshooting

### Error: "relation already exists"

- Safe to ignore if re-running migrations
- All migrations use `IF NOT EXISTS` / `IF EXISTS` for idempotency

### Error: "permission denied for schema public"

Run this to grant permissions:

```sql
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT USAGE ON SCHEMA public TO anon;
```

### Error: "function already exists"

Drop and recreate:

```sql
DROP FUNCTION IF EXISTS function_name CASCADE;
-- Then re-run the CREATE FUNCTION statement
```

### Tables created but no welcome credits

Re-run the trigger creation:

```sql
DROP TRIGGER IF EXISTS on_auth_user_created_credits ON auth.users;
CREATE TRIGGER on_auth_user_created_credits
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.create_user_credits_on_signup();
```

### RLS policies blocking service role

Ensure backend uses `SUPABASE_SERVICE_KEY` (not anon key):

```bash
# In .env
SUPABASE_SERVICE_KEY=eyJhbG... # Service role key
```

## Rollback Instructions

If you need to undo migrations (⚠️ **WARNING: This deletes all data**):

```sql
-- Drop tables in reverse order
DROP TABLE IF EXISTS public.gallery CASCADE;
DROP TABLE IF EXISTS public.user_media CASCADE;
DROP TABLE IF EXISTS public.profiles CASCADE;
DROP TABLE IF EXISTS public.api_keys CASCADE;
DROP TABLE IF EXISTS public.published_media_likes CASCADE;
DROP TABLE IF EXISTS public.published_media CASCADE;
DROP TABLE IF EXISTS public.credit_transactions CASCADE;
DROP TABLE IF EXISTS public.credit_packages CASCADE;
DROP TABLE IF EXISTS public.user_credits CASCADE;

-- Drop functions
DROP FUNCTION IF EXISTS public.track_media_generation CASCADE;
DROP FUNCTION IF EXISTS public.publish_to_gallery CASCADE;
DROP FUNCTION IF EXISTS public.get_user_media CASCADE;
DROP FUNCTION IF EXISTS public.get_profile_by_username CASCADE;
DROP FUNCTION IF EXISTS public.create_profile_on_signup CASCADE;
DROP FUNCTION IF EXISTS public.validate_api_key CASCADE;
DROP FUNCTION IF EXISTS public.admin_toggle_status CASCADE;
DROP FUNCTION IF EXISTS public.admin_update_tier CASCADE;
DROP FUNCTION IF EXISTS public.admin_grant_credits CASCADE;
DROP FUNCTION IF EXISTS public.increment_view_count CASCADE;
DROP FUNCTION IF EXISTS public.toggle_like CASCADE;
DROP FUNCTION IF EXISTS public.add_credits CASCADE;
DROP FUNCTION IF EXISTS public.deduct_credits CASCADE;
DROP FUNCTION IF EXISTS public.create_user_credits_on_signup CASCADE;
DROP FUNCTION IF EXISTS public.update_updated_at CASCADE;
```

## Production Checklist

Before deploying to production:

- [ ] Backup existing database
- [ ] Run migrations on staging environment first
- [ ] Test all API endpoints
- [ ] Verify RLS policies work correctly
- [ ] Grant admin access to production admin users
- [ ] Update Stripe price IDs in credit_packages
- [ ] Set up environment variables in backend
- [ ] Test user signup flow (auto-creation of credits + profile)
- [ ] Verify credit deduction works
- [ ] Test gallery publishing
- [ ] Monitor logs for errors

## Next Steps

After successful migration:

1. **Update Backend Environment**
   - Set `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` in `.env`
   - Restart backend service

2. **Configure Stripe**
   - Follow `docs/CREDITS_SETUP.md`
   - Update `stripe_price_id` in `credit_packages`

3. **Test Integration**
   - Run `tests/test_credits_integration.py`
   - Verify no 404 errors on `/api/credits`

4. **Deploy Frontend**
   - Build frontend with `npm run build`
   - Deploy to production

## Support

- **Migration Issues**: Check Supabase logs in Dashboard → Database → Logs
- **RLS Issues**: Review policies in Dashboard → Authentication → Policies
- **Backend Errors**: Check `journalctl -u oelala-api -f`
- **Documentation**: See `docs/USER_SYSTEM.md`, `docs/CREDITS_SETUP.md`

---

**Migration Version**: 006
**Last Updated**: 2026-01-12
