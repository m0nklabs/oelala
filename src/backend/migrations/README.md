# Supabase Database Migrations

This directory contains SQL migration files for setting up the Oelala database schema in Supabase.

## Migration Files

| File | Description | Tables Created |
|------|-------------|----------------|
| `001_credits_system.sql` | Credit system & Stripe payments | user_credits, credit_transactions, credit_packages |
| `002_published_media.sql` | Published media & community gallery | published_media, published_media_likes |
| `003_admin_system.sql` | Admin permissions & functions | Adds columns to user_credits |
| `004_api_keys.sql` | API key management for external integrations | api_keys |
| `005_user_profiles.sql` | User profiles with social features | profiles |
| `006_user_media.sql` | Media ownership tracking & gallery | user_media, gallery |

## Quick Start

### 1. Prerequisites
- Supabase project (cloud or self-hosted)
- Access to Supabase SQL Editor
- Environment variables configured in backend

### 2. Run Migrations

**⚠️ IMPORTANT**: Run migrations in numerical order (001 → 006)

1. Open Supabase SQL Editor in your dashboard
2. For each migration file (in order):
   - Copy the entire SQL contents
   - Paste into SQL Editor
   - Click **Run** or press `Ctrl+Enter`
   - Wait for "Success" message

### 3. Verify Installation

Check all tables were created:

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

## Detailed Documentation

For complete migration instructions, troubleshooting, and verification:
👉 **See [docs/MIGRATION_GUIDE.md](../../docs/MIGRATION_GUIDE.md)**

## What Each Migration Does

### 001_credits_system.sql
Creates the pay-as-you-go credit system:
- **user_credits**: User credit balances
- **credit_transactions**: Audit log of all credit movements
- **credit_packages**: Available packages for purchase
- Auto-creates 25 welcome credits on signup
- Includes atomic credit deduction function

### 002_published_media.sql
Handles community gallery features:
- **published_media**: User-published content
- **published_media_likes**: Like tracking
- RLS policies for SFW/NSFW content
- View count tracking
- Like/unlike toggle function

### 003_admin_system.sql
Adds admin functionality:
- Adds `tier`, `is_vip`, `is_admin` columns to user_credits
- Admin-only RLS policies
- Admin credit grant function
- Tier management function

### 004_api_keys.sql
API key management for programmatic access:
- **api_keys**: Stores API key hashes and metadata
- API key validation function
- Usage tracking (last_used_at, usage_count)

### 005_user_profiles.sql
User profile system:
- **profiles**: Username, display name, avatar, bio, social links
- Auto-creates profile on signup
- Unique username constraints
- Public/private profile visibility

### 006_user_media.sql
Media ownership and gallery extensions:
- **user_media**: Tracks all generated content per user
- **gallery**: Extended gallery metadata
- Links to published_media table
- Media tracking and publishing functions

## Row Level Security (RLS)

All tables have RLS policies enabled for security:

- **Users can only view/edit their own data**
- **Public content is accessible to everyone**
- **NSFW content requires authentication**
- **Admins have elevated access**

## Functions

Database helper functions included:

### Credit System
- `deduct_credits()` - Atomic credit deduction
- `add_credits()` - Add credits (purchase, bonus)

### Admin Functions
- `admin_grant_credits()` - Grant credits to user
- `admin_update_tier()` - Update user tier
- `admin_toggle_status()` - Toggle admin/VIP status

### Gallery Functions
- `toggle_like()` - Like/unlike content
- `increment_view_count()` - Track views

### Profile Functions
- `get_profile_by_username()` - Profile lookup
- `create_profile_on_signup()` - Auto-create profile

### Media Functions
- `track_media_generation()` - Log generated media
- `publish_to_gallery()` - Publish to community gallery
- `get_user_media()` - Fetch user's media library

## Triggers

Auto-triggered on events:

- **on_auth_user_created_credits** - Create user_credits with welcome bonus
- **on_auth_user_created_profile** - Create default profile
- **update_*_updated_at** - Auto-update timestamps

## Environment Variables

Backend requires these Supabase environment variables:

```bash
# .env
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJhbG...  # Service role key
SUPABASE_JWT_SECRET=your-jwt-secret
```

## Testing

After running migrations:

1. **Create Test User**: Sign up via frontend
2. **Verify Auto-Creation**:
   ```sql
   SELECT * FROM user_credits WHERE balance = 25;
   SELECT * FROM profiles ORDER BY created_at DESC LIMIT 1;
   ```
3. **Test Credit Deduction**:
   ```sql
   SELECT * FROM deduct_credits('USER_ID', 5, 'Test', 'test_ref', '{}');
   ```

## Rollback

⚠️ **WARNING**: Rollback deletes all data!

See [docs/MIGRATION_GUIDE.md](../../docs/MIGRATION_GUIDE.md) for rollback SQL.

## Troubleshooting

### "relation already exists"
Safe to ignore - migrations are idempotent.

### "permission denied"
Grant schema permissions:
```sql
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT USAGE ON SCHEMA public TO anon;
```

### Welcome credits not working
Check trigger exists:
```sql
SELECT * FROM information_schema.triggers 
WHERE trigger_name = 'on_auth_user_created_credits';
```

### Backend getting 404 errors
Ensure using `SUPABASE_SERVICE_KEY` (not anon key).

## Support

- **Migration Issues**: [docs/MIGRATION_GUIDE.md](../../docs/MIGRATION_GUIDE.md)
- **API Integration**: [docs/CREDITS_SETUP.md](../../docs/CREDITS_SETUP.md)
- **User System**: [docs/USER_SYSTEM.md](../../docs/USER_SYSTEM.md)

---

**Last Updated**: 2026-01-12  
**Migration Version**: 006
