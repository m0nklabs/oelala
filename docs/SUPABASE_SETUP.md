# Supabase Setup Guide

## Project Information

| Item | Value |
|------|-------|
| Project Name | oelala |
| Project Ref | `nsbjwhxdkxnyggtuxjjp` |
| Region | West EU (Paris) |
| Dashboard | https://supabase.com/dashboard/project/nsbjwhxdkxnyggtuxjjp |

## Connection Information

### REST API (PostgREST)
```
URL: https://nsbjwhxdkxnyggtuxjjp.supabase.co
```
Used by the backend for all CRUD operations via `httpx`.

### Direct Database Connection
```
Host: db.nsbjwhxdkxnyggtuxjjp.supabase.co
Port: 5432
Database: postgres
User: postgres
```
⚠️ **Note**: Direct connection is IPv6-only. The local server doesn't have IPv6 connectivity.

### Connection Pooler (Supavisor)
```
Host: aws-0-eu-central-1.pooler.supabase.com
Port: 6543
User: postgres.nsbjwhxdkxnyggtuxjjp
```

## Secrets & Environment Variables

### GitHub Organization Secrets (m0nklabs)
Available to all repos in the org:

| Secret | Description |
|--------|-------------|
| `SUPABASE_URL` | REST API URL |
| `SUPABASE_ANON_KEY` | Public anon key for client-side |
| `SUPABASE_SERVICE_KEY` | Service role key (full access) |
| `SUPABASE_ACCESS_TOKEN` | CLI/Management API token |

### GitHub Repository Secrets (oelala)
Repo-specific:

| Secret | Description |
|--------|-------------|
| `DATABASE_PASSWORD` | PostgreSQL password |
| `DATABASE_URL` | Full connection string |

### Local Environment (.env)
```bash
# Backend uses these
SUPABASE_URL=https://nsbjwhxdkxnyggtuxjjp.supabase.co
SUPABASE_SERVICE_KEY=<service_role_key>
```

## Running Migrations

### Option 1: Supabase CLI (Recommended)
The CLI bypasses IPv6 issues by using Supabase's management API.

```bash
# Set token (or export SUPABASE_ACCESS_TOKEN)
export SUPABASE_ACCESS_TOKEN="sbp_..."

# Link project (one-time)
npx supabase link --project-ref nsbjwhxdkxnyggtuxjjp

# Push migrations
npx supabase db push

# Or execute raw SQL
npx supabase db execute --file src/backend/migrations/008_webhooks.sql
```

### Option 2: SQL Editor (Manual)
1. Go to https://supabase.com/dashboard/project/nsbjwhxdkxnyggtuxjjp/sql
2. Copy SQL from `src/backend/migrations/*.sql`
3. Execute

### Option 3: REST API (CRUD only)
The REST API cannot execute DDL (CREATE TABLE, etc). Only for data operations.

## Migration Files

Located in `src/backend/migrations/`:

| File | Description |
|------|-------------|
| `001_users.sql` | User profiles, auth triggers |
| `002_credits.sql` | Credit system |
| `003_api_keys.sql` | API key management |
| `004_media.sql` | Media/gallery storage |
| `005_admin.sql` | Admin roles, audit logs |
| `006_user_roles.sql` | Role-based permissions |
| `007_user_suspension.sql` | User suspension system |
| `008_webhooks.sql` | Webhook delivery system |

## Architecture Notes

### Why REST API instead of direct PostgreSQL?
1. **IPv6 limitation**: Server has no IPv6 connectivity
2. **RLS enforcement**: Row Level Security works with JWT tokens
3. **Scalability**: PostgREST handles connection pooling
4. **Simplicity**: No psycopg2/asyncpg dependency management

### Backend Implementation
The backend uses `httpx` with the service role key:

```python
# src/backend/supabase_client.py
headers = {
    "apikey": SUPABASE_SERVICE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    "Content-Type": "application/json"
}

# Example: Fetch user
response = await client.get(
    f"{SUPABASE_URL}/rest/v1/profiles?id=eq.{user_id}",
    headers=headers
)
```

## Troubleshooting

### "Network is unreachable" for direct connection
The database host resolves to IPv6 only. Use:
- REST API for data operations
- Supabase CLI for migrations
- SQL Editor in dashboard

### "Tenant or user not found" for pooler
The pooler requires username format: `postgres.{project_ref}`

### RLS blocking queries
Ensure you're using the service role key, not the anon key.
Service role bypasses RLS policies.

## Useful Commands

```bash
# List projects
SUPABASE_ACCESS_TOKEN="..." npx supabase projects list

# Dump schema
SUPABASE_ACCESS_TOKEN="..." npx supabase db dump --schema public

# Pull remote schema
SUPABASE_ACCESS_TOKEN="..." npx supabase db pull

# Check migration status
SUPABASE_ACCESS_TOKEN="..." npx supabase migration list
```

## Related Documentation

- [Database Implementation Summary](DATABASE_IMPLEMENTATION_SUMMARY.md)
- [API v1 Documentation](API_v1.md)
- [Webhook System](../src/backend/migrations/008_webhooks.sql)
