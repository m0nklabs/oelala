### Changed

- **Backend**: Shared httpx.AsyncClient singleton for all Supabase admin API calls (was creating new TCP connection per request)
- **Backend**: TTL cache (60s) for admin status checks — eliminates repeated DB lookups per page load
- **Backend**: `asyncio.gather()` for parallel queries in profile stats, follow checks, and user detail endpoints
- **Backend**: Eliminated N+1 query in `list_users` — fetches only current page users' emails in parallel instead of ALL auth users
- **Backend**: `Prefer: count=exact` header on data queries eliminates separate count queries
- **Backend**: SELECT specific columns instead of `SELECT *` in transactions endpoint
- **Backend**: Reduced admin panel HTTP connections from ~30 to ~5 per page load
