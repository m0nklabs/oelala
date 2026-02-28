### Fixed

- Gallery like/view counting fully working (#25):
  - Replaced broken `toggle_like` RPC (used `auth.uid()` incompatible with service key) with direct Supabase table operations in `gallery_api.py`
  - Gallery list endpoint now batch-fetches `user_liked` per item for authenticated users (single extra query, not N+1)
  - `MediaDetailModal` now fetches fresh item data on open via `GET /api/gallery/{id}`, ensuring accurate `user_liked` state and correct view count increment
  - View count display in modal now reflects live data instead of stale list data
