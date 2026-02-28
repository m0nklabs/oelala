#!/usr/bin/env python3
"""
Migration: gallery_likes + view_count support
Creates tables and RPC functions for like/view counting on published_media.

Run: /home/flip/venvs/gpu/bin/python scripts/migrate_gallery_likes.py
"""

import os
import sys

# Load env vars
from pathlib import Path
env_file = Path("/home/flip/oelala/.env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_KEY")
if not url or not key:
    print("❌ SUPABASE_URL / SUPABASE_SERVICE_KEY not set")
    sys.exit(1)

from supabase import create_client
sb = create_client(url, key)

SQL_MIGRATIONS = [
    # 1. Add view_count and like_count to published_media (idempotent)
    """
    ALTER TABLE published_media
      ADD COLUMN IF NOT EXISTS view_count  INTEGER NOT NULL DEFAULT 0,
      ADD COLUMN IF NOT EXISTS like_count  INTEGER NOT NULL DEFAULT 0;
    """,

    # 2. Create gallery_likes table (unique like per user+media)
    """
    CREATE TABLE IF NOT EXISTS published_media_likes (
      id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      media_id   UUID NOT NULL REFERENCES published_media(id) ON DELETE CASCADE,
      user_id    UUID NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      UNIQUE (media_id, user_id)
    );
    """,

    # 3. Index for quick lookup
    """
    CREATE INDEX IF NOT EXISTS idx_pm_likes_media_id ON published_media_likes(media_id);
    CREATE INDEX IF NOT EXISTS idx_pm_likes_user_id  ON published_media_likes(user_id);
    """,

    # 4. toggle_like function (returns liked bool + updated like_count)
    """
    CREATE OR REPLACE FUNCTION toggle_like(p_media_id UUID)
    RETURNS TABLE(liked BOOLEAN, like_count INTEGER)
    LANGUAGE plpgsql SECURITY DEFINER AS $$
    DECLARE
      v_user_id UUID := auth.uid();
      v_exists  BOOLEAN;
      v_count   INTEGER;
    BEGIN
      -- Check if already liked
      SELECT EXISTS(
        SELECT 1 FROM published_media_likes
        WHERE media_id = p_media_id AND user_id = v_user_id
      ) INTO v_exists;

      IF v_exists THEN
        -- Unlike
        DELETE FROM published_media_likes
        WHERE media_id = p_media_id AND user_id = v_user_id;

        UPDATE published_media
        SET like_count = GREATEST(0, like_count - 1)
        WHERE id = p_media_id;
      ELSE
        -- Like (prevent self-like: allowed here, enforce at app layer)
        INSERT INTO published_media_likes (media_id, user_id)
        VALUES (p_media_id, v_user_id)
        ON CONFLICT DO NOTHING;

        UPDATE published_media
        SET like_count = like_count + 1
        WHERE id = p_media_id;
      END IF;

      -- Return new state
      SELECT pm.like_count INTO v_count
      FROM published_media pm WHERE pm.id = p_media_id;

      RETURN QUERY SELECT NOT v_exists, v_count;
    END;
    $$;
    """,

    # 5. increment_view_count function
    """
    CREATE OR REPLACE FUNCTION increment_view_count(p_media_id UUID)
    RETURNS INTEGER
    LANGUAGE plpgsql SECURITY DEFINER AS $$
    DECLARE
      v_count INTEGER;
    BEGIN
      UPDATE published_media
      SET view_count = view_count + 1
      WHERE id = p_media_id
      RETURNING view_count INTO v_count;
      RETURN v_count;
    END;
    $$;
    """,

    # 6. RLS policies for published_media_likes
    """
    ALTER TABLE published_media_likes ENABLE ROW LEVEL SECURITY;
    """,
    """
    DO $$ BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'published_media_likes' AND policyname = 'anyone_can_read_likes'
      ) THEN
        CREATE POLICY anyone_can_read_likes ON published_media_likes
          FOR SELECT USING (true);
      END IF;
    END $$;
    """,
    """
    DO $$ BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'published_media_likes' AND policyname = 'users_manage_own_likes'
      ) THEN
        CREATE POLICY users_manage_own_likes ON published_media_likes
          FOR ALL USING (auth.uid() = user_id);
      END IF;
    END $$;
    """,
]

print(f"🔌 Connecting to {url}")
print(f"🔧 Running {len(SQL_MIGRATIONS)} migration steps...")

for i, sql in enumerate(SQL_MIGRATIONS, 1):
    try:
        sb.rpc("exec_sql", {"sql": sql}).execute()
        print(f"  ✅ Step {i} OK")
    except Exception as e:
        err = str(e)
        # Try alternative: use postgrest directly via admin endpoint
        try:
            import httpx
            resp = httpx.post(
                f"{url}/rest/v1/rpc/exec_sql",
                headers={"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"sql": sql},
                timeout=30,
            )
            if resp.status_code < 300:
                print(f"  ✅ Step {i} OK (via REST)")
            else:
                print(f"  ⚠️  Step {i}: {resp.status_code} {resp.text[:200]}")
        except Exception as e2:
            print(f"  ⚠️  Step {i}: {err[:150]}")

print("\n✅ Migration complete (check warnings above for any issues)")
print("\nNote: If steps failed because exec_sql RPC doesn't exist,")
print("run the SQL manually in the Supabase SQL editor:")
print(f"  {url.replace('.supabase.co', '.supabase.co')}/dashboard/project/{url.split('//')[1].split('.')[0]}/sql")
