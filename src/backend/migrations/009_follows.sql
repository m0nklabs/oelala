-- =============================================================================
-- Migration 009: Following System
-- =============================================================================
-- Creates the follows table for user follow/unfollow relationships.
-- Part of Issue #57: Following system (follow/unfollow users)
-- =============================================================================

-- Follows table
CREATE TABLE IF NOT EXISTS follows (
    follower_id  UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    following_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (follower_id, following_id),
    -- Prevent self-follows
    CONSTRAINT no_self_follow CHECK (follower_id != following_id)
);

-- Indexes for efficient lookups
CREATE INDEX IF NOT EXISTS idx_follows_follower  ON follows(follower_id);
CREATE INDEX IF NOT EXISTS idx_follows_following ON follows(following_id);
CREATE INDEX IF NOT EXISTS idx_follows_created   ON follows(created_at DESC);

-- Enable RLS
ALTER TABLE follows ENABLE ROW LEVEL SECURITY;

-- Policy: users can see all follows (public data)
CREATE POLICY follows_select_all ON follows
    FOR SELECT USING (true);

-- Policy: users can only insert their own follows
CREATE POLICY follows_insert_own ON follows
    FOR INSERT WITH CHECK (follower_id = auth.uid());

-- Policy: users can only delete their own follows
CREATE POLICY follows_delete_own ON follows
    FOR DELETE USING (follower_id = auth.uid());

-- Add follower/following count columns to profiles for fast lookups
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS follower_count  INT NOT NULL DEFAULT 0;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS following_count INT NOT NULL DEFAULT 0;

-- Function to update follower counts on follow/unfollow
CREATE OR REPLACE FUNCTION update_follow_counts()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        -- Increment following_count for follower
        UPDATE profiles SET following_count = following_count + 1
        WHERE id = NEW.follower_id;
        -- Increment follower_count for followed user
        UPDATE profiles SET follower_count = follower_count + 1
        WHERE id = NEW.following_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        -- Decrement following_count for follower
        UPDATE profiles SET following_count = GREATEST(following_count - 1, 0)
        WHERE id = OLD.follower_id;
        -- Decrement follower_count for followed user
        UPDATE profiles SET follower_count = GREATEST(follower_count - 1, 0)
        WHERE id = OLD.following_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to auto-update counts
DROP TRIGGER IF EXISTS trigger_update_follow_counts ON follows;
CREATE TRIGGER trigger_update_follow_counts
    AFTER INSERT OR DELETE ON follows
    FOR EACH ROW
    EXECUTE FUNCTION update_follow_counts();
