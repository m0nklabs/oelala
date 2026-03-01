-- Migration: tool_profiles
-- Per-user, per-tool settings profiles with auto-save support
-- Stores all user-adjustable settings as JSONB for each tool (I2V, T2V, etc.)

CREATE TABLE IF NOT EXISTS tool_profiles (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    tool_name text NOT NULL,                -- e.g. 'image_to_video', 'text_to_video'
    profile_name text NOT NULL DEFAULT 'default',
    settings jsonb NOT NULL DEFAULT '{}'::jsonb,
    is_active boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT uq_tool_profiles_user_tool_name UNIQUE (user_id, tool_name, profile_name)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_tool_profiles_user_tool ON tool_profiles(user_id, tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_profiles_active ON tool_profiles(user_id, tool_name, is_active) WHERE is_active = true;

COMMENT ON TABLE tool_profiles IS 'Stores per-user settings profiles for each tool (I2V, T2V, etc). Auto-saved on every change.';
COMMENT ON COLUMN tool_profiles.tool_name IS 'Tool identifier: image_to_video, text_to_video, text_to_image, etc.';
COMMENT ON COLUMN tool_profiles.profile_name IS 'Profile name. "default" is auto-saved. Named profiles are user-created snapshots.';
COMMENT ON COLUMN tool_profiles.settings IS 'JSONB blob of all user-configurable settings for this tool.';
COMMENT ON COLUMN tool_profiles.is_active IS 'Which profile is currently loaded for this user+tool.';

-- RLS
ALTER TABLE tool_profiles ENABLE ROW LEVEL SECURITY;

-- Users can CRUD their own profiles only
CREATE POLICY "Users can view own tool profiles"
    ON tool_profiles FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own tool profiles"
    ON tool_profiles FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own tool profiles"
    ON tool_profiles FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own tool profiles"
    ON tool_profiles FOR DELETE
    USING (auth.uid() = user_id);

-- Service role bypass
CREATE POLICY "Service role full access to tool profiles"
    ON tool_profiles FOR ALL
    USING (auth.role() = 'service_role');

-- Function to ensure only one active profile per user+tool
CREATE OR REPLACE FUNCTION ensure_single_active_profile()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.is_active = true THEN
        UPDATE tool_profiles
        SET is_active = false, updated_at = now()
        WHERE user_id = NEW.user_id
          AND tool_name = NEW.tool_name
          AND id != NEW.id
          AND is_active = true;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_single_active_profile
    BEFORE INSERT OR UPDATE OF is_active ON tool_profiles
    FOR EACH ROW
    WHEN (NEW.is_active = true)
    EXECUTE FUNCTION ensure_single_active_profile();

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_tool_profile_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_tool_profile_updated
    BEFORE UPDATE ON tool_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_tool_profile_timestamp();
