-- ============================================================================
-- Oelala User Profiles - Supabase Migration
-- User profile information for social features and personalization
-- ============================================================================

-- ============================================================================
-- Table: profiles
-- Stores user profile information (display name, avatar, bio, etc.)
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    username TEXT UNIQUE,
    display_name TEXT,
    avatar_url TEXT,
    bio TEXT CHECK (char_length(bio) <= 500),
    social_links JSONB DEFAULT '{}',
    is_public BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT username_length CHECK (char_length(username) >= 3 AND char_length(username) <= 30),
    CONSTRAINT username_format CHECK (username ~ '^[a-zA-Z0-9][a-zA-Z0-9_-]*[a-zA-Z0-9]$|^[a-zA-Z0-9]{1,2}$')
);

COMMENT ON TABLE public.profiles IS 'User profiles with display information and social features';
COMMENT ON COLUMN public.profiles.username IS 'Unique username (3-30 chars, alphanumeric + _ -)';
COMMENT ON COLUMN public.profiles.display_name IS 'Display name shown in UI';
COMMENT ON COLUMN public.profiles.avatar_url IS 'URL to user avatar image';
COMMENT ON COLUMN public.profiles.bio IS 'User biography (max 500 chars)';
COMMENT ON COLUMN public.profiles.social_links IS 'Social media links (twitter, instagram, etc.)';
COMMENT ON COLUMN public.profiles.is_public IS 'Whether profile is publicly visible';

-- ============================================================================
-- Indexes for Performance
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_profiles_username
    ON public.profiles(username) WHERE username IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_profiles_public
    ON public.profiles(is_public) WHERE is_public = true;

-- ============================================================================
-- Row Level Security (RLS)
-- ============================================================================
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Anyone can view public profiles
CREATE POLICY "Anyone can view public profiles"
    ON public.profiles FOR SELECT
    USING (is_public = true);

-- Users can view their own private profile
CREATE POLICY "Users can view own profile"
    ON public.profiles FOR SELECT
    USING (auth.uid() = id);

-- Users can insert their own profile
CREATE POLICY "Users can create own profile"
    ON public.profiles FOR INSERT
    WITH CHECK (auth.uid() = id);

-- Users can update their own profile
CREATE POLICY "Users can update own profile"
    ON public.profiles FOR UPDATE
    USING (auth.uid() = id)
    WITH CHECK (auth.uid() = id);

-- Users can delete their own profile
CREATE POLICY "Users can delete own profile"
    ON public.profiles FOR DELETE
    USING (auth.uid() = id);

-- Admins can view all profiles
CREATE POLICY "Admins can view all profiles"
    ON public.profiles FOR SELECT
    USING (
        (SELECT is_admin FROM public.user_credits WHERE user_id = auth.uid()) = true
    );

-- ============================================================================
-- Auto-update timestamp trigger
-- ============================================================================
CREATE OR REPLACE FUNCTION public.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_profiles_updated_at
    BEFORE UPDATE ON public.profiles
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

-- ============================================================================
-- Auto-create profile on signup
-- ============================================================================
CREATE OR REPLACE FUNCTION public.create_profile_on_signup()
RETURNS TRIGGER AS $$
DECLARE
    base_username TEXT;
    default_username TEXT;
BEGIN
    -- Generate base username from email (before @)
    base_username := split_part(NEW.email, '@', 1);

    -- Remove non-alphanumeric characters (except underscore and hyphen)
    base_username := regexp_replace(base_username, '[^a-zA-Z0-9_-]', '', 'g');

    -- Remove leading/trailing underscores and hyphens
    base_username := trim(both '_-' from base_username);

    -- If empty after sanitization, use 'user'
    IF base_username = '' OR base_username IS NULL THEN
        base_username := 'user';
    END IF;

    -- Ensure minimum length of 3 characters for username
    IF char_length(base_username) < 3 THEN
        base_username := base_username || repeat('x', 3 - char_length(base_username));
    END IF;

    default_username := base_username;

    -- Ensure uniqueness by appending random suffix if needed
    WHILE EXISTS (SELECT 1 FROM public.profiles WHERE username = default_username) LOOP
        default_username := base_username || '_' || substr(md5(random()::text), 1, 4);
    END LOOP;

    -- Create profile with default values
    INSERT INTO public.profiles (id, username, display_name, is_public)
    VALUES (
        NEW.id,
        default_username,
        COALESCE(NEW.raw_user_meta_data->>'full_name', split_part(NEW.email, '@', 1)),
        true
    )
    ON CONFLICT (id) DO NOTHING;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Drop existing trigger if exists (for re-running migration)
DROP TRIGGER IF EXISTS on_auth_user_created_profile ON auth.users;

-- Create trigger on auth.users
CREATE TRIGGER on_auth_user_created_profile
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.create_profile_on_signup();

-- ============================================================================
-- Helper function: Get user profile by username
-- ============================================================================
CREATE OR REPLACE FUNCTION public.get_profile_by_username(
    p_username TEXT
) RETURNS TABLE (
    id UUID,
    username TEXT,
    display_name TEXT,
    avatar_url TEXT,
    bio TEXT,
    social_links JSONB,
    is_public BOOLEAN,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        profiles.id,
        profiles.username,
        profiles.display_name,
        profiles.avatar_url,
        profiles.bio,
        profiles.social_links,
        profiles.is_public,
        profiles.created_at
    FROM public.profiles
    WHERE profiles.username = p_username
      AND (profiles.is_public = true OR profiles.id = auth.uid());
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- Grant necessary permissions
-- ============================================================================
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.profiles TO authenticated;

-- Service role can do everything (for backend)
GRANT ALL ON public.profiles TO service_role;

-- Allow users to execute profile lookup function
GRANT EXECUTE ON FUNCTION public.get_profile_by_username TO authenticated;
GRANT EXECUTE ON FUNCTION public.get_profile_by_username TO anon;
