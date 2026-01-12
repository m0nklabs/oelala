-- ============================================================================
-- Oelala User Media & Gallery - Supabase Migration
-- Track generated content ownership and provide gallery features
-- ============================================================================

-- ============================================================================
-- Table: user_media
-- Tracks all generated content and associates it with users
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.user_media (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    storage_path TEXT NOT NULL,
    media_type TEXT NOT NULL CHECK (media_type IN ('image', 'video', 'audio')),
    workflow_id TEXT,
    metadata JSONB DEFAULT '{}',
    is_nsfw BOOLEAN DEFAULT false,
    is_published BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT user_media_user_storage_unique UNIQUE (user_id, storage_path)
);

COMMENT ON TABLE public.user_media IS 'User-generated media with ownership tracking';
COMMENT ON COLUMN public.user_media.storage_path IS 'Path to media in oelala-storage';
COMMENT ON COLUMN public.user_media.media_type IS 'Type of media: image, video, or audio';
COMMENT ON COLUMN public.user_media.workflow_id IS 'ComfyUI workflow used for generation';
COMMENT ON COLUMN public.user_media.metadata IS 'Generation parameters, prompt, settings, etc.';
COMMENT ON COLUMN public.user_media.is_nsfw IS 'Whether content is NSFW';
COMMENT ON COLUMN public.user_media.is_published IS 'Whether published to gallery';

-- ============================================================================
-- Indexes for Performance
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_user_media_user
    ON public.user_media(user_id);
CREATE INDEX IF NOT EXISTS idx_user_media_type
    ON public.user_media(media_type);
CREATE INDEX IF NOT EXISTS idx_user_media_created
    ON public.user_media(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_media_published
    ON public.user_media(is_published) WHERE is_published = true;

-- ============================================================================
-- Table: gallery
-- Published content in the community gallery (references published_media)
-- This extends published_media with additional gallery-specific features
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.gallery (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    media_id UUID REFERENCES public.published_media(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    title TEXT NOT NULL CHECK (char_length(title) <= 100),
    description TEXT CHECK (char_length(description) <= 500),
    tags TEXT[] DEFAULT '{}',
    likes_count INTEGER DEFAULT 0 CHECK (likes_count >= 0),
    views_count INTEGER DEFAULT 0 CHECK (views_count >= 0),
    is_nsfw BOOLEAN DEFAULT false,
    published_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(media_id)
);

COMMENT ON TABLE public.gallery IS 'Gallery entries with extended metadata';
COMMENT ON COLUMN public.gallery.media_id IS 'Reference to published_media table';
COMMENT ON COLUMN public.gallery.title IS 'Gallery title (max 100 chars)';
COMMENT ON COLUMN public.gallery.description IS 'Gallery description (max 500 chars)';
COMMENT ON COLUMN public.gallery.tags IS 'Tags for search and filtering';
COMMENT ON COLUMN public.gallery.likes_count IS 'Cached like count';
COMMENT ON COLUMN public.gallery.views_count IS 'Cached view count';

-- ============================================================================
-- Indexes for Gallery Performance
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_gallery_user
    ON public.gallery(user_id);
CREATE INDEX IF NOT EXISTS idx_gallery_published
    ON public.gallery(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_gallery_likes
    ON public.gallery(likes_count DESC);
CREATE INDEX IF NOT EXISTS idx_gallery_views
    ON public.gallery(views_count DESC);
CREATE INDEX IF NOT EXISTS idx_gallery_nsfw
    ON public.gallery(is_nsfw);

-- Composite index for common query pattern
CREATE INDEX IF NOT EXISTS idx_gallery_sfw_published
    ON public.gallery(is_nsfw, published_at DESC);

-- GIN index for tag search
CREATE INDEX IF NOT EXISTS idx_gallery_tags
    ON public.gallery USING GIN (tags);

-- ============================================================================
-- Row Level Security (RLS)
-- ============================================================================
ALTER TABLE public.user_media ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.gallery ENABLE ROW LEVEL SECURITY;

-- user_media: Users can only view their own media
CREATE POLICY "Users can view own media"
    ON public.user_media FOR SELECT
    USING (auth.uid() = user_id);

-- user_media: Users can insert their own media
CREATE POLICY "Users can create own media"
    ON public.user_media FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- user_media: Users can update their own media
CREATE POLICY "Users can update own media"
    ON public.user_media FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- user_media: Users can delete their own media
CREATE POLICY "Users can delete own media"
    ON public.user_media FOR DELETE
    USING (auth.uid() = user_id);

-- user_media: Admins can view all media
CREATE POLICY "Admins can view all media"
    ON public.user_media FOR SELECT
    USING (
        (SELECT is_admin FROM public.user_credits WHERE user_id = auth.uid()) = true
    );

-- gallery: Anyone can view SFW content
CREATE POLICY "Anyone can view SFW gallery"
    ON public.gallery FOR SELECT
    USING (is_nsfw = false);

-- gallery: Authenticated users can view NSFW content
CREATE POLICY "Authenticated users can view NSFW gallery"
    ON public.gallery FOR SELECT
    USING (is_nsfw = true AND auth.uid() IS NOT NULL);

-- gallery: Users can publish their own content
CREATE POLICY "Users can publish to gallery"
    ON public.gallery FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- gallery: Users can update their own gallery entries
CREATE POLICY "Users can update own gallery entries"
    ON public.gallery FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- gallery: Users can delete their own gallery entries
CREATE POLICY "Users can delete own gallery entries"
    ON public.gallery FOR DELETE
    USING (auth.uid() = user_id);

-- gallery: Admins can view all gallery entries
CREATE POLICY "Admins can view all gallery entries"
    ON public.gallery FOR SELECT
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

CREATE TRIGGER update_user_media_updated_at
    BEFORE UPDATE ON public.user_media
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

-- ============================================================================
-- Helper function: Track media generation
-- Called by backend after successful generation
-- ============================================================================
CREATE OR REPLACE FUNCTION public.track_media_generation(
    p_user_id UUID,
    p_storage_path TEXT,
    p_media_type TEXT,
    p_workflow_id TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}',
    p_is_nsfw BOOLEAN DEFAULT false
) RETURNS TABLE (
    success BOOLEAN,
    media_id UUID,
    error TEXT
) AS $$
DECLARE
    new_media_id UUID;
BEGIN
    -- Insert new media record
    INSERT INTO public.user_media (
        user_id,
        storage_path,
        media_type,
        workflow_id,
        metadata,
        is_nsfw,
        is_published
    ) VALUES (
        p_user_id,
        p_storage_path,
        p_media_type,
        p_workflow_id,
        p_metadata,
        p_is_nsfw,
        false
    )
    ON CONFLICT (user_id, storage_path) DO UPDATE SET
        workflow_id = EXCLUDED.workflow_id,
        metadata = EXCLUDED.metadata,
        is_nsfw = EXCLUDED.is_nsfw,
        media_type = EXCLUDED.media_type,
        updated_at = NOW()
    RETURNING id INTO new_media_id;

    RETURN QUERY SELECT true, new_media_id, NULL::TEXT;
EXCEPTION
    WHEN OTHERS THEN
        RETURN QUERY SELECT false, NULL::UUID, SQLERRM::TEXT;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- Helper function: Publish media to gallery
-- ============================================================================
CREATE OR REPLACE FUNCTION public.publish_to_gallery(
    p_user_id UUID,
    p_media_id UUID,
    p_title TEXT,
    p_description TEXT DEFAULT NULL,
    p_tags TEXT[] DEFAULT '{}'
) RETURNS TABLE (
    success BOOLEAN,
    gallery_id UUID,
    error TEXT
) AS $$
DECLARE
    v_gallery_id UUID;
    v_media_type TEXT;
    v_storage_path TEXT;
    v_is_nsfw BOOLEAN;
    v_published_media_id UUID;
BEGIN
    -- Verify user owns the media
    SELECT media_type, storage_path, is_nsfw
    INTO v_media_type, v_storage_path, v_is_nsfw
    FROM public.user_media
    WHERE id = p_media_id AND user_id = p_user_id;

    IF NOT FOUND THEN
        RETURN QUERY SELECT false, NULL::UUID, 'Media not found or access denied'::TEXT;
        RETURN;
    END IF;

    -- Create published_media entry first
    INSERT INTO public.published_media (
        user_id,
        storage_path,
        title,
        description,
        tags,
        is_nsfw,
        media_type,
        metadata
    ) VALUES (
        p_user_id,
        v_storage_path,
        p_title,
        p_description,
        p_tags,
        v_is_nsfw,
        v_media_type,
        '{}'::JSONB
    )
    ON CONFLICT (user_id, storage_path) DO UPDATE SET
        title = EXCLUDED.title,
        description = EXCLUDED.description,
        tags = EXCLUDED.tags,
        updated_at = NOW()
    RETURNING id INTO v_published_media_id;

    -- Create gallery entry
    INSERT INTO public.gallery (
        media_id,
        user_id,
        title,
        description,
        tags,
        is_nsfw
    ) VALUES (
        v_published_media_id,
        p_user_id,
        p_title,
        p_description,
        p_tags,
        v_is_nsfw
    )
    ON CONFLICT (media_id) DO UPDATE SET
        title = EXCLUDED.title,
        description = EXCLUDED.description,
        tags = EXCLUDED.tags,
        published_at = NOW()
    RETURNING id INTO v_gallery_id;

    -- Mark user_media as published
    UPDATE public.user_media
    SET is_published = true,
        updated_at = NOW()
    WHERE id = p_media_id;

    RETURN QUERY SELECT true, v_gallery_id, NULL::TEXT;
EXCEPTION
    WHEN OTHERS THEN
        RETURN QUERY SELECT false, NULL::UUID, SQLERRM::TEXT;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- Helper function: Get user's media library
-- ============================================================================
CREATE OR REPLACE FUNCTION public.get_user_media(
    p_user_id UUID,
    p_media_type TEXT DEFAULT NULL,
    p_limit INTEGER DEFAULT 50,
    p_offset INTEGER DEFAULT 0
) RETURNS TABLE (
    id UUID,
    storage_path TEXT,
    media_type TEXT,
    workflow_id TEXT,
    is_nsfw BOOLEAN,
    is_published BOOLEAN,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        user_media.id,
        user_media.storage_path,
        user_media.media_type,
        user_media.workflow_id,
        user_media.is_nsfw,
        user_media.is_published,
        user_media.created_at
    FROM public.user_media
    WHERE user_media.user_id = p_user_id
      AND (p_media_type IS NULL OR user_media.media_type = p_media_type)
    ORDER BY user_media.created_at DESC
    LIMIT p_limit
    OFFSET p_offset;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- Grant necessary permissions
-- ============================================================================
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.user_media TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.gallery TO authenticated;

-- Service role can do everything (for backend)
GRANT ALL ON public.user_media TO service_role;
GRANT ALL ON public.gallery TO service_role;

-- Allow users to execute helper functions
GRANT EXECUTE ON FUNCTION public.track_media_generation TO service_role;
GRANT EXECUTE ON FUNCTION public.publish_to_gallery TO authenticated;
GRANT EXECUTE ON FUNCTION public.get_user_media TO authenticated;
