-- ============================================================================
-- Oelala Published Media & Community Gallery - Supabase Migration
-- Run this in your Supabase SQL Editor to set up the gallery system
-- ============================================================================

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- Table: published_media
-- Stores user-published media items for the community gallery
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.published_media (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    storage_path TEXT NOT NULL,
    title TEXT NOT NULL CHECK (char_length(title) <= 100),
    description TEXT CHECK (char_length(description) <= 500),
    tags TEXT[] DEFAULT '{}',
    is_nsfw BOOLEAN DEFAULT FALSE,
    media_type TEXT NOT NULL CHECK (media_type IN ('video', 'image', 'audio')),
    thumbnail_url TEXT,
    metadata JSONB DEFAULT '{}',
    view_count INTEGER DEFAULT 0 CHECK (view_count >= 0),
    like_count INTEGER DEFAULT 0 CHECK (like_count >= 0),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE public.published_media IS 'User-published media items for community gallery';
COMMENT ON COLUMN public.published_media.storage_path IS 'Path to media file in user storage';
COMMENT ON COLUMN public.published_media.title IS 'User-provided title (max 100 chars)';
COMMENT ON COLUMN public.published_media.description IS 'Optional description (max 500 chars)';
COMMENT ON COLUMN public.published_media.tags IS 'Array of tags for filtering';
COMMENT ON COLUMN public.published_media.is_nsfw IS 'Whether content is NSFW (default: false for safety)';
COMMENT ON COLUMN public.published_media.metadata IS 'Prompt, settings, model info, etc.';

-- ============================================================================
-- Indexes for Performance
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_published_media_user 
    ON public.published_media(user_id);
CREATE INDEX IF NOT EXISTS idx_published_media_nsfw 
    ON public.published_media(is_nsfw);
CREATE INDEX IF NOT EXISTS idx_published_media_created 
    ON public.published_media(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_published_media_type 
    ON public.published_media(media_type);
CREATE INDEX IF NOT EXISTS idx_published_media_likes 
    ON public.published_media(like_count DESC);
CREATE INDEX IF NOT EXISTS idx_published_media_views 
    ON public.published_media(view_count DESC);

-- Composite index for common query pattern (SFW + sorting)
CREATE INDEX IF NOT EXISTS idx_published_media_sfw_created 
    ON public.published_media(is_nsfw, created_at DESC);

-- ============================================================================
-- Table: published_media_likes
-- Track user likes on published media
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.published_media_likes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    media_id UUID NOT NULL REFERENCES public.published_media(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(media_id, user_id)
);

COMMENT ON TABLE public.published_media_likes IS 'User likes on published media';

CREATE INDEX IF NOT EXISTS idx_published_media_likes_media 
    ON public.published_media_likes(media_id);
CREATE INDEX IF NOT EXISTS idx_published_media_likes_user 
    ON public.published_media_likes(user_id);

-- ============================================================================
-- Row Level Security (RLS)
-- ============================================================================

-- Enable RLS on all tables
ALTER TABLE public.published_media ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.published_media_likes ENABLE ROW LEVEL SECURITY;

-- published_media: Anyone can view SFW content, authenticated users can see NSFW
CREATE POLICY "Anyone can view SFW content" 
    ON public.published_media FOR SELECT 
    USING (is_nsfw = false);

CREATE POLICY "Authenticated users can view NSFW content" 
    ON public.published_media FOR SELECT 
    USING (is_nsfw = true AND auth.uid() IS NOT NULL);

-- published_media: Users can only insert/update/delete their own content
CREATE POLICY "Users can publish their own media" 
    ON public.published_media FOR INSERT 
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own published media" 
    ON public.published_media FOR UPDATE 
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own published media" 
    ON public.published_media FOR DELETE 
    USING (auth.uid() = user_id);

-- published_media_likes: Users can view all likes
CREATE POLICY "Anyone can view likes" 
    ON public.published_media_likes FOR SELECT 
    USING (true);

-- published_media_likes: Users can only manage their own likes
CREATE POLICY "Users can add their own likes" 
    ON public.published_media_likes FOR INSERT 
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can remove their own likes" 
    ON public.published_media_likes FOR DELETE 
    USING (auth.uid() = user_id);

-- ============================================================================
-- Auto-update timestamp trigger
-- ============================================================================
CREATE TRIGGER update_published_media_updated_at
    BEFORE UPDATE ON public.published_media
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

-- ============================================================================
-- Function: Increment view count
-- ============================================================================
CREATE OR REPLACE FUNCTION public.increment_view_count(p_media_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE public.published_media
    SET view_count = view_count + 1
    WHERE id = p_media_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- Function: Toggle like (add if not exists, remove if exists)
-- ============================================================================
CREATE OR REPLACE FUNCTION public.toggle_like(
    p_media_id UUID,
    p_user_id UUID
) RETURNS TABLE (
    liked BOOLEAN,
    like_count INTEGER
) AS $$
DECLARE
    v_liked BOOLEAN;
    v_count INTEGER;
BEGIN
    -- Try to delete existing like
    DELETE FROM public.published_media_likes
    WHERE media_id = p_media_id AND user_id = p_user_id;
    
    -- If nothing was deleted, insert new like
    IF NOT FOUND THEN
        INSERT INTO public.published_media_likes (media_id, user_id)
        VALUES (p_media_id, p_user_id);
        v_liked := true;
    ELSE
        v_liked := false;
    END IF;
    
    -- Update like count
    UPDATE public.published_media
    SET like_count = (
        SELECT COUNT(*) FROM public.published_media_likes
        WHERE media_id = p_media_id
    )
    WHERE id = p_media_id
    RETURNING like_count INTO v_count;
    
    RETURN QUERY SELECT v_liked, v_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- Grant necessary permissions
-- ============================================================================
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.published_media TO authenticated;
GRANT SELECT, INSERT, DELETE ON public.published_media_likes TO authenticated;

-- Service role can do everything (for backend)
GRANT ALL ON public.published_media TO service_role;
GRANT ALL ON public.published_media_likes TO service_role;

-- Allow service role to execute functions
GRANT EXECUTE ON FUNCTION public.increment_view_count TO service_role;
GRANT EXECUTE ON FUNCTION public.toggle_like TO service_role;

-- Also allow authenticated users to call toggle_like directly
GRANT EXECUTE ON FUNCTION public.toggle_like TO authenticated;
