-- ============================================================================
-- Oelala API Keys System - Supabase Migration
-- Programmatic API access with API keys for external integrations
-- ============================================================================

-- ============================================================================
-- Table: api_keys
-- Stores API keys for programmatic access
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,  -- User-friendly name (e.g., "My App", "Production Server")
    key_hash TEXT NOT NULL UNIQUE,  -- SHA-256 hash of the API key
    key_prefix TEXT NOT NULL,  -- First 15 chars for display (e.g., "oelala_12345678...")
    last_used_at TIMESTAMPTZ,
    usage_count INTEGER NOT NULL DEFAULT 0,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,  -- Optional expiration
    metadata JSONB DEFAULT '{}'  -- For rate limits, permissions, etc.
);

COMMENT ON TABLE public.api_keys IS 'API keys for programmatic access to Oelala';
COMMENT ON COLUMN public.api_keys.key_hash IS 'SHA-256 hash of the API key';
COMMENT ON COLUMN public.api_keys.key_prefix IS 'First 15 characters for display purposes';
COMMENT ON COLUMN public.api_keys.metadata IS 'Additional metadata (rate limits, scopes, etc.)';

-- Indexes for efficient lookups
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id
    ON public.api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash
    ON public.api_keys(key_hash) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_api_keys_active
    ON public.api_keys(is_active, user_id);

-- ============================================================================
-- Row Level Security (RLS)
-- ============================================================================
ALTER TABLE public.api_keys ENABLE ROW LEVEL SECURITY;

-- Users can view their own API keys
CREATE POLICY "Users can view own API keys"
    ON public.api_keys FOR SELECT
    USING (auth.uid() = user_id);

-- Users can create their own API keys
CREATE POLICY "Users can create own API keys"
    ON public.api_keys FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Users can update their own API keys (revoke, rename)
-- WITH CHECK prevents changing user_id to another user's ID
CREATE POLICY "Users can update own API keys"
    ON public.api_keys FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- Users can delete their own API keys
CREATE POLICY "Users can delete own API keys"
    ON public.api_keys FOR DELETE
    USING (auth.uid() = user_id);

-- ============================================================================
-- Helper function: Validate and update API key usage
-- Called by backend on each API request
-- ============================================================================
CREATE OR REPLACE FUNCTION public.validate_api_key(
    p_key_hash TEXT
) RETURNS TABLE (
    valid BOOLEAN,
    user_id UUID,
    key_id UUID,
    error TEXT
) AS $$
DECLARE
    key_record RECORD;
BEGIN
    -- Look up API key
    SELECT id, api_keys.user_id, is_active, expires_at
    INTO key_record
    FROM public.api_keys
    WHERE key_hash = p_key_hash
    FOR UPDATE;

    -- Check if key exists
    IF NOT FOUND THEN
        RETURN QUERY SELECT false, NULL::UUID, NULL::UUID, 'Invalid API key'::TEXT;
        RETURN;
    END IF;

    -- Check if key is active
    IF NOT key_record.is_active THEN
        RETURN QUERY SELECT false, NULL::UUID, NULL::UUID, 'API key is disabled'::TEXT;
        RETURN;
    END IF;

    -- Check if key is expired
    IF key_record.expires_at IS NOT NULL AND key_record.expires_at < NOW() THEN
        RETURN QUERY SELECT false, NULL::UUID, NULL::UUID, 'API key has expired'::TEXT;
        RETURN;
    END IF;

    -- Update usage stats
    UPDATE public.api_keys
    SET last_used_at = NOW(),
        usage_count = usage_count + 1
    WHERE id = key_record.id;

    -- Return success
    RETURN QUERY SELECT true, key_record.user_id, key_record.id, NULL::TEXT;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- Grant necessary permissions
-- ============================================================================
GRANT SELECT, INSERT, UPDATE, DELETE ON public.api_keys TO authenticated;
GRANT ALL ON public.api_keys TO service_role;
GRANT EXECUTE ON FUNCTION public.validate_api_key TO service_role;
