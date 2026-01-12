-- ============================================================================
-- Oelala User Suspension System - Supabase Migration
-- Run this in your Supabase SQL Editor to add suspend/ban functionality
-- ============================================================================

-- ============================================================================
-- Add suspension columns to user_credits table
-- ============================================================================
ALTER TABLE public.user_credits
ADD COLUMN IF NOT EXISTS is_suspended BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS suspended_at TIMESTAMPTZ DEFAULT NULL,
ADD COLUMN IF NOT EXISTS suspension_reason TEXT DEFAULT NULL;

COMMENT ON COLUMN public.user_credits.is_suspended IS 'Whether user is suspended (blocked from generating)';
COMMENT ON COLUMN public.user_credits.suspended_at IS 'When the user was suspended';
COMMENT ON COLUMN public.user_credits.suspension_reason IS 'Admin note explaining why user was suspended';

-- Index for quickly finding suspended users
CREATE INDEX IF NOT EXISTS idx_user_credits_suspended
    ON public.user_credits(is_suspended) WHERE is_suspended = true;

-- ============================================================================
-- Admin Function: Suspend/Unsuspend User
-- ============================================================================
CREATE OR REPLACE FUNCTION public.admin_toggle_suspension(
    p_user_id UUID,
    p_is_suspended BOOLEAN,
    p_reason TEXT DEFAULT NULL,
    p_admin_id UUID DEFAULT NULL
) RETURNS TABLE (
    success BOOLEAN,
    error TEXT
) AS $$
DECLARE
    v_admin_id UUID;
    is_caller_admin BOOLEAN;
BEGIN
    -- SECURITY: Always use authenticated user ID
    v_admin_id := auth.uid();
    
    -- Check if caller is admin
    SELECT uc.is_admin INTO is_caller_admin
    FROM public.user_credits uc
    WHERE uc.user_id = v_admin_id;
    
    IF NOT is_caller_admin THEN
        RETURN QUERY SELECT false, 'Not authorized - admin only';
        RETURN;
    END IF;
    
    -- Cannot suspend yourself
    IF p_user_id = v_admin_id THEN
        RETURN QUERY SELECT false, 'Cannot suspend yourself';
        RETURN;
    END IF;
    
    -- Update suspension status
    UPDATE public.user_credits
    SET 
        is_suspended = p_is_suspended,
        suspended_at = CASE WHEN p_is_suspended THEN NOW() ELSE NULL END,
        suspension_reason = CASE WHEN p_is_suspended THEN p_reason ELSE NULL END,
        updated_at = NOW()
    WHERE user_id = p_user_id;
    
    IF NOT FOUND THEN
        RETURN QUERY SELECT false, 'User not found';
        RETURN;
    END IF;
    
    -- Log the action in credit_transactions for audit trail
    INSERT INTO public.credit_transactions (
        user_id,
        amount,
        type,
        description,
        metadata
    ) VALUES (
        p_user_id,
        0,
        'admin',
        CASE WHEN p_is_suspended 
            THEN 'Account suspended: ' || COALESCE(p_reason, 'No reason provided')
            ELSE 'Account unsuspended'
        END,
        jsonb_build_object(
            'action', CASE WHEN p_is_suspended THEN 'suspend' ELSE 'unsuspend' END,
            'admin_id', v_admin_id::text,
            'reason', p_reason
        )
    );
    
    RETURN QUERY SELECT true, NULL::TEXT;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION public.admin_toggle_suspension(UUID, BOOLEAN, TEXT, UUID) TO authenticated;
