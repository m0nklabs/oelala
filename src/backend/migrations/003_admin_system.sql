-- ============================================================================
-- Oelala Admin & User Management System - Supabase Migration
-- Run this in your Supabase SQL Editor to add admin functionality
-- ============================================================================

-- ============================================================================
-- Add admin/tier columns to user_credits table
-- ============================================================================
ALTER TABLE public.user_credits
ADD COLUMN IF NOT EXISTS tier TEXT DEFAULT 'free' CHECK (tier IN ('free', 'pro', 'vip')),
ADD COLUMN IF NOT EXISTS is_vip BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT false;

COMMENT ON COLUMN public.user_credits.tier IS 'User tier: free, pro, or vip';
COMMENT ON COLUMN public.user_credits.is_vip IS 'Whether user has VIP status';
COMMENT ON COLUMN public.user_credits.is_admin IS 'Whether user is a site administrator';

-- Create index for admin lookups
CREATE INDEX IF NOT EXISTS idx_user_credits_admin
    ON public.user_credits(is_admin) WHERE is_admin = true;

CREATE INDEX IF NOT EXISTS idx_user_credits_vip
    ON public.user_credits(is_vip) WHERE is_vip = true;

CREATE INDEX IF NOT EXISTS idx_user_credits_tier
    ON public.user_credits(tier);

-- ============================================================================
-- Update auto-create function to include tier
-- ============================================================================
CREATE OR REPLACE FUNCTION public.create_user_credits_on_signup()
RETURNS TRIGGER AS $$
DECLARE
    welcome_credits INTEGER := 25;
BEGIN
    -- Create credit balance with welcome bonus
    INSERT INTO public.user_credits (user_id, balance, lifetime_purchased, lifetime_used, tier, is_vip, is_admin)
    VALUES (NEW.id, welcome_credits, 0, 0, 'free', false, false)
    ON CONFLICT (user_id) DO NOTHING;

    -- Log the welcome bonus transaction
    INSERT INTO public.credit_transactions (user_id, amount, type, description)
    VALUES (NEW.id, welcome_credits, 'bonus', 'Welcome bonus - thanks for joining Oelala!')
    ON CONFLICT DO NOTHING;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- Admin-only RLS policies
-- ============================================================================

-- Allow admins to view all user credits (for admin panel)
DROP POLICY IF EXISTS "Admins can view all user credits" ON public.user_credits;
CREATE POLICY "Admins can view all user credits"
    ON public.user_credits FOR SELECT
    USING (
        (SELECT is_admin FROM public.user_credits WHERE user_id = auth.uid()) = true
    );

-- Allow admins to update any user's credits/tier/status
DROP POLICY IF EXISTS "Admins can update any user credits" ON public.user_credits;
CREATE POLICY "Admins can update any user credits"
    ON public.user_credits FOR UPDATE
    USING (
        (SELECT is_admin FROM public.user_credits WHERE user_id = auth.uid()) = true
    );

-- Allow admins to view all transactions
DROP POLICY IF EXISTS "Admins can view all transactions" ON public.credit_transactions;
CREATE POLICY "Admins can view all transactions"
    ON public.credit_transactions FOR SELECT
    USING (
        (SELECT is_admin FROM public.user_credits WHERE user_id = auth.uid()) = true
    );

-- ============================================================================
-- Admin Functions
-- ============================================================================

-- Function to grant admin credits to a user (admin only)
CREATE OR REPLACE FUNCTION public.admin_grant_credits(
    p_user_id UUID,
    p_amount INTEGER,
    p_description TEXT DEFAULT NULL,
    p_admin_id UUID DEFAULT NULL
) RETURNS TABLE (
    success BOOLEAN,
    new_balance INTEGER,
    error TEXT
) AS $$
DECLARE
    new_bal INTEGER;
    v_admin_id UUID;
    current_balance INTEGER;
BEGIN
    -- SECURITY: Always use authenticated user ID, ignore caller-supplied p_admin_id
    v_admin_id := auth.uid();

    -- Check if authenticated user is admin
    IF NOT EXISTS (
        SELECT 1 FROM public.user_credits
        WHERE user_id = v_admin_id AND is_admin = true
    ) THEN
        RETURN QUERY SELECT false, 0, 'Unauthorized: Admin access required'::TEXT;
        RETURN;
    END IF;

    -- Check if target user exists and get current balance
    SELECT balance INTO current_balance
    FROM public.user_credits
    WHERE user_id = p_user_id;

    IF NOT FOUND THEN
        RETURN QUERY SELECT false, 0, 'User not found'::TEXT;
        RETURN;
    END IF;

    -- Prevent negative balance
    IF current_balance + p_amount < 0 THEN
        RETURN QUERY SELECT false, current_balance, 'Insufficient credits: resulting balance would be negative'::TEXT;
        RETURN;
    END IF;

    -- Add credits
    UPDATE public.user_credits
    SET balance = balance + p_amount,
        updated_at = NOW()
    WHERE user_id = p_user_id
    RETURNING balance INTO new_bal;

    -- Log transaction with authenticated admin ID
    INSERT INTO public.credit_transactions (user_id, amount, type, description, reference_id, metadata)
    VALUES (
        p_user_id,
        p_amount,
        'admin',
        COALESCE(p_description, 'Admin credit adjustment'),
        v_admin_id::TEXT,
        jsonb_build_object('admin_id', v_admin_id)
    );

    RETURN QUERY SELECT true, new_bal, NULL::TEXT;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to update user tier (admin only)
CREATE OR REPLACE FUNCTION public.admin_update_tier(
    p_user_id UUID,
    p_tier TEXT,
    p_admin_id UUID DEFAULT NULL
) RETURNS TABLE (
    success BOOLEAN,
    error TEXT
) AS $$
DECLARE
    v_admin_id UUID;
BEGIN
    -- SECURITY: Always use authenticated user ID, ignore caller-supplied p_admin_id
    v_admin_id := auth.uid();

    -- Check if authenticated user is admin
    IF NOT EXISTS (
        SELECT 1 FROM public.user_credits
        WHERE user_id = v_admin_id AND is_admin = true
    ) THEN
        RETURN QUERY SELECT false, 'Unauthorized: Admin access required'::TEXT;
        RETURN;
    END IF;

    -- Validate tier
    IF p_tier NOT IN ('free', 'pro', 'vip') THEN
        RETURN QUERY SELECT false, 'Invalid tier value'::TEXT;
        RETURN;
    END IF;

    -- Update tier
    UPDATE public.user_credits
    SET tier = p_tier,
        updated_at = NOW()
    WHERE user_id = p_user_id;

    IF NOT FOUND THEN
        RETURN QUERY SELECT false, 'User not found'::TEXT;
        RETURN;
    END IF;

    RETURN QUERY SELECT true, NULL::TEXT;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to toggle admin status (super admin only - must be called via service role)
CREATE OR REPLACE FUNCTION public.admin_toggle_status(
    p_user_id UUID,
    p_is_admin BOOLEAN DEFAULT NULL,
    p_is_vip BOOLEAN DEFAULT NULL
) RETURNS TABLE (
    success BOOLEAN,
    error TEXT
) AS $$
BEGIN
    -- Update admin/vip status (only fields that are not NULL)
    UPDATE public.user_credits
    SET
        is_admin = COALESCE(p_is_admin, is_admin),
        is_vip = COALESCE(p_is_vip, is_vip),
        updated_at = NOW()
    WHERE user_id = p_user_id;

    IF NOT FOUND THEN
        RETURN QUERY SELECT false, 'User not found'::TEXT;
        RETURN;
    END IF;

    RETURN QUERY SELECT true, NULL::TEXT;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- Grant permissions
-- ============================================================================
GRANT EXECUTE ON FUNCTION public.admin_grant_credits TO service_role;
GRANT EXECUTE ON FUNCTION public.admin_update_tier TO service_role;
GRANT EXECUTE ON FUNCTION public.admin_toggle_status TO service_role;

-- Also allow authenticated users (RLS will check admin status)
GRANT EXECUTE ON FUNCTION public.admin_grant_credits TO authenticated;
GRANT EXECUTE ON FUNCTION public.admin_update_tier TO authenticated;
