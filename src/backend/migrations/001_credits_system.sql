-- ============================================================================
-- Oelala Credits System - Supabase Migration
-- Run this in your Supabase SQL Editor to set up the credits tables
-- ============================================================================

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- Table: user_credits
-- Stores the current credit balance for each user
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.user_credits (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    balance INTEGER NOT NULL DEFAULT 0 CHECK (balance >= 0),
    lifetime_purchased INTEGER NOT NULL DEFAULT 0,
    lifetime_used INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE public.user_credits IS 'User credit balances for pay-as-you-go generation';
COMMENT ON COLUMN public.user_credits.balance IS 'Current available credits';
COMMENT ON COLUMN public.user_credits.lifetime_purchased IS 'Total credits ever purchased';
COMMENT ON COLUMN public.user_credits.lifetime_used IS 'Total credits ever used for generations';

-- ============================================================================
-- Table: credit_transactions
-- Audit log of all credit movements (purchases, usage, refunds)
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.credit_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    amount INTEGER NOT NULL,  -- Positive = credit added, negative = credit used
    type TEXT NOT NULL CHECK (type IN ('purchase', 'bonus', 'generation', 'refund', 'admin', 'promo')),
    description TEXT,
    reference_id TEXT,  -- Stripe payment_intent ID, job ID, promo code, etc.
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE public.credit_transactions IS 'Audit log of all credit transactions';
COMMENT ON COLUMN public.credit_transactions.amount IS 'Credits added (positive) or used (negative)';
COMMENT ON COLUMN public.credit_transactions.type IS 'Transaction type: purchase, bonus, generation, refund, admin, promo';
COMMENT ON COLUMN public.credit_transactions.reference_id IS 'External reference (Stripe ID, job ID, etc.)';

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_credit_transactions_user
    ON public.credit_transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_credit_transactions_created
    ON public.credit_transactions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_credit_transactions_type
    ON public.credit_transactions(type);
CREATE INDEX IF NOT EXISTS idx_credit_transactions_reference
    ON public.credit_transactions(reference_id) WHERE reference_id IS NOT NULL;

-- ============================================================================
-- Table: credit_packages
-- Available credit packages for purchase
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.credit_packages (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    credits INTEGER NOT NULL CHECK (credits > 0),
    price_cents INTEGER NOT NULL CHECK (price_cents > 0),
    currency TEXT DEFAULT 'EUR',
    stripe_price_id TEXT,  -- Stripe Price ID for checkout
    stripe_product_id TEXT,  -- Stripe Product ID
    is_active BOOLEAN DEFAULT true,
    sort_order INTEGER DEFAULT 0,
    description TEXT,
    badge TEXT,  -- e.g., 'POPULAR', 'BEST VALUE'
    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE public.credit_packages IS 'Credit packages available for purchase';

-- ============================================================================
-- Row Level Security (RLS)
-- ============================================================================

-- Enable RLS on all tables
ALTER TABLE public.user_credits ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.credit_transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.credit_packages ENABLE ROW LEVEL SECURITY;

-- user_credits: Users can only view their own balance
CREATE POLICY "Users can view own credits"
    ON public.user_credits FOR SELECT
    USING (auth.uid() = user_id);

-- credit_transactions: Users can only view their own transactions
CREATE POLICY "Users can view own transactions"
    ON public.credit_transactions FOR SELECT
    USING (auth.uid() = user_id);

-- credit_packages: Anyone can view active packages (public catalog)
CREATE POLICY "Anyone can view active packages"
    ON public.credit_packages FOR SELECT
    USING (is_active = true);

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

CREATE TRIGGER update_user_credits_updated_at
    BEFORE UPDATE ON public.user_credits
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

-- ============================================================================
-- Auto-create user credits on signup (with welcome bonus)
-- ============================================================================
CREATE OR REPLACE FUNCTION public.create_user_credits_on_signup()
RETURNS TRIGGER AS $$
DECLARE
    welcome_credits INTEGER := 25;
BEGIN
    -- Create credit balance with welcome bonus
    INSERT INTO public.user_credits (user_id, balance, lifetime_purchased, lifetime_used)
    VALUES (NEW.id, welcome_credits, 0, 0)
    ON CONFLICT (user_id) DO NOTHING;

    -- Log the welcome bonus transaction
    INSERT INTO public.credit_transactions (user_id, amount, type, description)
    VALUES (NEW.id, welcome_credits, 'bonus', 'Welcome bonus - thanks for joining Oelala!')
    ON CONFLICT DO NOTHING;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Drop existing trigger if exists (for re-running migration)
DROP TRIGGER IF EXISTS on_auth_user_created_credits ON auth.users;

-- Create trigger on auth.users
CREATE TRIGGER on_auth_user_created_credits
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.create_user_credits_on_signup();

-- ============================================================================
-- Insert default credit packages
-- ============================================================================
-- IMPORTANT: Replace 'price_xxx' with your actual Stripe Price IDs
-- Get Price IDs from: https://dashboard.stripe.com/test/products
-- Or create via CLI: stripe prices create --product=prod_xxx --unit-amount=500 --currency=eur

INSERT INTO public.credit_packages (id, name, credits, price_cents, currency, stripe_price_id, sort_order, description, badge) VALUES
    ('starter', 'Starter', 100, 500, 'EUR', 'price_xxx', 1, 'Perfect for trying out Oelala', NULL),
    ('basic', 'Basic', 500, 2000, 'EUR', 'price_xxx', 2, 'Great for regular creators', NULL),
    ('pro', 'Pro', 1500, 5000, 'EUR', 'price_xxx', 3, 'Best value for serious creators', 'POPULAR'),
    ('studio', 'Studio', 5000, 15000, 'EUR', 'price_xxx', 4, 'For power users and teams', 'BEST VALUE'),
    ('enterprise', 'Enterprise', 20000, 50000, 'EUR', 'price_xxx', 5, 'Maximum volume discount', NULL)
ON CONFLICT (id) DO UPDATE SET
    name = EXCLUDED.name,
    credits = EXCLUDED.credits,
    price_cents = EXCLUDED.price_cents,
    currency = EXCLUDED.currency,
    stripe_price_id = EXCLUDED.stripe_price_id,
    sort_order = EXCLUDED.sort_order,
    description = EXCLUDED.description,
    badge = EXCLUDED.badge;

-- ============================================================================
-- Helper function: Atomic credit deduction
-- Call this from backend to safely deduct credits
-- ============================================================================
CREATE OR REPLACE FUNCTION public.deduct_credits(
    p_user_id UUID,
    p_amount INTEGER,
    p_description TEXT DEFAULT NULL,
    p_reference_id TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'
) RETURNS TABLE (
    success BOOLEAN,
    new_balance INTEGER,
    error TEXT
) AS $$
DECLARE
    current_balance INTEGER;
    new_bal INTEGER;
BEGIN
    -- Lock the row and get current balance
    SELECT balance INTO current_balance
    FROM public.user_credits
    WHERE user_id = p_user_id
    FOR UPDATE;

    -- Check if user exists
    IF NOT FOUND THEN
        RETURN QUERY SELECT false, 0, 'User not found'::TEXT;
        RETURN;
    END IF;

    -- Check sufficient balance
    IF current_balance < p_amount THEN
        RETURN QUERY SELECT false, current_balance, 'Insufficient credits'::TEXT;
        RETURN;
    END IF;

    -- Deduct credits
    new_bal := current_balance - p_amount;

    UPDATE public.user_credits
    SET balance = new_bal,
        lifetime_used = lifetime_used + p_amount,
        updated_at = NOW()
    WHERE user_id = p_user_id;

    -- Log transaction
    INSERT INTO public.credit_transactions (user_id, amount, type, description, reference_id, metadata)
    VALUES (p_user_id, -p_amount, 'generation', p_description, p_reference_id, p_metadata);

    RETURN QUERY SELECT true, new_bal, NULL::TEXT;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- Helper function: Add credits (purchase, bonus, refund)
-- ============================================================================
CREATE OR REPLACE FUNCTION public.add_credits(
    p_user_id UUID,
    p_amount INTEGER,
    p_type TEXT,
    p_description TEXT DEFAULT NULL,
    p_reference_id TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'
) RETURNS TABLE (
    success BOOLEAN,
    new_balance INTEGER,
    error TEXT
) AS $$
DECLARE
    new_bal INTEGER;
    is_purchase BOOLEAN;
BEGIN
    is_purchase := p_type = 'purchase';

    -- Upsert user credits
    INSERT INTO public.user_credits (user_id, balance, lifetime_purchased, lifetime_used)
    VALUES (p_user_id, p_amount, CASE WHEN is_purchase THEN p_amount ELSE 0 END, 0)
    ON CONFLICT (user_id) DO UPDATE SET
        balance = public.user_credits.balance + p_amount,
        lifetime_purchased = public.user_credits.lifetime_purchased + CASE WHEN is_purchase THEN p_amount ELSE 0 END,
        updated_at = NOW()
    RETURNING balance INTO new_bal;

    -- Log transaction
    INSERT INTO public.credit_transactions (user_id, amount, type, description, reference_id, metadata)
    VALUES (p_user_id, p_amount, p_type, p_description, p_reference_id, p_metadata);

    RETURN QUERY SELECT true, new_bal, NULL::TEXT;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- Grant necessary permissions
-- ============================================================================
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT SELECT ON public.user_credits TO authenticated;
GRANT SELECT ON public.credit_transactions TO authenticated;
GRANT SELECT ON public.credit_packages TO authenticated;

-- Service role can do everything (for backend)
GRANT ALL ON public.user_credits TO service_role;
GRANT ALL ON public.credit_transactions TO service_role;
GRANT ALL ON public.credit_packages TO service_role;

-- Allow service role to execute functions
GRANT EXECUTE ON FUNCTION public.deduct_credits TO service_role;
GRANT EXECUTE ON FUNCTION public.add_credits TO service_role;
