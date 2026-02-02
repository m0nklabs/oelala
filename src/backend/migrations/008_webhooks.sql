-- Migration: Webhook Delivery System
-- Oelala Webhook System for async notifications
-- Run this in your Supabase SQL Editor

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- WEBHOOKS TABLE
-- Stores user-configured webhook endpoints
-- ============================================================================

CREATE TABLE IF NOT EXISTS webhooks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

    -- Endpoint configuration
    name VARCHAR(255) NOT NULL,                          -- Friendly name
    url TEXT NOT NULL,                                   -- Webhook URL (HTTPS only in prod)
    secret VARCHAR(255) NOT NULL,                        -- HMAC signing secret

    -- Event types to send (JSONB array)
    -- Values: 'job.queued', 'job.started', 'job.completed', 'job.failed'
    events JSONB NOT NULL DEFAULT '["job.completed", "job.failed"]'::jsonb,

    -- Status
    enabled BOOLEAN NOT NULL DEFAULT true,

    -- Metadata
    description TEXT,
    headers JSONB DEFAULT '{}'::jsonb,                   -- Custom headers to include

    -- Stats
    last_delivery_at TIMESTAMPTZ,
    last_delivery_status VARCHAR(50),                    -- 'success', 'failed', 'pending'
    total_deliveries INTEGER NOT NULL DEFAULT 0,
    successful_deliveries INTEGER NOT NULL DEFAULT 0,
    failed_deliveries INTEGER NOT NULL DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT webhooks_url_valid CHECK (url ~ '^https?://'),
    CONSTRAINT webhooks_events_valid CHECK (jsonb_typeof(events) = 'array')
);

-- Indexes for webhooks table
CREATE INDEX IF NOT EXISTS idx_webhooks_user_id ON webhooks(user_id);
CREATE INDEX IF NOT EXISTS idx_webhooks_enabled ON webhooks(enabled) WHERE enabled = true;
CREATE INDEX IF NOT EXISTS idx_webhooks_user_enabled ON webhooks(user_id, enabled);

-- ============================================================================
-- WEBHOOK DELIVERIES TABLE
-- Stores delivery attempts and history
-- ============================================================================

CREATE TABLE IF NOT EXISTS webhook_deliveries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    webhook_id UUID NOT NULL REFERENCES webhooks(id) ON DELETE CASCADE,

    -- Event data
    event_type VARCHAR(50) NOT NULL,                     -- 'job.queued', 'job.started', etc.
    event_id UUID NOT NULL DEFAULT uuid_generate_v4(),   -- Unique event identifier
    payload JSONB NOT NULL,                              -- Event payload sent

    -- Delivery status
    status VARCHAR(20) NOT NULL DEFAULT 'pending',       -- 'pending', 'success', 'failed', 'retrying'
    attempt_count INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 5,

    -- Response info
    response_status INTEGER,                             -- HTTP status code
    response_body TEXT,                                  -- Response body (truncated)
    response_time_ms INTEGER,                            -- Response time in milliseconds

    -- Error info
    error_message TEXT,

    -- Retry scheduling
    next_retry_at TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    delivered_at TIMESTAMPTZ,

    -- Indexes
    CONSTRAINT webhook_deliveries_status_valid CHECK (status IN ('pending', 'success', 'failed', 'retrying'))
);

-- Indexes for webhook_deliveries table
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_webhook_id ON webhook_deliveries(webhook_id);
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_status ON webhook_deliveries(status);
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_next_retry ON webhook_deliveries(next_retry_at)
    WHERE status = 'retrying' AND next_retry_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_event_type ON webhook_deliveries(event_type);
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_created_at ON webhook_deliveries(created_at DESC);

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function to update webhook stats after delivery
CREATE OR REPLACE FUNCTION update_webhook_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'success' AND OLD.status != 'success' THEN
        UPDATE webhooks
        SET
            last_delivery_at = NOW(),
            last_delivery_status = 'success',
            total_deliveries = total_deliveries + 1,
            successful_deliveries = successful_deliveries + 1,
            updated_at = NOW()
        WHERE id = NEW.webhook_id;
    ELSIF NEW.status = 'failed' AND OLD.status != 'failed' THEN
        UPDATE webhooks
        SET
            last_delivery_at = NOW(),
            last_delivery_status = 'failed',
            total_deliveries = total_deliveries + 1,
            failed_deliveries = failed_deliveries + 1,
            updated_at = NOW()
        WHERE id = NEW.webhook_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update stats
DROP TRIGGER IF EXISTS trigger_update_webhook_stats ON webhook_deliveries;
CREATE TRIGGER trigger_update_webhook_stats
    AFTER UPDATE OF status ON webhook_deliveries
    FOR EACH ROW
    EXECUTE FUNCTION update_webhook_stats();

-- Function to clean up old deliveries (keep last 30 days)
CREATE OR REPLACE FUNCTION cleanup_old_webhook_deliveries()
RETURNS void AS $$
BEGIN
    DELETE FROM webhook_deliveries
    WHERE created_at < NOW() - INTERVAL '30 days'
    AND status IN ('success', 'failed');
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================================

-- Enable RLS
ALTER TABLE webhooks ENABLE ROW LEVEL SECURITY;
ALTER TABLE webhook_deliveries ENABLE ROW LEVEL SECURITY;

-- Webhooks: Users can only manage their own webhooks
DROP POLICY IF EXISTS "Users can view own webhooks" ON webhooks;
CREATE POLICY "Users can view own webhooks" ON webhooks
    FOR SELECT USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can create own webhooks" ON webhooks;
CREATE POLICY "Users can create own webhooks" ON webhooks
    FOR INSERT WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update own webhooks" ON webhooks;
CREATE POLICY "Users can update own webhooks" ON webhooks
    FOR UPDATE USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete own webhooks" ON webhooks;
CREATE POLICY "Users can delete own webhooks" ON webhooks
    FOR DELETE USING (auth.uid() = user_id);

-- Service role can access all (for backend delivery)
DROP POLICY IF EXISTS "Service role can access all webhooks" ON webhooks;
CREATE POLICY "Service role can access all webhooks" ON webhooks
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- Webhook deliveries: Users can view deliveries for their webhooks
DROP POLICY IF EXISTS "Users can view own webhook deliveries" ON webhook_deliveries;
CREATE POLICY "Users can view own webhook deliveries" ON webhook_deliveries
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM webhooks
            WHERE webhooks.id = webhook_deliveries.webhook_id
            AND webhooks.user_id = auth.uid()
        )
    );

-- Service role can manage all deliveries
DROP POLICY IF EXISTS "Service role can access all deliveries" ON webhook_deliveries;
CREATE POLICY "Service role can access all deliveries" ON webhook_deliveries
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- ============================================================================
-- SEED DATA (Optional - for testing)
-- ============================================================================

-- Example webhook events for reference:
--
-- job.queued:
-- {
--   "event": "job.queued",
--   "event_id": "uuid",
--   "timestamp": "2024-01-15T12:00:00Z",
--   "data": {
--     "job_id": "prompt_id",
--     "job_type": "text-to-video",
--     "queue_position": 3,
--     "total_pending": 5,
--     "eta_seconds": 180
--   }
-- }
--
-- job.started:
-- {
--   "event": "job.started",
--   "event_id": "uuid",
--   "timestamp": "2024-01-15T12:02:00Z",
--   "data": {
--     "job_id": "prompt_id",
--     "job_type": "text-to-video"
--   }
-- }
--
-- job.completed:
-- {
--   "event": "job.completed",
--   "event_id": "uuid",
--   "timestamp": "2024-01-15T12:05:00Z",
--   "data": {
--     "job_id": "prompt_id",
--     "job_type": "text-to-video",
--     "output_url": "/media/generated/video_123.mp4",
--     "processing_time_seconds": 180
--   }
-- }
--
-- job.failed:
-- {
--   "event": "job.failed",
--   "event_id": "uuid",
--   "timestamp": "2024-01-15T12:03:00Z",
--   "data": {
--     "job_id": "prompt_id",
--     "job_type": "text-to-video",
--     "error": "Out of VRAM"
--   }
-- }

COMMENT ON TABLE webhooks IS 'User-configured webhook endpoints for async notifications';
COMMENT ON TABLE webhook_deliveries IS 'Webhook delivery attempts and history';
COMMENT ON COLUMN webhooks.events IS 'Array of event types: job.queued, job.started, job.completed, job.failed';
COMMENT ON COLUMN webhooks.secret IS 'HMAC-SHA256 signing secret for verifying webhook authenticity';
