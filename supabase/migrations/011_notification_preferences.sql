-- Migration: notification_preferences
-- Adds notification preferences to user profiles and tracks email notifications

-- Add notification preferences JSONB column to profiles
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS notification_preferences jsonb DEFAULT '{
  "email_on_job_complete": false,
  "email_on_job_failed": false
}'::jsonb;

COMMENT ON COLUMN profiles.notification_preferences IS 'User notification preferences (email_on_job_complete, email_on_job_failed)';

-- Email notification log — audit trail and deduplication
CREATE TABLE IF NOT EXISTS email_notifications (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    recipient_email text NOT NULL,
    event_type text NOT NULL CHECK (event_type IN ('job.completed', 'job.failed')),
    subject text NOT NULL,
    job_id text,                -- ComfyUI prompt_id
    job_type text,              -- e.g. 'text-to-video', 'image-to-video'
    status text NOT NULL DEFAULT 'sent' CHECK (status IN ('sent', 'failed', 'bounced')),
    error_message text,
    created_at timestamptz DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_email_notifications_user ON email_notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_email_notifications_created ON email_notifications(created_at DESC);

-- RLS
ALTER TABLE email_notifications ENABLE ROW LEVEL SECURITY;

-- Users can view their own notification history
CREATE POLICY "Users can view own email notifications"
    ON email_notifications FOR SELECT
    USING (auth.uid() = user_id);

-- Service role can insert
CREATE POLICY "Service role can manage email notifications"
    ON email_notifications FOR ALL
    USING (auth.role() = 'service_role');
