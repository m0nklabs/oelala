-- =============================================================================
-- Content Moderation System
-- Tables for content reporting and moderation actions
-- =============================================================================

-- Add moderation_status to published_media
ALTER TABLE published_media
  ADD COLUMN IF NOT EXISTS moderation_status text DEFAULT 'approved'
    CHECK (moderation_status IN ('pending', 'approved', 'rejected', 'hidden'));

-- Content reports table
CREATE TABLE IF NOT EXISTS content_reports (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  media_id uuid NOT NULL REFERENCES published_media(id) ON DELETE CASCADE,
  reporter_id uuid NOT NULL,
  reason text NOT NULL CHECK (reason IN ('inappropriate', 'copyright', 'spam', 'harassment', 'underage', 'other')),
  description text,
  status text DEFAULT 'pending' CHECK (status IN ('pending', 'reviewed', 'dismissed')),
  reviewed_by uuid,
  reviewed_at timestamptz,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Moderation actions audit log
CREATE TABLE IF NOT EXISTS moderation_actions (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  media_id uuid NOT NULL REFERENCES published_media(id) ON DELETE CASCADE,
  moderator_id uuid NOT NULL,
  action text NOT NULL CHECK (action IN ('approve', 'reject', 'hide', 'unhide', 'warn_user', 'dismiss_report')),
  reason text,
  report_id uuid REFERENCES content_reports(id) ON DELETE SET NULL,
  created_at timestamptz DEFAULT now()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_content_reports_status ON content_reports(status);
CREATE INDEX IF NOT EXISTS idx_content_reports_media_id ON content_reports(media_id);
CREATE INDEX IF NOT EXISTS idx_content_reports_reporter ON content_reports(reporter_id);
CREATE INDEX IF NOT EXISTS idx_moderation_actions_media_id ON moderation_actions(media_id);
CREATE INDEX IF NOT EXISTS idx_published_media_mod_status ON published_media(moderation_status);

-- Prevent duplicate reports from same user on same media
CREATE UNIQUE INDEX IF NOT EXISTS idx_content_reports_unique_user_media
  ON content_reports(media_id, reporter_id)
  WHERE status = 'pending';

-- RLS policies for content_reports
ALTER TABLE content_reports ENABLE ROW LEVEL SECURITY;

-- Users can insert their own reports
CREATE POLICY "Users can create reports"
  ON content_reports FOR INSERT
  WITH CHECK (auth.uid() = reporter_id);

-- Users can see their own reports
CREATE POLICY "Users can view own reports"
  ON content_reports FOR SELECT
  USING (auth.uid() = reporter_id);

-- Service role can do everything (for admin API)
CREATE POLICY "Service role full access reports"
  ON content_reports FOR ALL
  USING (auth.role() = 'service_role');

-- RLS policies for moderation_actions
ALTER TABLE moderation_actions ENABLE ROW LEVEL SECURITY;

-- Service role can do everything
CREATE POLICY "Service role full access mod actions"
  ON moderation_actions FOR ALL
  USING (auth.role() = 'service_role');

-- Updated_at trigger for content_reports
CREATE OR REPLACE FUNCTION update_content_reports_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_content_reports_updated_at ON content_reports;
CREATE TRIGGER trigger_content_reports_updated_at
  BEFORE UPDATE ON content_reports
  FOR EACH ROW
  EXECUTE FUNCTION update_content_reports_updated_at();
