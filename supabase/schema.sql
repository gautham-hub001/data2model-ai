-- Run this in the Supabase SQL editor at https://app.supabase.com

CREATE TABLE IF NOT EXISTS sessions (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       TEXT        NOT NULL,          -- Clerk user ID (sub claim)
  dataset_name  TEXT        NOT NULL,
  dataset_id    TEXT,                          -- reference to Supabase Storage object
  analysis_result JSONB,
  recommendation  JSONB,
  generated_code  TEXT,
  smote_applied   BOOLEAN DEFAULT FALSE,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for fast lookup by user
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);

-- Row-level security: users can only see their own sessions
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users see own sessions"
  ON sessions FOR SELECT
  USING (user_id = current_setting('request.jwt.claims', true)::jsonb->>'sub');

-- Storage bucket for CSV uploads (create manually in Supabase dashboard or via CLI)
-- Bucket name: datasets
-- Access: private (signed URLs only)
