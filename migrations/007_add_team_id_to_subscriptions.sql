-- Migration: Link subscriptions to teams for per-team billing
-- Date: 2026-02-14

ALTER TABLE subscriptions
ADD COLUMN IF NOT EXISTS team_id UUID REFERENCES teams(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_subscriptions_team ON subscriptions(team_id);

-- Backfill: Set team_id to the user's personal team for existing subscriptions
UPDATE subscriptions s
SET team_id = t.id
FROM teams t
WHERE t.owner_id = s.user_id
  AND t.slug LIKE 'personal-%'
  AND s.team_id IS NULL;
