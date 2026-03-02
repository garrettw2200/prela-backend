-- Migration 008: Add team_id to api_keys for team-scoped keys
-- Date: 2026-03-01
-- Description: Allows API keys to be owned by a team rather than just a user.
--   user_id remains the creator; team_id marks the key as team-visible.
--   NULL team_id = personal key (visible only to the creating user).
--   Non-NULL team_id = team key (visible to all team members, manageable by admins/owners).

ALTER TABLE api_keys
ADD COLUMN IF NOT EXISTS team_id UUID REFERENCES teams(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_api_keys_team ON api_keys(team_id);

COMMENT ON COLUMN api_keys.team_id IS 'If set, key is visible to all team members. NULL means personal/user key.';
