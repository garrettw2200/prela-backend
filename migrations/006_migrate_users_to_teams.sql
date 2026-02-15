-- Migration: Auto-create personal teams for existing users
-- Date: 2026-02-14
-- Description: Ensures every existing user has a personal team. Idempotent.

-- Create personal teams for users who don't have one yet
INSERT INTO teams (name, slug, owner_id)
SELECT
    u.email || '''s Team',
    'personal-' || u.id::TEXT,
    u.id
FROM users u
WHERE NOT EXISTS (
    SELECT 1 FROM teams t
    WHERE t.owner_id = u.id
    AND t.slug = 'personal-' || u.id::TEXT
);

-- Add owners as team_members with 'owner' role for teams where they aren't yet a member
INSERT INTO team_members (team_id, user_id, role)
SELECT
    t.id,
    t.owner_id,
    'owner'
FROM teams t
WHERE t.slug LIKE 'personal-%'
AND NOT EXISTS (
    SELECT 1 FROM team_members tm
    WHERE tm.team_id = t.id
    AND tm.user_id = t.owner_id
);
