"""PostgreSQL database client utilities."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import asyncpg
from asyncpg import Pool

from .config import settings

logger = logging.getLogger(__name__)

# Global connection pool
_pool: Pool | None = None


async def get_database_pool() -> Pool:
    """Get or create PostgreSQL connection pool.

    Returns:
        AsyncPG connection pool.

    Raises:
        Exception: If connection fails.
    """
    global _pool

    if _pool is None:
        try:
            _pool = await asyncpg.create_pool(
                settings.database_url,
                min_size=2,
                max_size=10,
                command_timeout=60,
            )
            logger.info("PostgreSQL connection pool created")
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise

    return _pool


async def close_database_pool():
    """Close PostgreSQL connection pool."""
    global _pool

    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL connection pool closed")


@asynccontextmanager
async def get_db_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """Get a database connection from the pool.

    Usage:
        async with get_db_connection() as conn:
            result = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
    """
    pool = await get_database_pool()
    async with pool.acquire() as connection:
        yield connection


async def fetch_one(query: str, *args) -> dict[str, Any] | None:
    """Execute query and return a single row.

    Args:
        query: SQL query with $1, $2, etc. placeholders.
        *args: Query parameters.

    Returns:
        Dictionary of column names to values, or None if no rows.
    """
    async with get_db_connection() as conn:
        row = await conn.fetchrow(query, *args)
        return dict(row) if row else None


async def fetch_all(query: str, *args) -> list[dict[str, Any]]:
    """Execute query and return all rows.

    Args:
        query: SQL query with $1, $2, etc. placeholders.
        *args: Query parameters.

    Returns:
        List of dictionaries (column names to values).
    """
    async with get_db_connection() as conn:
        rows = await conn.fetch(query, *args)
        return [dict(row) for row in rows]


async def execute(query: str, *args) -> str:
    """Execute a query (INSERT, UPDATE, DELETE).

    Args:
        query: SQL query with $1, $2, etc. placeholders.
        *args: Query parameters.

    Returns:
        Status string (e.g., "INSERT 0 1").
    """
    async with get_db_connection() as conn:
        return await conn.execute(query, *args)


async def insert_returning(query: str, *args) -> dict[str, Any]:
    """Execute INSERT/UPDATE with RETURNING clause.

    Args:
        query: SQL query with RETURNING clause.
        *args: Query parameters.

    Returns:
        Dictionary of returned row.
    """
    async with get_db_connection() as conn:
        row = await conn.fetchrow(query, *args)
        return dict(row) if row else {}


# User and subscription helper functions


async def get_user_by_clerk_id(clerk_id: str) -> dict[str, Any] | None:
    """Get user by Clerk ID.

    Args:
        clerk_id: Clerk user ID.

    Returns:
        User record or None.
    """
    return await fetch_one(
        "SELECT * FROM users WHERE clerk_id = $1",
        clerk_id,
    )


async def create_user(clerk_id: str, email: str, full_name: str | None = None, profile_image_url: str | None = None) -> dict[str, Any]:
    """Create a new user from Clerk data.

    Args:
        clerk_id: Clerk user ID.
        email: User email address.
        full_name: User's full name (optional).
        profile_image_url: Profile image URL (optional).

    Returns:
        Created user record.
    """
    return await insert_returning(
        """
        INSERT INTO users (clerk_id, email, full_name, profile_image_url)
        VALUES ($1, $2, $3, $4)
        RETURNING *
        """,
        clerk_id,
        email,
        full_name,
        profile_image_url,
    )


async def get_subscription_by_user_id(user_id: str) -> dict[str, Any] | None:
    """Get active subscription for a user.

    Args:
        user_id: User UUID.

    Returns:
        Subscription record or None.
    """
    return await fetch_one(
        """
        SELECT * FROM subscriptions
        WHERE user_id = $1
        ORDER BY created_at DESC
        LIMIT 1
        """,
        user_id,
    )


async def create_free_subscription(user_id: str) -> dict[str, Any]:
    """Create a free tier subscription for a new user.

    Args:
        user_id: User UUID.

    Returns:
        Created subscription record.
    """
    from datetime import datetime, timedelta

    period_start = datetime.utcnow()
    period_end = period_start + timedelta(days=30)

    return await insert_returning(
        """
        INSERT INTO subscriptions (
            user_id, tier, status, trace_limit,
            current_period_start, current_period_end
        )
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING *
        """,
        user_id,
        "free",
        "active",
        50000,  # 50k traces/month for free tier
        period_start,
        period_end,
    )


async def update_subscription_tier(subscription_id: str, tier: str, trace_limit: int) -> dict[str, Any]:
    """Update subscription tier and limits.

    Args:
        subscription_id: Subscription UUID.
        tier: New tier (free, lunch-money, pro, enterprise).
        trace_limit: New monthly trace limit.

    Returns:
        Updated subscription record.
    """
    return await insert_returning(
        """
        UPDATE subscriptions
        SET tier = $2, trace_limit = $3, updated_at = NOW()
        WHERE id = $1
        RETURNING *
        """,
        subscription_id,
        tier,
        trace_limit,
    )


async def increment_usage(subscription_id: str, traces: int = 0, spans: int = 0) -> None:
    """Increment usage counters for a subscription.

    Args:
        subscription_id: Subscription UUID.
        traces: Number of traces to add.
        spans: Number of spans to add.
    """
    await execute(
        """
        UPDATE subscriptions
        SET monthly_usage = monthly_usage + $2
        WHERE id = $1
        """,
        subscription_id,
        traces + spans,  # Total objects ingested
    )


async def reset_monthly_usage(subscription_id: str) -> None:
    """Reset monthly usage counter (run at period end).

    Args:
        subscription_id: Subscription UUID.
    """
    await execute(
        "UPDATE subscriptions SET monthly_usage = 0 WHERE id = $1",
        subscription_id,
    )


async def verify_api_key(key_hash: str) -> dict[str, Any] | None:
    """Verify API key and return user + subscription data.

    Args:
        key_hash: bcrypt hash of the API key.

    Returns:
        Dictionary with user, subscription, and api_key data, or None.
    """
    return await fetch_one(
        """
        SELECT
            u.id as user_id,
            u.clerk_id,
            u.email,
            s.tier,
            s.status as subscription_status,
            s.trace_limit,
            s.monthly_usage,
            ak.id as api_key_id,
            ak.name as api_key_name
        FROM api_keys ak
        JOIN users u ON ak.user_id = u.id
        JOIN subscriptions s ON s.user_id = u.id
        WHERE ak.key_hash = $1
        AND (ak.expires_at IS NULL OR ak.expires_at > NOW())
        ORDER BY s.created_at DESC
        LIMIT 1
        """,
        key_hash,
    )


async def update_api_key_last_used(api_key_id: str) -> None:
    """Update last_used_at timestamp for an API key.

    Args:
        api_key_id: API key UUID.
    """
    await execute(
        "UPDATE api_keys SET last_used_at = NOW() WHERE id = $1",
        api_key_id,
    )


async def update_subscription_tier(
    user_id: str,
    tier: str,
    stripe_customer_id: str | None = None,
    stripe_subscription_id: str | None = None,
    status: str = "active",
    team_id: str | None = None,
) -> dict[str, Any]:
    """Update subscription tier and Stripe IDs.

    Args:
        user_id: User UUID.
        tier: New tier (free, lunch-money, pro, enterprise).
        stripe_customer_id: Stripe customer ID.
        stripe_subscription_id: Stripe subscription ID.
        status: Subscription status.
        team_id: Team UUID to link subscription to (optional).

    Returns:
        Updated subscription record.
    """
    from datetime import datetime, timedelta

    # Calculate trace limit based on tier
    tier_limits = {
        "free": 50_000,
        "lunch-money": 100_000,
        "pro": 1_000_000,
        "enterprise": None,  # Unlimited
    }
    trace_limit = tier_limits.get(tier, 50_000)

    period_start = datetime.utcnow()
    period_end = period_start + timedelta(days=30)

    return await insert_returning(
        """
        UPDATE subscriptions
        SET tier = $2,
            status = $3,
            trace_limit = $4,
            stripe_customer_id = $5,
            stripe_subscription_id = $6,
            current_period_start = $7,
            current_period_end = $8,
            team_id = COALESCE($9, team_id),
            updated_at = NOW()
        WHERE user_id = $1
        RETURNING *
        """,
        user_id,
        tier,
        status,
        trace_limit,
        stripe_customer_id,
        stripe_subscription_id,
        period_start,
        period_end,
        team_id,
    )


async def update_subscription_status(
    stripe_subscription_id: str,
    status: str,
) -> dict[str, Any]:
    """Update subscription status by Stripe subscription ID.

    Args:
        stripe_subscription_id: Stripe subscription ID.
        status: New status (active, canceled, past_due, trialing).

    Returns:
        Updated subscription record.
    """
    return await insert_returning(
        """
        UPDATE subscriptions
        SET status = $2, updated_at = NOW()
        WHERE stripe_subscription_id = $1
        RETURNING *
        """,
        stripe_subscription_id,
        status,
    )


async def create_api_key(
    user_id: str,
    key_hash: str,
    key_prefix: str,
    name: str,
) -> dict[str, Any]:
    """Create a new API key.

    Args:
        user_id: User UUID.
        key_hash: SHA256 hash of the API key.
        key_prefix: First 16 characters for display.
        name: Name/description for the key.

    Returns:
        Created API key record.
    """
    return await insert_returning(
        """
        INSERT INTO api_keys (user_id, key_hash, key_prefix, name)
        VALUES ($1, $2, $3, $4)
        RETURNING *
        """,
        user_id,
        key_hash,
        key_prefix,
        name,
    )


async def get_api_keys_by_user_id(user_id: str) -> list[dict[str, Any]]:
    """Get all API keys for a user.

    Args:
        user_id: User UUID.

    Returns:
        List of API key records.
    """
    return await fetch_all(
        """
        SELECT id, key_prefix, name, last_used_at, created_at
        FROM api_keys
        WHERE user_id = $1
        ORDER BY created_at DESC
        """,
        user_id,
    )


async def delete_api_key(api_key_id: str, user_id: str) -> None:
    """Delete an API key.

    Args:
        api_key_id: API key UUID.
        user_id: User UUID (for authorization check).
    """
    await execute(
        "DELETE FROM api_keys WHERE id = $1 AND user_id = $2",
        api_key_id,
        user_id,
    )


# Data source helper functions


async def create_data_source(
    user_id: str,
    project_id: str,
    source_type: str,
    name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Create a new data source connection.

    Args:
        user_id: User UUID.
        project_id: Prela project ID.
        source_type: Source type (e.g. 'langfuse').
        name: Display name for the connection.
        config: Connection config (host, public_key, encrypted_secret_key).

    Returns:
        Created data source record.
    """
    import json

    return await insert_returning(
        """
        INSERT INTO data_sources (user_id, project_id, type, name, config)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING *
        """,
        user_id,
        project_id,
        source_type,
        name,
        json.dumps(config),
    )


async def get_data_sources_by_user_id(user_id: str) -> list[dict[str, Any]]:
    """Get all data sources for a user.

    Args:
        user_id: User UUID.

    Returns:
        List of data source records.
    """
    return await fetch_all(
        """
        SELECT * FROM data_sources
        WHERE user_id = $1
        ORDER BY created_at DESC
        """,
        user_id,
    )


async def get_data_source_by_id(
    source_id: str, user_id: str
) -> dict[str, Any] | None:
    """Get a specific data source with ownership check.

    Args:
        source_id: Data source UUID.
        user_id: User UUID (for authorization).

    Returns:
        Data source record, or None if not found / not owned.
    """
    return await fetch_one(
        "SELECT * FROM data_sources WHERE id = $1 AND user_id = $2",
        source_id,
        user_id,
    )


async def get_active_data_sources() -> list[dict[str, Any]]:
    """Get all active data sources for background sync.

    Returns:
        List of active data source records.
    """
    return await fetch_all(
        "SELECT * FROM data_sources WHERE status = 'active' ORDER BY last_sync_at ASC NULLS FIRST"
    )


async def update_data_source_status(
    source_id: str,
    status: str,
    error_message: str | None = None,
) -> None:
    """Update the status and error message of a data source.

    Args:
        source_id: Data source UUID.
        status: New status ('active', 'error', 'paused').
        error_message: Error description (cleared if None).
    """
    await execute(
        """
        UPDATE data_sources
        SET status = $2, error_message = $3, updated_at = NOW()
        WHERE id = $1
        """,
        source_id,
        status,
        error_message,
    )


async def update_data_source_last_sync(
    source_id: str,
    last_sync_at: Any,
    config: dict[str, Any],
) -> None:
    """Update last_sync_at and config after a successful sync.

    Args:
        source_id: Data source UUID.
        last_sync_at: Timestamp of sync completion.
        config: Updated config (with new last_synced_timestamp).
    """
    import json

    await execute(
        """
        UPDATE data_sources
        SET last_sync_at = $2, config = $3, status = 'active', error_message = NULL, updated_at = NOW()
        WHERE id = $1
        """,
        source_id,
        last_sync_at,
        json.dumps(config),
    )


async def delete_data_source(source_id: str, user_id: str) -> None:
    """Delete a data source.

    Args:
        source_id: Data source UUID.
        user_id: User UUID (for authorization check).
    """
    await execute(
        "DELETE FROM data_sources WHERE id = $1 AND user_id = $2",
        source_id,
        user_id,
    )


# Team collaboration helper functions


async def create_team(name: str, slug: str, owner_id: str) -> dict[str, Any]:
    """Create a new team and add the owner as a member.

    Args:
        name: Team display name.
        slug: URL-safe unique identifier.
        owner_id: User UUID of the team owner.

    Returns:
        Created team record.
    """
    async with get_db_connection() as conn:
        async with conn.transaction():
            team = await conn.fetchrow(
                """
                INSERT INTO teams (name, slug, owner_id)
                VALUES ($1, $2, $3)
                RETURNING *
                """,
                name, slug, owner_id,
            )
            team = dict(team)
            await conn.execute(
                """
                INSERT INTO team_members (team_id, user_id, role)
                VALUES ($1, $2, 'owner')
                """,
                team["id"], owner_id,
            )
            return team


async def get_teams_for_user(user_id: str) -> list[dict[str, Any]]:
    """Get all teams a user belongs to.

    Args:
        user_id: User UUID.

    Returns:
        List of team records with the user's role.
    """
    return await fetch_all(
        """
        SELECT t.*, tm.role as user_role
        FROM teams t
        JOIN team_members tm ON t.id = tm.team_id
        WHERE tm.user_id = $1
        ORDER BY t.created_at DESC
        """,
        user_id,
    )


async def get_team_by_id(team_id: str) -> dict[str, Any] | None:
    """Get team by ID.

    Args:
        team_id: Team UUID.

    Returns:
        Team record or None.
    """
    return await fetch_one(
        "SELECT * FROM teams WHERE id = $1",
        team_id,
    )


async def update_team(team_id: str, name: str) -> dict[str, Any]:
    """Update team name.

    Args:
        team_id: Team UUID.
        name: New team name.

    Returns:
        Updated team record.
    """
    return await insert_returning(
        "UPDATE teams SET name = $2, updated_at = NOW() WHERE id = $1 RETURNING *",
        team_id, name,
    )


async def delete_team(team_id: str) -> None:
    """Delete a team (cascades to members, invitations, project assignments).

    Args:
        team_id: Team UUID.
    """
    await execute("DELETE FROM teams WHERE id = $1", team_id)


# Team member helper functions


async def get_team_members(team_id: str) -> list[dict[str, Any]]:
    """Get all members of a team with user details.

    Args:
        team_id: Team UUID.

    Returns:
        List of member records with user info.
    """
    return await fetch_all(
        """
        SELECT tm.id, tm.team_id, tm.user_id, tm.role, tm.joined_at,
               u.email, u.full_name, u.profile_image_url
        FROM team_members tm
        JOIN users u ON tm.user_id = u.id
        WHERE tm.team_id = $1
        ORDER BY tm.joined_at ASC
        """,
        team_id,
    )


async def add_team_member(
    team_id: str, user_id: str, role: str = "member", invited_by: str | None = None
) -> dict[str, Any]:
    """Add a user to a team.

    Args:
        team_id: Team UUID.
        user_id: User UUID to add.
        role: Role to assign (owner, admin, member, viewer).
        invited_by: User UUID who invited this member.

    Returns:
        Created team_members record.
    """
    return await insert_returning(
        """
        INSERT INTO team_members (team_id, user_id, role, invited_by)
        VALUES ($1, $2, $3, $4)
        RETURNING *
        """,
        team_id, user_id, role, invited_by,
    )


async def update_team_member_role(team_id: str, user_id: str, role: str) -> dict[str, Any]:
    """Update a team member's role.

    Args:
        team_id: Team UUID.
        user_id: User UUID.
        role: New role (owner, admin, member, viewer).

    Returns:
        Updated team_members record.
    """
    return await insert_returning(
        """
        UPDATE team_members SET role = $3
        WHERE team_id = $1 AND user_id = $2
        RETURNING *
        """,
        team_id, user_id, role,
    )


async def remove_team_member(team_id: str, user_id: str) -> None:
    """Remove a user from a team.

    Args:
        team_id: Team UUID.
        user_id: User UUID to remove.
    """
    await execute(
        "DELETE FROM team_members WHERE team_id = $1 AND user_id = $2",
        team_id, user_id,
    )


async def get_team_member_count(team_id: str) -> int:
    """Get the number of members in a team.

    Args:
        team_id: Team UUID.

    Returns:
        Member count.
    """
    result = await fetch_one(
        "SELECT COUNT(*) as count FROM team_members WHERE team_id = $1",
        team_id,
    )
    return result["count"] if result else 0


# Team invitation helper functions


async def create_team_invitation(
    team_id: str, email: str, role: str, invited_by: str, token: str, expires_at: Any
) -> dict[str, Any]:
    """Create a team invitation.

    Args:
        team_id: Team UUID.
        email: Invitee's email address.
        role: Role to assign on acceptance.
        invited_by: User UUID of inviter.
        token: Unique invitation token.
        expires_at: Expiration timestamp.

    Returns:
        Created invitation record.
    """
    return await insert_returning(
        """
        INSERT INTO team_invitations (team_id, email, role, invited_by, token, expires_at)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING *
        """,
        team_id, email, role, invited_by, token, expires_at,
    )


async def get_invitation_by_token(token: str) -> dict[str, Any] | None:
    """Get invitation by unique token.

    Args:
        token: Invitation token.

    Returns:
        Invitation record with team name, or None.
    """
    return await fetch_one(
        """
        SELECT ti.*, t.name as team_name
        FROM team_invitations ti
        JOIN teams t ON ti.team_id = t.id
        WHERE ti.token = $1
        """,
        token,
    )


async def accept_invitation(invitation_id: str) -> dict[str, Any]:
    """Mark an invitation as accepted.

    Args:
        invitation_id: Invitation UUID.

    Returns:
        Updated invitation record.
    """
    return await insert_returning(
        """
        UPDATE team_invitations
        SET status = 'accepted', accepted_at = NOW(), updated_at = NOW()
        WHERE id = $1
        RETURNING *
        """,
        invitation_id,
    )


async def get_pending_invitations(team_id: str) -> list[dict[str, Any]]:
    """Get all pending invitations for a team.

    Args:
        team_id: Team UUID.

    Returns:
        List of pending invitation records with inviter info.
    """
    return await fetch_all(
        """
        SELECT ti.*, u.email as invited_by_email, u.full_name as invited_by_name
        FROM team_invitations ti
        JOIN users u ON ti.invited_by = u.id
        WHERE ti.team_id = $1 AND ti.status = 'pending'
        ORDER BY ti.created_at DESC
        """,
        team_id,
    )


async def revoke_invitation(invitation_id: str) -> dict[str, Any]:
    """Revoke a pending invitation.

    Args:
        invitation_id: Invitation UUID.

    Returns:
        Updated invitation record.
    """
    return await insert_returning(
        """
        UPDATE team_invitations
        SET status = 'revoked', updated_at = NOW()
        WHERE id = $1
        RETURNING *
        """,
        invitation_id,
    )


# Project access helper functions


async def assign_project_to_team(project_id: str, team_id: str) -> dict[str, Any]:
    """Assign a ClickHouse project to a team.

    Args:
        project_id: ClickHouse project ID.
        team_id: Team UUID.

    Returns:
        Created project_teams record.
    """
    return await insert_returning(
        """
        INSERT INTO project_teams (project_id, team_id)
        VALUES ($1, $2)
        RETURNING *
        """,
        project_id, team_id,
    )


async def remove_project_from_team(project_id: str, team_id: str) -> None:
    """Remove a project assignment from a team.

    Args:
        project_id: ClickHouse project ID.
        team_id: Team UUID.
    """
    await execute(
        "DELETE FROM project_teams WHERE project_id = $1 AND team_id = $2",
        project_id, team_id,
    )


async def get_projects_for_team(team_id: str) -> list[dict[str, Any]]:
    """Get all project IDs assigned to a team.

    Args:
        team_id: Team UUID.

    Returns:
        List of project_teams records.
    """
    return await fetch_all(
        "SELECT * FROM project_teams WHERE team_id = $1 ORDER BY created_at DESC",
        team_id,
    )


async def check_project_access(user_id: str, project_id: str) -> dict[str, Any] | None:
    """Check if a user has access to a project via any team membership.

    Args:
        user_id: User UUID.
        project_id: ClickHouse project ID.

    Returns:
        Dict with team_id and role if access exists, or None.
    """
    return await fetch_one(
        """
        SELECT pt.team_id, tm.role
        FROM project_teams pt
        JOIN team_members tm ON pt.team_id = tm.team_id
        WHERE pt.project_id = $1 AND tm.user_id = $2
        LIMIT 1
        """,
        project_id, user_id,
    )


async def get_accessible_project_ids(user_id: str) -> list[str]:
    """Get all project IDs accessible to a user via team membership.

    Args:
        user_id: User UUID.

    Returns:
        List of project_id strings.
    """
    rows = await fetch_all(
        """
        SELECT DISTINCT pt.project_id
        FROM project_teams pt
        JOIN team_members tm ON pt.team_id = tm.team_id
        WHERE tm.user_id = $1
        """,
        user_id,
    )
    return [row["project_id"] for row in rows]


# Trace comment helper functions


async def create_trace_comment(
    trace_id: str, project_id: str, user_id: str, content: str,
    parent_comment_id: str | None = None,
) -> dict[str, Any]:
    """Create a comment on a trace.

    Args:
        trace_id: Trace ID (from ClickHouse).
        project_id: Project ID (from ClickHouse).
        user_id: Commenting user UUID.
        content: Comment text.
        parent_comment_id: Parent comment UUID for threaded replies.

    Returns:
        Created comment record.
    """
    return await insert_returning(
        """
        INSERT INTO trace_comments (trace_id, project_id, user_id, content, parent_comment_id)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING *
        """,
        trace_id, project_id, user_id, content, parent_comment_id,
    )


async def get_trace_comments(trace_id: str, project_id: str) -> list[dict[str, Any]]:
    """Get all comments for a trace with user details.

    Args:
        trace_id: Trace ID.
        project_id: Project ID.

    Returns:
        List of comment records with user info, ordered by creation time.
    """
    return await fetch_all(
        """
        SELECT tc.*, u.email, u.full_name, u.profile_image_url
        FROM trace_comments tc
        JOIN users u ON tc.user_id = u.id
        WHERE tc.trace_id = $1 AND tc.project_id = $2
        ORDER BY tc.created_at ASC
        """,
        trace_id, project_id,
    )


async def update_trace_comment(comment_id: str, user_id: str, content: str) -> dict[str, Any]:
    """Update a trace comment (only by the author).

    Args:
        comment_id: Comment UUID.
        user_id: User UUID (for ownership check).
        content: New comment text.

    Returns:
        Updated comment record.
    """
    return await insert_returning(
        """
        UPDATE trace_comments
        SET content = $3, updated_at = NOW()
        WHERE id = $1 AND user_id = $2
        RETURNING *
        """,
        comment_id, user_id, content,
    )


async def delete_trace_comment(comment_id: str, user_id: str) -> None:
    """Delete a trace comment (only by the author).

    Args:
        comment_id: Comment UUID.
        user_id: User UUID (for ownership check).
    """
    await execute(
        "DELETE FROM trace_comments WHERE id = $1 AND user_id = $2",
        comment_id, user_id,
    )
