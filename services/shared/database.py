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
        100000,  # 100k traces/month for free tier
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
) -> dict[str, Any]:
    """Update subscription tier and Stripe IDs.

    Args:
        user_id: User UUID.
        tier: New tier (free, lunch-money, pro, enterprise).
        stripe_customer_id: Stripe customer ID.
        stripe_subscription_id: Stripe subscription ID.
        status: Subscription status.

    Returns:
        Updated subscription record.
    """
    from datetime import datetime, timedelta

    # Calculate trace limit based on tier
    tier_limits = {
        "free": 100_000,
        "lunch-money": 100_000,
        "pro": 1_000_000,
        "enterprise": None,  # Unlimited
    }
    trace_limit = tier_limits.get(tier, 100_000)

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
