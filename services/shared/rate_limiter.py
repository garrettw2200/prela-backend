"""Redis-based rate limiter for tier-based trace ingestion limits."""

import logging
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as redis

from .config import settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """Redis-based rate limiter for enforcing monthly trace limits by tier.

    Rate limits by tier:
    - free: 50k traces/month
    - lunch-money: 100k traces/month
    - pro: 1M traces/month
    - enterprise: unlimited
    """

    TIER_LIMITS = {
        "free": 50_000,
        "lunch-money": 100_000,
        "pro": 1_000_000,
        "enterprise": None,  # Unlimited
    }

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """Initialize rate limiter.

        Args:
            redis_client: Redis client instance. If None, creates a new one.
        """
        self._redis = redis_client
        self._redis_url = settings.redis_url

    async def get_redis(self) -> redis.Redis:
        """Get or create Redis client.

        Returns:
            Redis client instance.
        """
        if self._redis is None:
            self._redis = await redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            logger.info("Redis client created for rate limiter")
        return self._redis

    async def close(self):
        """Close Redis connection."""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None
            logger.info("Redis connection closed")

    def _get_key(self, user_id: str, period: str) -> str:
        """Generate Redis key for rate limiting.

        Args:
            user_id: User UUID.
            period: Period string (e.g., "2026-02" for February 2026).

        Returns:
            Redis key string.
        """
        return f"rate_limit:traces:{user_id}:{period}"

    def _get_current_period(self) -> str:
        """Get current period string (YYYY-MM format).

        Returns:
            Period string (e.g., "2026-02").
        """
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m")

    async def check_limit(
        self,
        user_id: str,
        tier: str,
        traces_count: int = 1,
    ) -> tuple[bool, int, int]:
        """Check if user is within rate limit for current period.

        Args:
            user_id: User UUID.
            tier: Subscription tier (free, lunch-money, pro, enterprise).
            traces_count: Number of traces to add (default: 1).

        Returns:
            Tuple of (allowed, current_usage, limit):
                - allowed: True if within limit, False if exceeded
                - current_usage: Current trace count for this period
                - limit: Maximum traces allowed for this tier (None = unlimited)
        """
        # Enterprise tier has no limit
        if tier == "enterprise":
            return True, 0, None

        # Get tier limit
        limit = self.TIER_LIMITS.get(tier)
        if limit is None:
            logger.warning(f"Unknown tier: {tier}, defaulting to free tier limit")
            limit = self.TIER_LIMITS["free"]

        # Get Redis client
        r = await self.get_redis()

        # Get current period
        period = self._get_current_period()
        key = self._get_key(user_id, period)

        try:
            # Get current usage
            current_usage = await r.get(key)
            current_usage = int(current_usage) if current_usage else 0

            # Check if adding traces would exceed limit
            new_usage = current_usage + traces_count

            if new_usage > limit:
                logger.warning(
                    f"Rate limit exceeded for user {user_id} (tier: {tier}): "
                    f"{new_usage}/{limit} traces"
                )
                return False, current_usage, limit

            return True, current_usage, limit

        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            # On error, allow the request (fail open)
            return True, 0, limit

    async def increment(
        self,
        user_id: str,
        traces_count: int = 1,
    ) -> int:
        """Increment trace count for user in current period.

        Args:
            user_id: User UUID.
            traces_count: Number of traces to add (default: 1).

        Returns:
            New trace count for this period.
        """
        r = await self.get_redis()
        period = self._get_current_period()
        key = self._get_key(user_id, period)

        try:
            # Increment counter
            new_count = await r.incrby(key, traces_count)

            # Set expiration to end of next month (to handle edge cases)
            # This ensures the key is automatically cleaned up
            await r.expire(key, 60 * 60 * 24 * 60)  # 60 days

            logger.debug(f"Incremented traces for user {user_id}: {new_count}")
            return new_count

        except Exception as e:
            logger.error(f"Error incrementing rate limit: {e}")
            return 0

    async def get_usage(self, user_id: str) -> int:
        """Get current trace count for user in current period.

        Args:
            user_id: User UUID.

        Returns:
            Current trace count.
        """
        r = await self.get_redis()
        period = self._get_current_period()
        key = self._get_key(user_id, period)

        try:
            current_usage = await r.get(key)
            return int(current_usage) if current_usage else 0
        except Exception as e:
            logger.error(f"Error getting usage: {e}")
            return 0

    async def reset_usage(self, user_id: str, period: Optional[str] = None):
        """Reset trace count for user (admin function).

        Args:
            user_id: User UUID.
            period: Period to reset (default: current period).
        """
        r = await self.get_redis()
        period = period or self._get_current_period()
        key = self._get_key(user_id, period)

        try:
            await r.delete(key)
            logger.info(f"Reset usage for user {user_id} in period {period}")
        except Exception as e:
            logger.error(f"Error resetting usage: {e}")


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


async def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance.

    Returns:
        RateLimiter instance.
    """
    global _rate_limiter

    if _rate_limiter is None:
        _rate_limiter = RateLimiter()

    return _rate_limiter


async def close_rate_limiter():
    """Close global rate limiter instance."""
    global _rate_limiter

    if _rate_limiter is not None:
        await _rate_limiter.close()
        _rate_limiter = None
