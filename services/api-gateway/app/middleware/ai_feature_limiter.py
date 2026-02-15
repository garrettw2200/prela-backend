"""AI feature usage limiter for Pro tier."""

from datetime import datetime, timezone
from fastapi import HTTPException
import logging
import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Pro tier AI limits (from PRICING_STRATEGY_V2.md)
AI_LIMITS = {
    "hallucination": 10_000,  # per month
    "drift": 50,  # baselines per month
    "nlp": 1_000,  # searches per month
    "security": 10_000,  # scans per month
    "debug": 50,  # sessions per month (Pro tier)
    "eval_generation": 100,  # generations per month
}


class AIFeatureLimiter:
    """Enforce AI feature usage limits for Pro tier."""

    def __init__(self, redis_client: redis.Redis = None):
        """Initialize AI feature limiter.

        Args:
            redis_client: Redis client for usage tracking
        """
        self._redis = redis_client

    async def get_redis(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._redis is None:
            from shared.config import settings

            self._redis = await redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            logger.info("Redis client created for AI feature limiter")
        return self._redis

    def _get_current_period(self) -> str:
        """Get current period string (YYYY-MM format).

        Returns:
            Period string (e.g., "2026-02").
        """
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m")

    def _get_usage_key(self, user_id: str, feature: str) -> str:
        """Generate Redis key for AI feature usage tracking.

        Args:
            user_id: User UUID
            feature: Feature name (hallucination, drift, nlp)

        Returns:
            Redis key string
        """
        period = self._get_current_period()
        return f"ai_{feature}_usage:{user_id}:{period}"

    async def check_limit(
        self,
        user_id: str,
        subscription_tier: str,
        feature: str,
        increment: int = 1,
    ) -> tuple[bool, int, int]:
        """Check if user is within AI feature usage limits.

        Args:
            user_id: User UUID
            subscription_tier: Subscription tier (free, lunch-money, pro, enterprise)
            feature: Feature name (hallucination, drift, nlp)
            increment: Number of units to add (default: 1)

        Returns:
            Tuple of (allowed, current_usage, limit):
                - allowed: True if within limit, False if exceeded
                - current_usage: Current usage for this period
                - limit: Maximum allowed for this tier

        Raises:
            HTTPException: If user doesn't have access to this feature
        """
        # Only Pro and Enterprise tiers have AI features
        if subscription_tier not in ["pro", "enterprise"]:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"AI feature '{feature}' requires Pro or Enterprise tier. "
                    f"Current tier: '{subscription_tier}'. "
                    f"Upgrade at: https://prela.app/pricing"
                ),
            )

        # Enterprise tier has unlimited usage
        if subscription_tier == "enterprise":
            return True, 0, None

        # Get limit for this feature
        limit = AI_LIMITS.get(feature, 0)

        # Get Redis client
        r = await self.get_redis()
        usage_key = self._get_usage_key(user_id, feature)

        try:
            # Get current usage
            current_usage = await r.get(usage_key)
            current_usage = int(current_usage) if current_usage else 0

            # Check if adding would exceed limit
            new_usage = current_usage + increment

            # For Pro tier: Allow overages (will be billed)
            # But warn when approaching soft cap
            if new_usage > limit:
                logger.warning(
                    f"AI feature '{feature}' usage for user {user_id} "
                    f"exceeded base limit: {new_usage}/{limit}. "
                    f"Overages will be billed."
                )

            # Always allow for Pro tier (overages are billed)
            # Only reject if way over soft cap (5x limit)
            soft_cap = limit * 5
            if new_usage > soft_cap:
                logger.error(
                    f"AI feature '{feature}' usage for user {user_id} "
                    f"exceeded soft cap: {new_usage}/{soft_cap}. "
                    f"Consider upgrading to Enterprise."
                )
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"AI feature '{feature}' usage significantly exceeded. "
                        f"Current: {current_usage}/{limit} base limit. "
                        f"Heavy usage detected. Please contact sales@prela.app "
                        f"for Enterprise tier with custom limits."
                    ),
                )

            return True, current_usage, limit

        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Error checking AI feature limit: {e}")
            # On error, allow the request (fail open)
            return True, 0, limit

    async def increment(
        self,
        user_id: str,
        feature: str,
        increment: int = 1,
    ) -> int:
        """Increment AI feature usage counter.

        Args:
            user_id: User UUID
            feature: Feature name (hallucination, drift, nlp)
            increment: Number of units to add (default: 1)

        Returns:
            New usage count for this period
        """
        r = await self.get_redis()
        usage_key = self._get_usage_key(user_id, feature)

        try:
            # Increment counter
            new_count = await r.incrby(usage_key, increment)

            # Set expiration to end of next month (60 days)
            await r.expire(usage_key, 60 * 60 * 24 * 60)

            logger.debug(
                f"Incremented AI feature '{feature}' usage for user {user_id}: {new_count}"
            )
            return new_count

        except Exception as e:
            logger.error(f"Error incrementing AI feature usage: {e}")
            return 0

    async def get_usage(self, user_id: str, feature: str) -> int:
        """Get current AI feature usage for user in current period.

        Args:
            user_id: User UUID
            feature: Feature name (hallucination, drift, nlp)

        Returns:
            Current usage count
        """
        r = await self.get_redis()
        usage_key = self._get_usage_key(user_id, feature)

        try:
            current_usage = await r.get(usage_key)
            return int(current_usage) if current_usage else 0
        except Exception as e:
            logger.error(f"Error getting AI feature usage: {e}")
            return 0

    async def get_all_usage(self, user_id: str) -> dict[str, dict]:
        """Get all AI feature usage for user in current period.

        Args:
            user_id: User UUID

        Returns:
            Dictionary with usage and limit for all features
        """
        result = {}
        for feature, limit in AI_LIMITS.items():
            usage = await self.get_usage(user_id, feature)
            result[feature] = {"used": usage, "limit": limit}
        return result


# Global instance
_ai_feature_limiter: AIFeatureLimiter = None


async def get_ai_feature_limiter() -> AIFeatureLimiter:
    """Get global AI feature limiter instance.

    Returns:
        AIFeatureLimiter instance
    """
    global _ai_feature_limiter

    if _ai_feature_limiter is None:
        _ai_feature_limiter = AIFeatureLimiter()

    return _ai_feature_limiter


# Dependency for FastAPI endpoints
async def check_ai_feature_limit(
    user_id: str,
    subscription_tier: str,
    feature: str,
    increment: int = 1,
):
    """FastAPI dependency to check AI feature limits.

    Usage:
        @router.post("/ai/hallucination-check")
        async def check_hallucination(
            request: HallucinationCheckRequest,
            user_id: str = Depends(get_current_user),
            tier: str = Depends(get_user_tier),
        ):
            # Check limit before processing
            await check_ai_feature_limit(user_id, tier, "hallucination")

            # Process request...
            result = await perform_hallucination_check(request)

            return result

    Args:
        user_id: User UUID
        subscription_tier: Subscription tier
        feature: Feature name (hallucination, drift, nlp)
        increment: Number of units to use

    Raises:
        HTTPException: If user doesn't have access or exceeds soft cap
    """
    limiter = await get_ai_feature_limiter()

    # Check limit
    allowed, current, limit = await limiter.check_limit(
        user_id, subscription_tier, feature, increment
    )

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"AI feature usage limit exceeded: {current}/{limit}",
        )

    # Increment usage
    await limiter.increment(user_id, feature, increment)
