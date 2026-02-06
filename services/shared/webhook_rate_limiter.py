"""
Webhook-specific rate limiting.

Provides rate limiting for webhook endpoints to prevent abuse.
"""

import logging
from datetime import datetime, timezone
from fastapi import HTTPException
from .redis import get_redis_client

logger = logging.getLogger(__name__)


class WebhookRateLimiter:
    """Rate limiter for webhook endpoints."""

    def __init__(self, limit_per_minute: int = 100):
        """Initialize webhook rate limiter.

        Args:
            limit_per_minute: Maximum requests per minute per API key
        """
        self.limit_per_minute = limit_per_minute

    async def check_rate_limit(self, api_key: str, endpoint: str = "webhook") -> None:
        """Check if API key has exceeded webhook rate limit.

        Args:
            api_key: API key making the request
            endpoint: Endpoint name for tracking

        Raises:
            HTTPException: 429 if rate limit exceeded
        """
        try:
            redis_client = await get_redis_client()

            # Create rate limit key based on current minute
            now = datetime.now(timezone.utc)
            minute_key = now.strftime("%Y-%m-%d %H:%M")
            key = f"webhook_ratelimit:{endpoint}:{api_key}:{minute_key}"

            # Increment counter
            count = await redis_client.incr(key)

            # Set expiration on first increment
            if count == 1:
                await redis_client.expire(key, 60)  # Expire after 1 minute

            # Check limit
            if count > self.limit_per_minute:
                logger.warning(
                    f"Webhook rate limit exceeded: {api_key} "
                    f"({count}/{self.limit_per_minute} requests/minute)"
                )
                raise HTTPException(
                    status_code=429,
                    detail=f"Webhook rate limit exceeded. Maximum {self.limit_per_minute} requests per minute.",
                    headers={
                        "X-RateLimit-Limit": str(self.limit_per_minute),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int((now.replace(second=0, microsecond=0).timestamp()) + 60)),
                    }
                )

        except HTTPException:
            raise
        except Exception as e:
            # Log error but don't block request if rate limiter fails
            logger.error(f"Webhook rate limiter error: {e}")
            # Fail open - allow request to proceed


# Global rate limiter instance
_webhook_rate_limiter = None


def get_webhook_rate_limiter(limit_per_minute: int = 100) -> WebhookRateLimiter:
    """Get or create global webhook rate limiter instance.

    Args:
        limit_per_minute: Maximum requests per minute

    Returns:
        WebhookRateLimiter instance
    """
    global _webhook_rate_limiter
    if _webhook_rate_limiter is None:
        _webhook_rate_limiter = WebhookRateLimiter(limit_per_minute=limit_per_minute)
    return _webhook_rate_limiter
