"""Tests for rate limiter."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from shared.rate_limiter import RateLimiter


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis_mock = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.incrby = AsyncMock(return_value=1)
    redis_mock.expire = AsyncMock(return_value=True)
    redis_mock.delete = AsyncMock(return_value=True)
    return redis_mock


@pytest.fixture
def rate_limiter(mock_redis):
    """Create a rate limiter with mock Redis."""
    limiter = RateLimiter(redis_client=mock_redis)
    return limiter


class TestRateLimiter:
    """Test RateLimiter class."""

    @pytest.mark.asyncio
    async def test_get_key_format(self, rate_limiter):
        """Test Redis key format."""
        user_id = "user-123"
        period = "2026-02"
        key = rate_limiter._get_key(user_id, period)
        assert key == "rate_limit:traces:user-123:2026-02"

    @pytest.mark.asyncio
    async def test_get_current_period(self, rate_limiter):
        """Test current period format."""
        period = rate_limiter._get_current_period()
        # Should be YYYY-MM format
        assert len(period) == 7
        assert period[4] == "-"
        # Should be current year and month
        now = datetime.now()
        expected = now.strftime("%Y-%m")
        assert period == expected

    @pytest.mark.asyncio
    async def test_check_limit_enterprise_unlimited(self, rate_limiter, mock_redis):
        """Test that enterprise tier has no limit."""
        allowed, usage, limit = await rate_limiter.check_limit(
            user_id="user-123",
            tier="enterprise",
            traces_count=1_000_000,
        )
        assert allowed is True
        assert usage == 0
        assert limit is None
        # Should not call Redis for enterprise
        mock_redis.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_limit_free_tier_within_limit(self, rate_limiter, mock_redis):
        """Test free tier within limit."""
        # Mock Redis to return current usage of 1000
        mock_redis.get = AsyncMock(return_value="1000")

        allowed, usage, limit = await rate_limiter.check_limit(
            user_id="user-123",
            tier="free",
            traces_count=1,
        )
        assert allowed is True
        assert usage == 1000
        assert limit == 50_000

    @pytest.mark.asyncio
    async def test_check_limit_free_tier_exceeded(self, rate_limiter, mock_redis):
        """Test free tier exceeds limit."""
        # Mock Redis to return current usage at limit
        mock_redis.get = AsyncMock(return_value="50000")

        allowed, usage, limit = await rate_limiter.check_limit(
            user_id="user-123",
            tier="free",
            traces_count=1,
        )
        assert allowed is False
        assert usage == 50_000
        assert limit == 50_000

    @pytest.mark.asyncio
    async def test_check_limit_lunch_money_tier(self, rate_limiter, mock_redis):
        """Test lunch-money tier limits."""
        mock_redis.get = AsyncMock(return_value="50000")

        allowed, usage, limit = await rate_limiter.check_limit(
            user_id="user-123",
            tier="lunch-money",
            traces_count=1,
        )
        assert allowed is True
        assert usage == 50_000
        assert limit == 100_000

    @pytest.mark.asyncio
    async def test_check_limit_pro_tier(self, rate_limiter, mock_redis):
        """Test pro tier limits."""
        mock_redis.get = AsyncMock(return_value="500000")

        allowed, usage, limit = await rate_limiter.check_limit(
            user_id="user-123",
            tier="pro",
            traces_count=1,
        )
        assert allowed is True
        assert usage == 500_000
        assert limit == 1_000_000

    @pytest.mark.asyncio
    async def test_check_limit_unknown_tier_defaults_to_free(self, rate_limiter, mock_redis):
        """Test unknown tier defaults to free tier limit."""
        mock_redis.get = AsyncMock(return_value="0")

        allowed, usage, limit = await rate_limiter.check_limit(
            user_id="user-123",
            tier="unknown-tier",
            traces_count=1,
        )
        assert allowed is True
        assert limit == 50_000

    @pytest.mark.asyncio
    async def test_check_limit_batch_traces(self, rate_limiter, mock_redis):
        """Test checking limit for batch of traces."""
        mock_redis.get = AsyncMock(return_value="49000")

        # Adding 2000 traces would exceed free tier limit
        allowed, usage, limit = await rate_limiter.check_limit(
            user_id="user-123",
            tier="free",
            traces_count=2000,
        )
        assert allowed is False
        assert usage == 49_000
        assert limit == 50_000

    @pytest.mark.asyncio
    async def test_check_limit_error_handling(self, rate_limiter, mock_redis):
        """Test error handling fails open."""
        # Simulate Redis error
        mock_redis.get = AsyncMock(side_effect=Exception("Redis error"))

        allowed, usage, limit = await rate_limiter.check_limit(
            user_id="user-123",
            tier="free",
            traces_count=1,
        )
        # Should fail open (allow the request)
        assert allowed is True
        assert usage == 0
        assert limit == 50_000

    @pytest.mark.asyncio
    async def test_increment(self, rate_limiter, mock_redis):
        """Test incrementing trace count."""
        mock_redis.incrby = AsyncMock(return_value=1001)

        new_count = await rate_limiter.increment(
            user_id="user-123",
            traces_count=1,
        )
        assert new_count == 1001
        mock_redis.incrby.assert_called_once()
        mock_redis.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_increment_batch(self, rate_limiter, mock_redis):
        """Test incrementing batch of traces."""
        mock_redis.incrby = AsyncMock(return_value=1100)

        new_count = await rate_limiter.increment(
            user_id="user-123",
            traces_count=100,
        )
        assert new_count == 1100

    @pytest.mark.asyncio
    async def test_increment_error_handling(self, rate_limiter, mock_redis):
        """Test increment error handling."""
        mock_redis.incrby = AsyncMock(side_effect=Exception("Redis error"))

        new_count = await rate_limiter.increment(
            user_id="user-123",
            traces_count=1,
        )
        # Should return 0 on error
        assert new_count == 0

    @pytest.mark.asyncio
    async def test_get_usage(self, rate_limiter, mock_redis):
        """Test getting current usage."""
        mock_redis.get = AsyncMock(return_value="5000")

        usage = await rate_limiter.get_usage(user_id="user-123")
        assert usage == 5000

    @pytest.mark.asyncio
    async def test_get_usage_no_usage(self, rate_limiter, mock_redis):
        """Test getting usage when none exists."""
        mock_redis.get = AsyncMock(return_value=None)

        usage = await rate_limiter.get_usage(user_id="user-123")
        assert usage == 0

    @pytest.mark.asyncio
    async def test_reset_usage(self, rate_limiter, mock_redis):
        """Test resetting usage."""
        await rate_limiter.reset_usage(user_id="user-123")
        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_usage_specific_period(self, rate_limiter, mock_redis):
        """Test resetting usage for specific period."""
        await rate_limiter.reset_usage(user_id="user-123", period="2025-12")
        mock_redis.delete.assert_called_once()
        # Verify the key includes the specified period
        call_args = mock_redis.delete.call_args[0]
        assert "2025-12" in call_args[0]

    @pytest.mark.asyncio
    async def test_tier_limits_configuration(self, rate_limiter):
        """Test tier limits are correctly configured."""
        assert RateLimiter.TIER_LIMITS["free"] == 50_000
        assert RateLimiter.TIER_LIMITS["lunch-money"] == 100_000
        assert RateLimiter.TIER_LIMITS["pro"] == 1_000_000
        assert RateLimiter.TIER_LIMITS["enterprise"] is None
