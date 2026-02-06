"""Tests for AI feature usage limits and enforcement."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from app.middleware.ai_feature_limiter import (
    AIFeatureLimiter,
    AI_LIMITS,
    check_ai_feature_limit,
)


class TestAIFeatureLimiter:
    """Test AI feature usage limiting."""

    @pytest.fixture
    def redis_mock(self):
        """Create a mock Redis client."""
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.incrby = AsyncMock(return_value=1)
        redis.expire = AsyncMock()
        return redis

    @pytest.fixture
    def limiter(self, redis_mock):
        """Create AIFeatureLimiter with mock Redis."""
        return AIFeatureLimiter(redis_client=redis_mock)

    @pytest.mark.asyncio
    async def test_free_tier_blocked(self, limiter):
        """Test that free tier users are blocked from AI features."""
        with pytest.raises(HTTPException) as exc_info:
            await limiter.check_limit(
                user_id="test-user",
                subscription_tier="free",
                feature="hallucination",
                increment=1,
            )

        assert exc_info.value.status_code == 403
        assert "requires Pro or Enterprise tier" in exc_info.value.detail
        assert "https://prela.app/pricing" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_lunch_money_tier_blocked(self, limiter):
        """Test that Lunch Money tier users are blocked from AI features."""
        with pytest.raises(HTTPException) as exc_info:
            await limiter.check_limit(
                user_id="test-user",
                subscription_tier="lunch-money",
                feature="hallucination",
                increment=1,
            )

        assert exc_info.value.status_code == 403
        assert "requires Pro or Enterprise tier" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_enterprise_tier_unlimited(self, limiter, redis_mock):
        """Test that Enterprise tier has unlimited access."""
        allowed, current, limit = await limiter.check_limit(
            user_id="test-user",
            subscription_tier="enterprise",
            feature="hallucination",
            increment=1,
        )

        # Enterprise gets unlimited access
        assert allowed is True
        assert current == 0
        assert limit is None

        # Redis should not be called for Enterprise
        redis_mock.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_pro_tier_within_limit(self, limiter, redis_mock):
        """Test Pro tier within usage limits."""
        # Mock current usage: 5,000 / 10,000
        redis_mock.get = AsyncMock(return_value="5000")

        allowed, current, limit = await limiter.check_limit(
            user_id="test-user",
            subscription_tier="pro",
            feature="hallucination",
            increment=1,
        )

        # Should be allowed
        assert allowed is True
        assert current == 5000
        assert limit == AI_LIMITS["hallucination_checks"]

    @pytest.mark.asyncio
    async def test_pro_tier_at_limit(self, limiter, redis_mock):
        """Test Pro tier at base limit (should still allow with overage)."""
        # Mock current usage: 10,000 / 10,000
        redis_mock.get = AsyncMock(return_value="10000")

        allowed, current, limit = await limiter.check_limit(
            user_id="test-user",
            subscription_tier="pro",
            feature="hallucination",
            increment=1,
        )

        # Should still be allowed (overages are billed)
        assert allowed is True
        assert current == 10000
        assert limit == AI_LIMITS["hallucination_checks"]

    @pytest.mark.asyncio
    async def test_pro_tier_overage_allowed(self, limiter, redis_mock):
        """Test Pro tier with overages is allowed (up to soft cap)."""
        # Mock current usage: 20,000 / 10,000 (2x base limit)
        redis_mock.get = AsyncMock(return_value="20000")

        allowed, current, limit = await limiter.check_limit(
            user_id="test-user",
            subscription_tier="pro",
            feature="hallucination",
            increment=1,
        )

        # Should be allowed (within 5x soft cap)
        assert allowed is True
        assert current == 20000
        assert limit == AI_LIMITS["hallucination_checks"]

    @pytest.mark.asyncio
    async def test_pro_tier_soft_cap_exceeded(self, limiter, redis_mock):
        """Test Pro tier exceeding 5x soft cap is blocked."""
        # Mock current usage: 55,000 / 10,000 (5.5x base limit, over 5x soft cap)
        redis_mock.get = AsyncMock(return_value="55000")

        with pytest.raises(HTTPException) as exc_info:
            await limiter.check_limit(
                user_id="test-user",
                subscription_tier="pro",
                feature="hallucination",
                increment=1,
            )

        assert exc_info.value.status_code == 429
        assert "significantly exceeded" in exc_info.value.detail
        assert "contact sales@prela.app" in exc_info.value.detail
        assert "Enterprise tier" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_increment_usage(self, limiter, redis_mock):
        """Test incrementing usage counter."""
        redis_mock.incrby = AsyncMock(return_value=42)

        new_count = await limiter.increment(
            user_id="test-user", feature="hallucination", increment=5
        )

        assert new_count == 42
        redis_mock.incrby.assert_called_once()
        redis_mock.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_usage(self, limiter, redis_mock):
        """Test getting current usage."""
        redis_mock.get = AsyncMock(return_value="1234")

        usage = await limiter.get_usage(user_id="test-user", feature="hallucination")

        assert usage == 1234

    @pytest.mark.asyncio
    async def test_get_usage_no_data(self, limiter, redis_mock):
        """Test getting usage when no data exists."""
        redis_mock.get = AsyncMock(return_value=None)

        usage = await limiter.get_usage(user_id="test-user", feature="hallucination")

        assert usage == 0

    @pytest.mark.asyncio
    async def test_get_all_usage(self, limiter, redis_mock):
        """Test getting all AI feature usage."""
        # Mock different usage values for each feature
        usage_map = {
            "ai_hallucination_usage:test-user:2026-02": "5000",
            "ai_drift_usage:test-user:2026-02": "25",
            "ai_nlp_usage:test-user:2026-02": "500",
        }
        redis_mock.get = AsyncMock(side_effect=lambda key: usage_map.get(key, "0"))

        usage = await limiter.get_all_usage(user_id="test-user")

        assert usage["hallucination_checks"] == 5000
        assert usage["drift_baselines"] == 25
        assert usage["nlp_searches"] == 500

    @pytest.mark.asyncio
    async def test_usage_key_format(self, limiter):
        """Test that usage keys are formatted correctly."""
        with patch.object(limiter, "_get_current_period", return_value="2026-02"):
            key = limiter._get_usage_key("user-123", "hallucination")
            assert key == "ai_hallucination_usage:user-123:2026-02"

            key = limiter._get_usage_key("user-123", "drift")
            assert key == "ai_drift_usage:user-123:2026-02"

            key = limiter._get_usage_key("user-123", "nlp")
            assert key == "ai_nlp_usage:user-123:2026-02"

    @pytest.mark.asyncio
    async def test_period_changes_monthly(self, limiter):
        """Test that period string changes monthly."""
        period = limiter._get_current_period()
        # Should be in format YYYY-MM
        assert len(period) == 7
        assert period[4] == "-"
        # Should be current period (2026-02)
        assert period == "2026-02"

    @pytest.mark.asyncio
    async def test_drift_baseline_limits(self, limiter, redis_mock):
        """Test drift baseline specific limits."""
        redis_mock.get = AsyncMock(return_value="30")

        allowed, current, limit = await limiter.check_limit(
            user_id="test-user",
            subscription_tier="pro",
            feature="drift",
            increment=1,
        )

        assert allowed is True
        assert current == 30
        assert limit == AI_LIMITS["drift_baselines"]
        assert limit == 50

    @pytest.mark.asyncio
    async def test_nlp_search_limits(self, limiter, redis_mock):
        """Test NLP search specific limits."""
        redis_mock.get = AsyncMock(return_value="800")

        allowed, current, limit = await limiter.check_limit(
            user_id="test-user",
            subscription_tier="pro",
            feature="nlp",
            increment=1,
        )

        assert allowed is True
        assert current == 800
        assert limit == AI_LIMITS["nlp_searches"]
        assert limit == 1000

    @pytest.mark.asyncio
    async def test_redis_error_fails_open(self, limiter, redis_mock):
        """Test that Redis errors fail open (allow request)."""
        # Mock Redis error
        redis_mock.get = AsyncMock(side_effect=Exception("Redis connection failed"))

        allowed, current, limit = await limiter.check_limit(
            user_id="test-user",
            subscription_tier="pro",
            feature="hallucination",
            increment=1,
        )

        # Should fail open
        assert allowed is True
        assert current == 0

    def test_ai_limits_constants(self):
        """Test that AI limit constants are correct."""
        assert AI_LIMITS["hallucination_checks"] == 10_000
        assert AI_LIMITS["drift_baselines"] == 50
        assert AI_LIMITS["nlp_searches"] == 1_000


class TestCheckAIFeatureLimitDependency:
    """Test the FastAPI dependency function."""

    @pytest.mark.asyncio
    @patch("app.middleware.ai_feature_limiter.get_ai_feature_limiter")
    async def test_dependency_blocks_when_not_allowed(self, mock_get_limiter):
        """Test that dependency raises HTTPException when limit exceeded."""
        # Mock limiter that returns not allowed
        limiter = AsyncMock()
        limiter.check_limit = AsyncMock(return_value=(False, 55000, 10000))
        limiter.increment = AsyncMock()
        mock_get_limiter.return_value = limiter

        with pytest.raises(HTTPException) as exc_info:
            await check_ai_feature_limit(
                user_id="test-user",
                subscription_tier="pro",
                feature="hallucination",
                increment=1,
            )

        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    @patch("app.middleware.ai_feature_limiter.get_ai_feature_limiter")
    async def test_dependency_increments_on_success(self, mock_get_limiter):
        """Test that dependency increments usage when allowed."""
        # Mock limiter that returns allowed
        limiter = AsyncMock()
        limiter.check_limit = AsyncMock(return_value=(True, 5000, 10000))
        limiter.increment = AsyncMock(return_value=5001)
        mock_get_limiter.return_value = limiter

        # Should not raise exception
        await check_ai_feature_limit(
            user_id="test-user",
            subscription_tier="pro",
            feature="hallucination",
            increment=1,
        )

        # Verify increment was called
        limiter.increment.assert_called_once_with("test-user", "hallucination", 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
