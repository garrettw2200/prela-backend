"""Tests for AI feature usage limits and enforcement."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from app.middleware.ai_feature_limiter import (
    AIFeatureLimiter,
    AI_LIMITS,
    LUNCH_MONEY_LIMITS,
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
        assert "https://prela.app/pricing" in exc_info.value.detail

    # --- Lunch Money tier tests ---

    @pytest.mark.asyncio
    async def test_lunch_money_within_cap(self, limiter, redis_mock):
        """Test that Lunch Money users within their hard cap are allowed."""
        redis_mock.get = AsyncMock(return_value="5")  # 5 of 10 debug sessions used

        allowed, current, limit = await limiter.check_limit(
            user_id="test-user",
            subscription_tier="lunch-money",
            feature="debug",
            increment=1,
        )

        assert allowed is True
        assert current == 5
        assert limit == LUNCH_MONEY_LIMITS["debug"]  # 10

    @pytest.mark.asyncio
    async def test_lunch_money_at_hard_cap_blocked(self, limiter, redis_mock):
        """Test that Lunch Money users at their hard cap are blocked (no overages)."""
        redis_mock.get = AsyncMock(return_value="10")  # 10/10 debug sessions used

        with pytest.raises(HTTPException) as exc_info:
            await limiter.check_limit(
                user_id="test-user",
                subscription_tier="lunch-money",
                feature="debug",
                increment=1,
            )

        assert exc_info.value.status_code == 429
        assert "Upgrade to Pro" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_lunch_money_pro_only_feature_blocked(self, limiter):
        """Test that Lunch Money users are blocked from Pro-only features (nlp, security)."""
        with pytest.raises(HTTPException) as exc_info:
            await limiter.check_limit(
                user_id="test-user",
                subscription_tier="lunch-money",
                feature="nlp",
                increment=1,
            )

        assert exc_info.value.status_code == 403
        assert "requires Pro or Enterprise tier" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_lunch_money_security_blocked(self, limiter):
        """Test that Lunch Money users are blocked from security scanning."""
        with pytest.raises(HTTPException) as exc_info:
            await limiter.check_limit(
                user_id="test-user",
                subscription_tier="lunch-money",
                feature="security",
                increment=1,
            )

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_lunch_money_hallucination_within_cap(self, limiter, redis_mock):
        """Test Lunch Money hallucination checks within 500/month cap."""
        redis_mock.get = AsyncMock(return_value="400")

        allowed, current, limit = await limiter.check_limit(
            user_id="test-user",
            subscription_tier="lunch-money",
            feature="hallucination",
            increment=1,
        )

        assert allowed is True
        assert limit == LUNCH_MONEY_LIMITS["hallucination"]  # 500

    @pytest.mark.asyncio
    async def test_lunch_money_eval_within_cap(self, limiter, redis_mock):
        """Test Lunch Money eval generations within 5/month cap."""
        redis_mock.get = AsyncMock(return_value="4")

        allowed, current, limit = await limiter.check_limit(
            user_id="test-user",
            subscription_tier="lunch-money",
            feature="eval_generation",
            increment=1,
        )

        assert allowed is True
        assert limit == LUNCH_MONEY_LIMITS["eval_generation"]  # 5

    @pytest.mark.asyncio
    async def test_lunch_money_drift_within_cap(self, limiter, redis_mock):
        """Test Lunch Money drift baselines within 10/month cap."""
        redis_mock.get = AsyncMock(return_value="9")

        allowed, current, limit = await limiter.check_limit(
            user_id="test-user",
            subscription_tier="lunch-money",
            feature="drift",
            increment=1,
        )

        assert allowed is True
        assert limit == LUNCH_MONEY_LIMITS["drift"]  # 10

    # --- Enterprise tier tests ---

    @pytest.mark.asyncio
    async def test_enterprise_tier_unlimited(self, limiter, redis_mock):
        """Test that Enterprise tier has unlimited access."""
        allowed, current, limit = await limiter.check_limit(
            user_id="test-user",
            subscription_tier="enterprise",
            feature="hallucination",
            increment=1,
        )

        assert allowed is True
        assert current == 0
        assert limit is None

        # Redis should not be called for Enterprise
        redis_mock.get.assert_not_called()

    # --- Pro tier tests ---

    @pytest.mark.asyncio
    async def test_pro_tier_within_limit(self, limiter, redis_mock):
        """Test Pro tier within usage limits."""
        redis_mock.get = AsyncMock(return_value="5000")

        allowed, current, limit = await limiter.check_limit(
            user_id="test-user",
            subscription_tier="pro",
            feature="hallucination",
            increment=1,
        )

        assert allowed is True
        assert current == 5000
        assert limit == AI_LIMITS["hallucination"]  # 10,000

    @pytest.mark.asyncio
    async def test_pro_tier_at_limit(self, limiter, redis_mock):
        """Test Pro tier at base limit (should still allow with overage)."""
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
        assert limit == AI_LIMITS["hallucination"]  # 10,000

    @pytest.mark.asyncio
    async def test_pro_tier_overage_allowed(self, limiter, redis_mock):
        """Test Pro tier with overages is allowed (up to soft cap)."""
        redis_mock.get = AsyncMock(return_value="20000")

        allowed, current, limit = await limiter.check_limit(
            user_id="test-user",
            subscription_tier="pro",
            feature="hallucination",
            increment=1,
        )

        assert allowed is True
        assert current == 20000
        assert limit == AI_LIMITS["hallucination"]  # 10,000

    @pytest.mark.asyncio
    async def test_pro_tier_soft_cap_exceeded(self, limiter, redis_mock):
        """Test Pro tier exceeding 5x soft cap is blocked."""
        # 55,000 > 10,000 * 5 = 50,000 soft cap
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
    async def test_drift_baseline_limits(self, limiter, redis_mock):
        """Test drift baseline limits (100/month for Pro)."""
        redis_mock.get = AsyncMock(return_value="30")

        allowed, current, limit = await limiter.check_limit(
            user_id="test-user",
            subscription_tier="pro",
            feature="drift",
            increment=1,
        )

        assert allowed is True
        assert current == 30
        assert limit == AI_LIMITS["drift"]
        assert limit == 100

    @pytest.mark.asyncio
    async def test_debug_session_limits(self, limiter, redis_mock):
        """Test debug session limits (200/month for Pro)."""
        redis_mock.get = AsyncMock(return_value="150")

        allowed, current, limit = await limiter.check_limit(
            user_id="test-user",
            subscription_tier="pro",
            feature="debug",
            increment=1,
        )

        assert allowed is True
        assert current == 150
        assert limit == AI_LIMITS["debug"]
        assert limit == 200

    @pytest.mark.asyncio
    async def test_eval_generation_limits(self, limiter, redis_mock):
        """Test eval generation limits (250/month for Pro)."""
        redis_mock.get = AsyncMock(return_value="100")

        allowed, current, limit = await limiter.check_limit(
            user_id="test-user",
            subscription_tier="pro",
            feature="eval_generation",
            increment=1,
        )

        assert allowed is True
        assert current == 100
        assert limit == AI_LIMITS["eval_generation"]
        assert limit == 250

    @pytest.mark.asyncio
    async def test_nlp_search_limits(self, limiter, redis_mock):
        """Test NLP search specific limits (1,000/month for Pro)."""
        redis_mock.get = AsyncMock(return_value="800")

        allowed, current, limit = await limiter.check_limit(
            user_id="test-user",
            subscription_tier="pro",
            feature="nlp",
            increment=1,
        )

        assert allowed is True
        assert current == 800
        assert limit == AI_LIMITS["nlp"]
        assert limit == 1000

    @pytest.mark.asyncio
    async def test_redis_error_fails_open(self, limiter, redis_mock):
        """Test that Redis errors fail open (allow request)."""
        redis_mock.get = AsyncMock(side_effect=Exception("Redis connection failed"))

        allowed, current, limit = await limiter.check_limit(
            user_id="test-user",
            subscription_tier="pro",
            feature="hallucination",
            increment=1,
        )

        assert allowed is True
        assert current == 0

    # --- Increment and usage tests ---

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
    async def test_usage_key_format(self, limiter):
        """Test that usage keys are formatted correctly."""
        with patch.object(limiter, "_get_current_period", return_value="2026-03"):
            key = limiter._get_usage_key("user-123", "hallucination")
            assert key == "ai_hallucination_usage:user-123:2026-03"

            key = limiter._get_usage_key("user-123", "drift")
            assert key == "ai_drift_usage:user-123:2026-03"

            key = limiter._get_usage_key("user-123", "nlp")
            assert key == "ai_nlp_usage:user-123:2026-03"

    @pytest.mark.asyncio
    async def test_period_changes_monthly(self, limiter):
        """Test that period string changes monthly."""
        period = limiter._get_current_period()
        assert len(period) == 7
        assert period[4] == "-"

    # --- Constants tests ---

    def test_ai_limits_constants(self):
        """Test that Pro AI limit constants are correct."""
        assert AI_LIMITS["hallucination"] == 10_000
        assert AI_LIMITS["drift"] == 100
        assert AI_LIMITS["nlp"] == 1_000
        assert AI_LIMITS["debug"] == 200
        assert AI_LIMITS["eval_generation"] == 250
        assert AI_LIMITS["security"] == 10_000

    def test_lunch_money_limits_constants(self):
        """Test that Lunch Money AI limit constants are correct."""
        assert LUNCH_MONEY_LIMITS["hallucination"] == 500
        assert LUNCH_MONEY_LIMITS["drift"] == 10
        assert LUNCH_MONEY_LIMITS["debug"] == 10
        assert LUNCH_MONEY_LIMITS["eval_generation"] == 5
        assert "nlp" not in LUNCH_MONEY_LIMITS
        assert "security" not in LUNCH_MONEY_LIMITS


class TestCheckAIFeatureLimitDependency:
    """Test the FastAPI dependency function."""

    @pytest.mark.asyncio
    @patch("app.middleware.ai_feature_limiter.get_ai_feature_limiter")
    async def test_dependency_blocks_when_not_allowed(self, mock_get_limiter):
        """Test that dependency raises HTTPException when limit exceeded."""
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
        limiter = AsyncMock()
        limiter.check_limit = AsyncMock(return_value=(True, 5000, 10000))
        limiter.increment = AsyncMock(return_value=5001)
        mock_get_limiter.return_value = limiter

        await check_ai_feature_limit(
            user_id="test-user",
            subscription_tier="pro",
            feature="hallucination",
            increment=1,
        )

        limiter.increment.assert_called_once_with("test-user", "hallucination", 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
