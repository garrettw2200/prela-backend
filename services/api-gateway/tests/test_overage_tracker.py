"""Tests for Pro tier overage tracking and billing."""

import pytest
from datetime import datetime, timezone
from decimal import Decimal

from app.services.overage_tracker import (
    OverageTracker,
    OVERAGE_PRICING,
    PRO_BASE_LIMITS,
)


class TestOverageTracker:
    """Test overage calculation and tracking."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tracker = OverageTracker(db_connection=None)

    def test_no_overages_within_limits(self):
        """Test that usage within limits has no overages."""
        usage = {
            "traces": 500_000,
            "users": 3,
            "ai_hallucination_checks": 5_000,
            "ai_drift_baselines": 50,          # under new limit of 100
            "ai_nlp_searches": 500,
            "ai_debug_sessions": 100,           # under new limit of 200
            "ai_eval_generations": 100,         # under new limit of 250
            "retention_days": 90,
        }

        overages = self.tracker.calculate_overages(
            subscription_id="test-sub-id",
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            usage=usage,
        )

        assert overages.get("traces") is None or overages["traces"] == Decimal("0.00")
        assert overages.get("users") is None or overages["users"] == Decimal("0.00")
        assert overages.get("ai_hallucination") is None or overages["ai_hallucination"] == Decimal("0.00")
        assert overages.get("ai_drift") is None or overages["ai_drift"] == Decimal("0.00")
        assert overages.get("ai_nlp") is None or overages["ai_nlp"] == Decimal("0.00")
        assert overages.get("ai_debug") is None or overages["ai_debug"] == Decimal("0.00")
        assert overages.get("ai_eval") is None or overages["ai_eval"] == Decimal("0.00")
        assert overages.get("retention") is None or overages["retention"] == Decimal("0.00")
        assert overages["total"] == Decimal("0.00")

    def test_trace_overages(self):
        """Test trace overage calculation."""
        usage = {
            "traces": 1_500_000,  # 500k over 1M limit
            "users": 5,
            "ai_hallucination_checks": 10_000,
            "ai_drift_baselines": 100,
            "ai_nlp_searches": 1_000,
            "retention_days": 90,
        }

        overages = self.tracker.calculate_overages(
            subscription_id="test-sub-id",
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            usage=usage,
        )

        # 500k traces over = 5 units of 100k @ $8 each = $40
        assert overages["traces"] == Decimal("40.00")
        assert overages["total"] == Decimal("40.00")

    def test_user_overages(self):
        """Test user overage calculation."""
        usage = {
            "traces": 1_000_000,
            "users": 8,  # 3 over 5 limit
            "ai_hallucination_checks": 10_000,
            "ai_drift_baselines": 100,
            "ai_nlp_searches": 1_000,
            "retention_days": 90,
        }

        overages = self.tracker.calculate_overages(
            subscription_id="test-sub-id",
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            usage=usage,
        )

        # 3 users over @ $12 each = $36
        assert overages["users"] == Decimal("36.00")
        assert overages["total"] == Decimal("36.00")

    def test_ai_hallucination_overages(self):
        """Test AI hallucination check overage calculation."""
        usage = {
            "traces": 1_000_000,
            "users": 5,
            "ai_hallucination_checks": 25_000,  # 15k over 10k limit
            "ai_drift_baselines": 100,
            "ai_nlp_searches": 1_000,
            "retention_days": 90,
        }

        overages = self.tracker.calculate_overages(
            subscription_id="test-sub-id",
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            usage=usage,
        )

        # 15k checks over = 1.5 units of 10k @ $5 each = $7.50
        assert overages["ai_hallucination"] == Decimal("7.50")
        assert overages["total"] == Decimal("7.50")

    def test_ai_drift_overages(self):
        """Test AI drift baseline overage calculation (new limit: 100/month)."""
        usage = {
            "traces": 1_000_000,
            "users": 5,
            "ai_hallucination_checks": 10_000,
            "ai_drift_baselines": 130,  # 30 over new 100 limit
            "ai_nlp_searches": 1_000,
            "retention_days": 90,
        }

        overages = self.tracker.calculate_overages(
            subscription_id="test-sub-id",
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            usage=usage,
        )

        # 30 baselines over = 3 units of 10 @ $2 each = $6
        assert overages["ai_drift"] == Decimal("6.00")
        assert overages["total"] == Decimal("6.00")

    def test_ai_nlp_overages(self):
        """Test AI NLP search overage calculation."""
        usage = {
            "traces": 1_000_000,
            "users": 5,
            "ai_hallucination_checks": 10_000,
            "ai_drift_baselines": 100,
            "ai_nlp_searches": 3_500,  # 2.5k over 1k limit
            "retention_days": 90,
        }

        overages = self.tracker.calculate_overages(
            subscription_id="test-sub-id",
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            usage=usage,
        )

        # 2.5k searches over = 2.5 units of 1k @ $3 each = $7.50
        assert overages["ai_nlp"] == Decimal("7.50")
        assert overages["total"] == Decimal("7.50")

    def test_ai_debug_overages(self):
        """Test AI debug session overage calculation (new limit: 200/month)."""
        usage = {
            "traces": 1_000_000,
            "users": 5,
            "ai_hallucination_checks": 10_000,
            "ai_drift_baselines": 100,
            "ai_nlp_searches": 1_000,
            "ai_debug_sessions": 220,  # 20 over new 200 limit
            "retention_days": 90,
        }

        overages = self.tracker.calculate_overages(
            subscription_id="test-sub-id",
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            usage=usage,
        )

        # 20 sessions over = 2 units of 10 @ $1 each = $2
        assert overages["ai_debug"] == Decimal("2.00")
        assert overages["total"] == Decimal("2.00")

    def test_ai_eval_generation_overages(self):
        """Test AI eval generation overage calculation (new: 250/month, $1/10)."""
        usage = {
            "traces": 1_000_000,
            "users": 5,
            "ai_hallucination_checks": 10_000,
            "ai_drift_baselines": 100,
            "ai_nlp_searches": 1_000,
            "ai_eval_generations": 260,  # 10 over 250 limit
            "retention_days": 90,
        }

        overages = self.tracker.calculate_overages(
            subscription_id="test-sub-id",
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            usage=usage,
        )

        # 10 evals over = 1 unit of 10 @ $1 = $1.00
        assert overages["ai_eval"] == Decimal("1.00")
        assert overages["total"] == Decimal("1.00")

    def test_ai_eval_no_overage_within_limit(self):
        """Test that eval generations within limit produce no overage."""
        usage = {
            "traces": 1_000_000,
            "users": 5,
            "ai_hallucination_checks": 10_000,
            "ai_drift_baselines": 100,
            "ai_nlp_searches": 1_000,
            "ai_eval_generations": 200,  # under 250 limit
            "retention_days": 90,
        }

        overages = self.tracker.calculate_overages(
            subscription_id="test-sub-id",
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            usage=usage,
        )

        assert overages.get("ai_eval") is None or overages["ai_eval"] == Decimal("0.00")
        assert overages["total"] == Decimal("0.00")

    def test_retention_overages(self):
        """Test retention overage calculation."""
        usage = {
            "traces": 1_000_000,
            "users": 5,
            "ai_hallucination_checks": 10_000,
            "ai_drift_baselines": 100,
            "ai_nlp_searches": 1_000,
            "retention_days": 180,  # 90 days over 90 limit
        }

        overages = self.tracker.calculate_overages(
            subscription_id="test-sub-id",
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            usage=usage,
        )

        # 90 days over = 3 units of 30 @ $10 each = $30
        assert overages["retention"] == Decimal("30.00")
        assert overages["total"] == Decimal("30.00")

    def test_multiple_overages_combined(self):
        """Test multiple overages combined."""
        usage = {
            "traces": 2_000_000,             # 1M over = $80
            "users": 8,                       # 3 over = $36
            "ai_hallucination_checks": 20_000, # 10k over = $5
            "ai_drift_baselines": 120,         # 20 over new 100 limit = $4
            "ai_nlp_searches": 2_000,          # 1k over = $3
            "ai_eval_generations": 260,         # 10 over 250 limit = $1
            "retention_days": 120,             # 30 days over = $10
        }

        overages = self.tracker.calculate_overages(
            subscription_id="test-sub-id",
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            usage=usage,
        )

        assert overages["traces"] == Decimal("80.00")
        assert overages["users"] == Decimal("36.00")
        assert overages["ai_hallucination"] == Decimal("5.00")
        assert overages["ai_drift"] == Decimal("4.00")
        assert overages["ai_nlp"] == Decimal("3.00")
        assert overages["ai_eval"] == Decimal("1.00")
        assert overages["retention"] == Decimal("10.00")

        expected_total = Decimal("139.00")
        assert overages["total"] == expected_total

    def test_fractional_overages_rounded_correctly(self):
        """Test that fractional overages are rounded to 2 decimal places."""
        usage = {
            "traces": 1_150_000,  # 150k over = 1.5 units @ $8 = $12.00
            "users": 5,
            "ai_hallucination_checks": 10_000,
            "ai_drift_baselines": 100,
            "ai_nlp_searches": 1_000,
            "retention_days": 90,
        }

        overages = self.tracker.calculate_overages(
            subscription_id="test-sub-id",
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            usage=usage,
        )

        assert overages["traces"] == Decimal("12.00")
        assert isinstance(overages["traces"], Decimal)
        assert overages["traces"].as_tuple().exponent == -2

    def test_pricing_constants(self):
        """Test that pricing constants are set correctly."""
        assert OVERAGE_PRICING["traces_per_100k"] == Decimal("8.00")
        assert OVERAGE_PRICING["users"] == Decimal("12.00")
        assert OVERAGE_PRICING["ai_hallucination_per_10k"] == Decimal("5.00")
        assert OVERAGE_PRICING["ai_drift_per_10"] == Decimal("2.00")
        assert OVERAGE_PRICING["ai_nlp_per_1k"] == Decimal("3.00")
        assert OVERAGE_PRICING["retention_per_30_days"] == Decimal("10.00")
        assert OVERAGE_PRICING["ai_debug_per_10"] == Decimal("1.00")
        assert OVERAGE_PRICING["ai_eval_per_10"] == Decimal("1.00")

    def test_base_limits_constants(self):
        """Test that base limits are set correctly."""
        assert PRO_BASE_LIMITS["traces"] == 1_000_000
        assert PRO_BASE_LIMITS["users"] == 5
        assert PRO_BASE_LIMITS["ai_hallucination_checks"] == 10_000
        assert PRO_BASE_LIMITS["ai_drift_baselines"] == 100
        assert PRO_BASE_LIMITS["ai_nlp_searches"] == 1_000
        assert PRO_BASE_LIMITS["retention_days"] == 90
        assert PRO_BASE_LIMITS["ai_debug_sessions"] == 200
        assert PRO_BASE_LIMITS["ai_eval_generations"] == 250


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
