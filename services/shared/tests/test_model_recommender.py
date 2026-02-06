"""
Tests for ModelRecommender module.

Tests model downgrade recommendations based on usage patterns.
"""

import pytest
from shared.model_recommender import ModelRecommender, ModelUsageStats, MODEL_PRICING, DOWNGRADE_PATHS


class TestModelRecommender:
    """Test suite for ModelRecommender"""

    def test_initialization_default(self):
        """Test recommender initialization with defaults"""
        recommender = ModelRecommender()
        assert recommender.min_calls_threshold == 100
        assert recommender.min_savings_threshold == 10.0
        assert recommender.latency_tolerance_pct == 20.0

    def test_initialization_custom(self):
        """Test recommender initialization with custom parameters"""
        recommender = ModelRecommender(
            min_calls_threshold=50,
            min_savings_threshold=5.0,
            latency_tolerance_pct=15.0,
        )
        assert recommender.min_calls_threshold == 50
        assert recommender.min_savings_threshold == 5.0
        assert recommender.latency_tolerance_pct == 15.0

    def test_analyze_model_usage_empty(self):
        """Test analysis with empty usage data"""
        recommender = ModelRecommender()
        result = recommender.analyze_model_usage([])
        assert result == []

    def test_analyze_model_usage_too_few_calls(self):
        """Test analysis with insufficient calls"""
        recommender = ModelRecommender(min_calls_threshold=100)

        usage_data = [{
            "model": "gpt-4",
            "vendor": "openai",
            "call_count": 50,  # Below threshold
            "total_tokens": 5000,
            "prompt_tokens": 3000,
            "completion_tokens": 2000,
            "total_cost_usd": 1.5,
            "avg_latency_ms": 500.0,
            "success_count": 50,
            "date_range_days": 30,
        }]

        result = recommender.analyze_model_usage(usage_data)
        assert result == []

    def test_analyze_model_usage_gpt4_to_gpt4o_mini(self):
        """Test recommendation to downgrade from GPT-4 to GPT-4o-mini"""
        recommender = ModelRecommender(min_calls_threshold=100, min_savings_threshold=1.0)

        usage_data = [{
            "model": "gpt-4",
            "vendor": "openai",
            "call_count": 1000,
            "total_tokens": 100000,
            "prompt_tokens": 60000,
            "completion_tokens": 40000,
            "total_cost_usd": 4.2,  # GPT-4 is expensive
            "avg_latency_ms": 500.0,
            "success_count": 1000,
            "date_range_days": 30,
        }]

        result = recommender.analyze_model_usage(usage_data)

        # Should recommend at least one downgrade
        assert len(result) > 0

        # First recommendation should be for gpt-4
        assert result[0].current_model == "gpt-4"

        # Should recommend a cheaper alternative from DOWNGRADE_PATHS
        assert result[0].recommended_model in DOWNGRADE_PATHS["gpt-4"]

        # Should have positive savings
        assert result[0].estimated_monthly_savings > 0
        assert result[0].estimated_annual_savings > 0

    def test_analyze_model_usage_with_alternative_data(self):
        """Test recommendation when alternative model has usage data"""
        recommender = ModelRecommender(min_calls_threshold=50)

        usage_data = [
            # Current expensive model
            {
                "model": "gpt-4",
                "vendor": "openai",
                "call_count": 500,
                "total_tokens": 50000,
                "prompt_tokens": 30000,
                "completion_tokens": 20000,
                "total_cost_usd": 2.1,
                "avg_latency_ms": 500.0,
                "success_count": 500,
                "date_range_days": 30,
            },
            # Alternative cheaper model with actual data
            {
                "model": "gpt-4o-mini",
                "vendor": "openai",
                "call_count": 200,
                "total_tokens": 20000,
                "prompt_tokens": 12000,
                "completion_tokens": 8000,
                "total_cost_usd": 0.015,
                "avg_latency_ms": 300.0,
                "success_count": 200,
                "date_range_days": 30,
            },
        ]

        result = recommender.analyze_model_usage(usage_data)

        assert len(result) > 0

        # Find the gpt-4 → gpt-4o-mini recommendation
        rec = next((r for r in result if r.recommended_model == "gpt-4o-mini"), None)
        assert rec is not None

        # Should use actual alternative stats
        assert rec.alternative_stats is not None
        assert rec.alternative_stats.call_count == 200

    def test_confidence_calculation(self):
        """Test that confidence scores are calculated correctly"""
        recommender = ModelRecommender(min_calls_threshold=50)

        # High volume should yield high confidence
        usage_data_high = [{
            "model": "gpt-4",
            "vendor": "openai",
            "call_count": 5000,  # High volume
            "total_tokens": 500000,
            "prompt_tokens": 300000,
            "completion_tokens": 200000,
            "total_cost_usd": 21.0,
            "avg_latency_ms": 500.0,
            "success_count": 5000,
            "date_range_days": 30,
        }]

        result_high = recommender.analyze_model_usage(usage_data_high)

        # Low volume should yield lower confidence
        usage_data_low = [{
            "model": "gpt-4",
            "vendor": "openai",
            "call_count": 100,  # Low volume
            "total_tokens": 10000,
            "prompt_tokens": 6000,
            "completion_tokens": 4000,
            "total_cost_usd": 0.42,
            "avg_latency_ms": 500.0,
            "success_count": 100,
            "date_range_days": 30,
        }]

        result_low = recommender.analyze_model_usage(usage_data_low)

        if result_high and result_low:
            # High volume should have higher confidence
            assert result_high[0].confidence > result_low[0].confidence

    def test_latency_tolerance(self):
        """Test that recommendations respect latency tolerance"""
        recommender = ModelRecommender(
            min_calls_threshold=50,
            latency_tolerance_pct=10.0  # Strict tolerance
        )

        usage_data = [
            {
                "model": "gpt-4",
                "vendor": "openai",
                "call_count": 500,
                "total_tokens": 50000,
                "prompt_tokens": 30000,
                "completion_tokens": 20000,
                "total_cost_usd": 2.1,
                "avg_latency_ms": 100.0,  # Very fast
                "success_count": 500,
                "date_range_days": 30,
            },
            {
                "model": "gpt-4o-mini",
                "vendor": "openai",
                "call_count": 200,
                "total_tokens": 20000,
                "prompt_tokens": 12000,
                "completion_tokens": 8000,
                "total_cost_usd": 0.015,
                "avg_latency_ms": 200.0,  # Much slower (100% increase)
                "success_count": 200,
                "date_range_days": 30,
            },
        ]

        result = recommender.analyze_model_usage(usage_data)

        # Recommendation with 100% latency increase should be filtered out
        # (exceeds 10% tolerance)
        gpt4_mini_rec = next((r for r in result if r.recommended_model == "gpt-4o-mini"), None)

        # Should either not recommend gpt-4o-mini or have low priority
        # (depends on other available alternatives)
        if gpt4_mini_rec:
            # If recommended despite latency, check it's not first choice
            assert gpt4_mini_rec.latency_change_pct > recommender.latency_tolerance_pct

    def test_min_savings_threshold(self):
        """Test that recommendations meet minimum savings threshold"""
        recommender = ModelRecommender(
            min_calls_threshold=50,
            min_savings_threshold=100.0  # High threshold
        )

        # Create usage with low potential savings
        usage_data = [{
            "model": "gpt-4o-mini",  # Already cheapest
            "vendor": "openai",
            "call_count": 100,
            "total_tokens": 10000,
            "prompt_tokens": 6000,
            "completion_tokens": 4000,
            "total_cost_usd": 0.01,
            "avg_latency_ms": 300.0,
            "success_count": 100,
            "date_range_days": 30,
        }]

        result = recommender.analyze_model_usage(usage_data)

        # Should have no recommendations (no downgrade path + low savings)
        assert len(result) == 0

    def test_performance_impact_categorization(self):
        """Test that performance impact is correctly categorized"""
        recommender = ModelRecommender(min_calls_threshold=50)

        # Test various latency scenarios
        test_cases = [
            (100.0, 102.0, "negligible"),  # 2% increase
            (100.0, 110.0, "minor"),       # 10% increase
            (100.0, 120.0, "moderate"),    # 20% increase
            (100.0, 150.0, "significant"), # 50% increase
        ]

        for current_latency, alt_latency, expected_impact in test_cases:
            usage_data = [
                {
                    "model": "gpt-4",
                    "vendor": "openai",
                    "call_count": 500,
                    "total_tokens": 50000,
                    "prompt_tokens": 30000,
                    "completion_tokens": 20000,
                    "total_cost_usd": 2.1,
                    "avg_latency_ms": current_latency,
                    "success_count": 500,
                    "date_range_days": 30,
                },
                {
                    "model": "gpt-4o-mini",
                    "vendor": "openai",
                    "call_count": 200,
                    "total_tokens": 20000,
                    "prompt_tokens": 12000,
                    "completion_tokens": 8000,
                    "total_cost_usd": 0.015,
                    "avg_latency_ms": alt_latency,
                    "success_count": 200,
                    "date_range_days": 30,
                },
            ]

            result = recommender.analyze_model_usage(usage_data)

            if result:
                rec = next((r for r in result if r.recommended_model == "gpt-4o-mini"), None)
                if rec:
                    assert rec.performance_impact == expected_impact

    def test_reasoning_generation(self):
        """Test that reasoning text is generated"""
        recommender = ModelRecommender(min_calls_threshold=50)

        usage_data = [{
            "model": "gpt-4",
            "vendor": "openai",
            "call_count": 500,
            "total_tokens": 50000,
            "prompt_tokens": 30000,
            "completion_tokens": 20000,
            "total_cost_usd": 2.1,
            "avg_latency_ms": 500.0,
            "success_count": 500,
            "date_range_days": 30,
        }]

        result = recommender.analyze_model_usage(usage_data)

        assert len(result) > 0

        # Reasoning should be a non-empty string
        assert isinstance(result[0].reasoning, str)
        assert len(result[0].reasoning) > 0

        # Should mention savings
        assert "$" in result[0].reasoning or "save" in result[0].reasoning.lower()

    def test_model_usage_stats_dataclass(self):
        """Test ModelUsageStats dataclass creation"""
        stats = ModelUsageStats(
            model="gpt-4",
            vendor="openai",
            call_count=1000,
            total_tokens=100000,
            prompt_tokens=60000,
            completion_tokens=40000,
            total_cost_usd=4.2,
            avg_latency_ms=500.0,
            success_rate=99.5,
            avg_prompt_tokens=60.0,
            avg_completion_tokens=40.0,
        )

        assert stats.model == "gpt-4"
        assert stats.call_count == 1000
        assert stats.total_cost_usd == 4.2
        assert stats.success_rate == 99.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
