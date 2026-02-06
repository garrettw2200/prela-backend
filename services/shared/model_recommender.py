"""
Model Recommender for Cost Optimization (P2.4.1)

Analyzes LLM usage patterns and recommends cheaper model alternatives
based on performance metrics and cost savings potential.

100% internal - no external LLM dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# Model pricing database (USD per 1M tokens)
# Source: OpenAI/Anthropic pricing as of January 2026
MODEL_PRICING = {
    # OpenAI models
    "gpt-4": {"input": 30.00, "output": 60.00, "vendor": "openai"},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00, "vendor": "openai"},
    "gpt-4o": {"input": 2.50, "output": 10.00, "vendor": "openai"},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "vendor": "openai"},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50, "vendor": "openai"},
    # Anthropic models
    "claude-opus-4": {"input": 15.00, "output": 75.00, "vendor": "anthropic"},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00, "vendor": "anthropic"},
    "claude-sonnet-3-5": {"input": 3.00, "output": 15.00, "vendor": "anthropic"},
    "claude-haiku-3-5": {"input": 0.80, "output": 4.00, "vendor": "anthropic"},
    "claude-haiku-3": {"input": 0.25, "output": 1.25, "vendor": "anthropic"},
}

# Model downgrade recommendations (from → to alternatives)
DOWNGRADE_PATHS = {
    "gpt-4": ["gpt-4-turbo", "gpt-4o", "gpt-4o-mini"],
    "gpt-4-turbo": ["gpt-4o", "gpt-4o-mini"],
    "gpt-4o": ["gpt-4o-mini"],
    "claude-opus-4": ["claude-sonnet-4", "claude-haiku-3-5"],
    "claude-sonnet-4": ["claude-haiku-3-5"],
    "claude-sonnet-3-5": ["claude-haiku-3-5"],
}


@dataclass
class ModelUsageStats:
    """Statistics for a specific model's usage"""

    model: str
    vendor: str
    call_count: int
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_cost_usd: float
    avg_latency_ms: float
    success_rate: float  # percentage of successful calls
    avg_prompt_tokens: float
    avg_completion_tokens: float


@dataclass
class ModelRecommendation:
    """A recommendation to switch from one model to another"""

    current_model: str
    recommended_model: str
    confidence: float  # 0.0-1.0
    estimated_monthly_savings: float
    estimated_annual_savings: float
    performance_impact: str  # "negligible", "minor", "moderate", "significant"
    latency_change_pct: float  # percentage change in latency
    reasoning: str
    current_stats: ModelUsageStats
    alternative_stats: ModelUsageStats | None


class ModelRecommender:
    """
    Analyzes model usage and recommends cost-effective alternatives.

    Uses actual performance data from ClickHouse to make informed recommendations.
    """

    def __init__(
        self,
        min_calls_threshold: int = 100,
        min_savings_threshold: float = 10.0,
        latency_tolerance_pct: float = 20.0,
    ):
        """
        Initialize the model recommender.

        Args:
            min_calls_threshold: Minimum calls required to make a recommendation
            min_savings_threshold: Minimum monthly savings to recommend (USD)
            latency_tolerance_pct: Maximum acceptable latency increase (percentage)
        """
        self.min_calls_threshold = min_calls_threshold
        self.min_savings_threshold = min_savings_threshold
        self.latency_tolerance_pct = latency_tolerance_pct

    def analyze_model_usage(
        self,
        usage_data: list[dict[str, Any]],
    ) -> list[ModelRecommendation]:
        """
        Analyze model usage and generate downgrade recommendations.

        Args:
            usage_data: List of dicts with keys: model, vendor, call_count, total_tokens,
                       prompt_tokens, completion_tokens, total_cost_usd, avg_latency_ms,
                       success_count, date_range_days

        Returns:
            List of ModelRecommendation objects, sorted by savings potential
        """
        if not usage_data:
            return []

        # Group by model
        models_map: dict[str, ModelUsageStats] = {}
        for entry in usage_data:
            model = entry["model"]
            if model not in models_map:
                avg_prompt = (
                    entry["prompt_tokens"] / entry["call_count"]
                    if entry["call_count"] > 0
                    else 0
                )
                avg_completion = (
                    entry["completion_tokens"] / entry["call_count"]
                    if entry["call_count"] > 0
                    else 0
                )
                success_rate = (
                    entry.get("success_count", entry["call_count"]) / entry["call_count"] * 100
                    if entry["call_count"] > 0
                    else 0
                )

                models_map[model] = ModelUsageStats(
                    model=model,
                    vendor=entry.get("vendor", ""),
                    call_count=entry["call_count"],
                    total_tokens=entry["total_tokens"],
                    prompt_tokens=entry["prompt_tokens"],
                    completion_tokens=entry["completion_tokens"],
                    total_cost_usd=entry["total_cost_usd"],
                    avg_latency_ms=entry.get("avg_latency_ms", 0.0),
                    success_rate=success_rate,
                    avg_prompt_tokens=avg_prompt,
                    avg_completion_tokens=avg_completion,
                )

        # Generate recommendations
        recommendations = []
        for model, stats in models_map.items():
            # Skip if not enough calls
            if stats.call_count < self.min_calls_threshold:
                logger.debug(f"Skipping {model}: only {stats.call_count} calls")
                continue

            # Check if downgrade paths exist
            if model not in DOWNGRADE_PATHS:
                logger.debug(f"No downgrade paths for {model}")
                continue

            # Evaluate each alternative
            for alt_model in DOWNGRADE_PATHS[model]:
                alt_stats = models_map.get(alt_model)

                # Calculate potential savings
                savings = self._calculate_savings(
                    stats,
                    alt_model,
                    alt_stats,
                    date_range_days=usage_data[0].get("date_range_days", 30),
                )

                if savings["monthly_savings"] < self.min_savings_threshold:
                    continue

                # Assess performance impact
                perf_impact, latency_change = self._assess_performance_impact(
                    stats, alt_stats
                )

                # Skip if latency increase exceeds tolerance
                if latency_change > self.latency_tolerance_pct:
                    logger.debug(
                        f"Skipping {model}→{alt_model}: latency increase {latency_change:.1f}% exceeds {self.latency_tolerance_pct}%"
                    )
                    continue

                # Calculate confidence
                confidence = self._calculate_confidence(stats, alt_stats, latency_change)

                # Generate reasoning
                reasoning = self._generate_reasoning(
                    stats,
                    alt_model,
                    savings["monthly_savings"],
                    savings["annual_savings"],
                    perf_impact,
                    latency_change,
                    alt_stats,
                )

                rec = ModelRecommendation(
                    current_model=model,
                    recommended_model=alt_model,
                    confidence=confidence,
                    estimated_monthly_savings=savings["monthly_savings"],
                    estimated_annual_savings=savings["annual_savings"],
                    performance_impact=perf_impact,
                    latency_change_pct=latency_change,
                    reasoning=reasoning,
                    current_stats=stats,
                    alternative_stats=alt_stats,
                )
                recommendations.append(rec)

        # Sort by savings potential
        return sorted(
            recommendations, key=lambda r: r.estimated_annual_savings, reverse=True
        )

    def _calculate_savings(
        self,
        current: ModelUsageStats,
        alt_model: str,
        alt_stats: ModelUsageStats | None,
        date_range_days: int = 30,
    ) -> dict[str, float]:
        """Calculate potential savings from switching models"""
        # Get pricing
        current_pricing = MODEL_PRICING.get(current.model)
        alt_pricing = MODEL_PRICING.get(alt_model)

        if not current_pricing or not alt_pricing:
            return {"monthly_savings": 0.0, "annual_savings": 0.0}

        # Calculate cost per call for current model
        if current.call_count > 0:
            current_cost_per_call = current.total_cost_usd / current.call_count
        else:
            return {"monthly_savings": 0.0, "annual_savings": 0.0}

        # Estimate cost per call for alternative
        # Use actual alt stats if available, otherwise estimate from pricing
        if alt_stats and alt_stats.call_count > 0:
            alt_cost_per_call = alt_stats.total_cost_usd / alt_stats.call_count
        else:
            # Estimate based on current token usage and alt pricing
            avg_prompt = current.avg_prompt_tokens
            avg_completion = current.avg_completion_tokens
            alt_cost_per_call = (
                avg_prompt / 1_000_000 * alt_pricing["input"]
                + avg_completion / 1_000_000 * alt_pricing["output"]
            )

        # Project to monthly/annual
        calls_per_day = current.call_count / date_range_days
        monthly_calls = calls_per_day * 30
        annual_calls = calls_per_day * 365

        monthly_savings = (current_cost_per_call - alt_cost_per_call) * monthly_calls
        annual_savings = (current_cost_per_call - alt_cost_per_call) * annual_calls

        return {
            "monthly_savings": max(0.0, monthly_savings),
            "annual_savings": max(0.0, annual_savings),
        }

    def _assess_performance_impact(
        self,
        current: ModelUsageStats,
        alt: ModelUsageStats | None,
    ) -> tuple[str, float]:
        """
        Assess performance impact of switching models.

        Returns:
            (impact_level, latency_change_pct)
        """
        if not alt or alt.avg_latency_ms == 0:
            # No data available, assume minor impact
            return ("minor", 0.0)

        latency_change_pct = (
            (alt.avg_latency_ms - current.avg_latency_ms) / current.avg_latency_ms * 100
            if current.avg_latency_ms > 0
            else 0.0
        )

        # Categorize impact
        if abs(latency_change_pct) < 5:
            impact = "negligible"
        elif abs(latency_change_pct) < 15:
            impact = "minor"
        elif abs(latency_change_pct) < 30:
            impact = "moderate"
        else:
            impact = "significant"

        return (impact, latency_change_pct)

    def _calculate_confidence(
        self,
        current: ModelUsageStats,
        alt: ModelUsageStats | None,
        latency_change: float,
    ) -> float:
        """
        Calculate confidence score for recommendation (0.0-1.0).

        Factors:
        - More calls = higher confidence
        - Alternative has usage data = higher confidence
        - Lower latency impact = higher confidence
        - Higher success rates = higher confidence
        """
        confidence = 0.5  # Base confidence

        # Factor 1: Call volume (max +0.25)
        if current.call_count >= 1000:
            confidence += 0.25
        elif current.call_count >= 500:
            confidence += 0.20
        elif current.call_count >= 200:
            confidence += 0.15
        else:
            confidence += 0.10

        # Factor 2: Alternative has data (max +0.15)
        if alt and alt.call_count >= 100:
            confidence += 0.15
        elif alt and alt.call_count >= 10:
            confidence += 0.10

        # Factor 3: Latency impact (max +0.10)
        if abs(latency_change) < 5:
            confidence += 0.10
        elif abs(latency_change) < 15:
            confidence += 0.05

        # Cap at 1.0
        return min(1.0, confidence)

    def _generate_reasoning(
        self,
        current: ModelUsageStats,
        alt_model: str,
        monthly_savings: float,
        annual_savings: float,
        perf_impact: str,
        latency_change: float,
        alt_stats: ModelUsageStats | None,
    ) -> str:
        """Generate human-readable reasoning for recommendation"""
        parts = []

        # Savings summary
        parts.append(
            f"Switching from {current.model} to {alt_model} could save ${monthly_savings:.2f}/month (${annual_savings:.2f}/year)."
        )

        # Performance impact
        if perf_impact == "negligible":
            parts.append("Performance impact is negligible.")
        elif perf_impact == "minor":
            direction = "faster" if latency_change < 0 else "slower"
            parts.append(
                f"Minor performance impact: {abs(latency_change):.1f}% {direction} on average."
            )
        elif perf_impact == "moderate":
            parts.append(
                f"Moderate performance impact: {latency_change:+.1f}% latency change. Consider testing for quality."
            )

        # Data availability
        if alt_stats and alt_stats.call_count >= 50:
            parts.append(
                f"Based on {alt_stats.call_count} actual {alt_model} calls with {alt_stats.success_rate:.1f}% success rate."
            )
        else:
            parts.append(
                f"Estimated based on pricing. Consider A/B testing {alt_model} for quality validation."
            )

        # Usage pattern
        parts.append(
            f"Current usage: {current.call_count} calls averaging {current.avg_prompt_tokens:.0f} prompt + {current.avg_completion_tokens:.0f} completion tokens."
        )

        return " ".join(parts)
