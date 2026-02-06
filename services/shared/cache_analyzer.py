"""
Cache Analyzer for Cost Optimization (P2.4.3)

Analyzes prompt patterns and recommends caching opportunities
based on duplicate/similar prompts.

100% internal - uses task_clusterer for similarity detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .task_clusterer import PromptCluster, TaskClusterer

logger = logging.getLogger(__name__)


@dataclass
class CacheRecommendation:
    """A recommendation to implement caching for a cluster of similar prompts"""

    cluster_id: str
    representative_prompt: str
    duplicate_count: int
    frequency_per_day: float
    avg_tokens_per_call: float
    current_monthly_cost: float
    estimated_monthly_savings: float
    estimated_annual_savings: float
    cache_hit_rate: float  # estimated percentage of hits
    reasoning: str
    confidence: float  # 0.0-1.0


class CacheAnalyzer:
    """
    Analyzes prompt patterns and recommends caching strategies.

    Uses TaskClusterer to identify duplicate/similar prompts that could benefit
    from caching.
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_frequency_per_day: float = 2.0,
        min_monthly_savings: float = 5.0,
        cache_overhead_pct: float = 10.0,
    ):
        """
        Initialize the cache analyzer.

        Args:
            min_cluster_size: Minimum duplicates to recommend caching
            min_frequency_per_day: Minimum daily frequency to recommend
            min_monthly_savings: Minimum monthly savings threshold (USD)
            cache_overhead_pct: Estimated overhead for cache operations (percentage)
        """
        self.min_cluster_size = min_cluster_size
        self.min_frequency_per_day = min_frequency_per_day
        self.min_monthly_savings = min_monthly_savings
        self.cache_overhead_pct = cache_overhead_pct
        self.clusterer = TaskClusterer(
            similarity_threshold=0.9,  # High threshold for caching
            min_cluster_size=min_cluster_size,
        )

    def analyze_caching_opportunities(
        self,
        prompts: list[dict[str, Any]],
    ) -> list[CacheRecommendation]:
        """
        Analyze prompts and generate caching recommendations.

        Args:
            prompts: List of dicts with keys: text, tokens, cost, timestamp, model

        Returns:
            List of CacheRecommendation objects, sorted by savings potential
        """
        if not prompts:
            return []

        # Cluster similar prompts
        clusters = self.clusterer.cluster_prompts(prompts)

        if not clusters:
            logger.info("No clusters found for caching analysis")
            return []

        # Generate recommendations from clusters
        recommendations = []
        for cluster in clusters:
            # Skip if doesn't meet criteria
            if cluster.member_count < self.min_cluster_size:
                continue
            if cluster.frequency_per_day < self.min_frequency_per_day:
                continue

            # Calculate cache savings
            savings = self._calculate_cache_savings(cluster)

            if savings["monthly_savings"] < self.min_monthly_savings:
                continue

            # Calculate confidence
            confidence = self._calculate_confidence(cluster)

            # Generate reasoning
            reasoning = self._generate_reasoning(cluster, savings)

            rec = CacheRecommendation(
                cluster_id=cluster.cluster_id,
                representative_prompt=cluster.representative_text,
                duplicate_count=cluster.member_count,
                frequency_per_day=cluster.frequency_per_day,
                avg_tokens_per_call=cluster.avg_tokens,
                current_monthly_cost=self._project_monthly_cost(cluster),
                estimated_monthly_savings=savings["monthly_savings"],
                estimated_annual_savings=savings["annual_savings"],
                cache_hit_rate=savings["hit_rate"],
                reasoning=reasoning,
                confidence=confidence,
            )
            recommendations.append(rec)

        # Sort by savings potential
        return sorted(
            recommendations, key=lambda r: r.estimated_annual_savings, reverse=True
        )

    def _calculate_cache_savings(self, cluster: PromptCluster) -> dict[str, float]:
        """
        Calculate potential savings from caching this cluster.

        Caching saves:
        - Input token processing (100% of prompt tokens)
        - Output token generation (depends on cache implementation)

        For semantic caching:
        - First call: full cost
        - Cache hits: ~90% savings (still need to compute similarity)
        """
        # Estimate cache hit rate (all duplicates would be hits)
        total_calls = cluster.member_count
        cache_hits = total_calls - 1  # First call is miss
        hit_rate = cache_hits / total_calls * 100 if total_calls > 0 else 0

        # Savings per hit (assume 90% cost reduction with semantic cache)
        savings_per_hit = cluster.total_cost / total_calls * 0.90

        # Total savings (accounting for cache overhead)
        gross_savings = savings_per_hit * cache_hits
        cache_overhead = gross_savings * (self.cache_overhead_pct / 100)
        net_savings = gross_savings - cache_overhead

        # Project to monthly/annual
        if cluster.frequency_per_day > 0:
            daily_savings = net_savings / total_calls * cluster.frequency_per_day
        else:
            daily_savings = 0

        monthly_savings = daily_savings * 30
        annual_savings = daily_savings * 365

        return {
            "monthly_savings": max(0.0, monthly_savings),
            "annual_savings": max(0.0, annual_savings),
            "hit_rate": hit_rate,
        }

    def _project_monthly_cost(self, cluster: PromptCluster) -> float:
        """Project current monthly cost for this cluster"""
        if cluster.frequency_per_day > 0:
            calls_per_month = cluster.frequency_per_day * 30
            cost_per_call = cluster.total_cost / cluster.member_count
            return cost_per_call * calls_per_month
        return 0.0

    def _calculate_confidence(self, cluster: PromptCluster) -> float:
        """
        Calculate confidence score for caching recommendation (0.0-1.0).

        Factors:
        - More duplicates = higher confidence
        - Higher frequency = higher confidence
        - More consistent pattern = higher confidence
        """
        confidence = 0.5  # Base confidence

        # Factor 1: Duplicate count (max +0.30)
        if cluster.member_count >= 50:
            confidence += 0.30
        elif cluster.member_count >= 20:
            confidence += 0.25
        elif cluster.member_count >= 10:
            confidence += 0.20
        else:
            confidence += 0.15

        # Factor 2: Frequency (max +0.20)
        if cluster.frequency_per_day >= 10:
            confidence += 0.20
        elif cluster.frequency_per_day >= 5:
            confidence += 0.15
        elif cluster.frequency_per_day >= 2:
            confidence += 0.10

        # Cap at 1.0
        return min(1.0, confidence)

    def _generate_reasoning(
        self, cluster: PromptCluster, savings: dict[str, float]
    ) -> str:
        """Generate human-readable reasoning for recommendation"""
        parts = []

        # Summary
        parts.append(
            f"Detected {cluster.member_count} identical/similar prompts occurring {cluster.frequency_per_day:.1f} times per day."
        )

        # Savings potential
        parts.append(
            f"Caching could save ${savings['monthly_savings']:.2f}/month (${savings['annual_savings']:.2f}/year) "
            f"with an estimated {savings['hit_rate']:.0f}% cache hit rate."
        )

        # Token savings
        parts.append(
            f"Average {cluster.avg_tokens:.0f} tokens per call. "
            f"Cache would eliminate redundant processing for {cluster.member_count - 1} duplicate calls."
        )

        # Implementation suggestion
        if self.clusterer.encoder_available:
            parts.append(
                "Implement semantic caching using embeddings for fuzzy matching of similar prompts."
            )
        else:
            parts.append(
                "Implement exact-match caching using hash-based lookup for identical prompts."
            )

        # Frequency insight
        if cluster.frequency_per_day >= 10:
            parts.append(
                "High frequency pattern detected - caching would have immediate impact."
            )
        elif cluster.frequency_per_day >= 5:
            parts.append("Moderate frequency pattern - caching recommended.")

        return " ".join(parts)

    def estimate_cache_storage(
        self, recommendations: list[CacheRecommendation]
    ) -> dict[str, Any]:
        """
        Estimate storage requirements for implementing caching.

        Returns:
            Dict with storage metrics
        """
        total_unique_prompts = len(recommendations)
        total_avg_tokens = (
            sum(r.avg_tokens_per_call for r in recommendations) / total_unique_prompts
            if total_unique_prompts > 0
            else 0
        )

        # Estimate storage (rough: 1 token ≈ 4 chars ≈ 4 bytes)
        avg_prompt_bytes = total_avg_tokens * 4
        avg_response_bytes = avg_prompt_bytes * 2  # Responses often longer

        # Add embedding storage if using semantic cache (1536 dimensions × 4 bytes)
        embedding_bytes = 1536 * 4 if self.clusterer.encoder_available else 0

        bytes_per_cache_entry = avg_prompt_bytes + avg_response_bytes + embedding_bytes
        total_storage_mb = bytes_per_cache_entry * total_unique_prompts / 1_000_000

        return {
            "unique_prompts": total_unique_prompts,
            "avg_tokens_per_entry": total_avg_tokens,
            "bytes_per_entry": bytes_per_cache_entry,
            "total_storage_mb": total_storage_mb,
            "semantic_cache": self.clusterer.encoder_available,
        }
