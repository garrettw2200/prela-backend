"""
Cost Optimization API Router

Provides endpoints for P2.4.1 (Model Downgrades) and P2.4.3 (Caching Recommendations).

100% internal - no external LLM dependencies.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..auth import require_tier

from shared import (
    CacheAnalyzer,
    CacheRecommendation,
    ModelRecommender,
    ModelRecommendation,
    get_clickhouse_client,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Response Models
class ModelRecommendationResponse(BaseModel):
    """Response model for model downgrade recommendations"""

    current_model: str
    recommended_model: str
    confidence: float = Field(ge=0.0, le=1.0)
    estimated_monthly_savings: float
    estimated_annual_savings: float
    performance_impact: str
    latency_change_pct: float
    reasoning: str
    current_call_count: int
    current_avg_latency_ms: float


class CacheRecommendationResponse(BaseModel):
    """Response model for caching recommendations"""

    cluster_id: str
    representative_prompt: str
    duplicate_count: int
    frequency_per_day: float
    avg_tokens_per_call: float
    current_monthly_cost: float
    estimated_monthly_savings: float
    estimated_annual_savings: float
    cache_hit_rate: float
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


class CostAnalyticsResponse(BaseModel):
    """Response model for cost analytics"""

    total_cost_usd: float
    total_tokens: int
    total_calls: int
    avg_cost_per_call: float
    avg_tokens_per_call: float
    cost_by_model: dict[str, float]
    cost_by_vendor: dict[str, float]
    date_range_days: int


# Endpoints


@router.get("/model-recommendations", response_model=dict[str, Any])
async def get_model_recommendations(
    project_id: str = Query(..., description="Project ID"),
    time_window: str = Query("30d", description="Time window (7d, 30d, 90d)"),
    vendor: str = Query("all", description="Filter by vendor (openai, anthropic, all)"),
    user: dict = Depends(require_tier("lunch-money")),
) -> dict[str, Any]:
    """
    Get model downgrade recommendations (P2.4.1).

    Analyzes LLM usage patterns and suggests cheaper alternatives.
    """
    try:
        client = get_clickhouse_client()

        # Parse time window
        days_map = {"7d": 7, "30d": 30, "90d": 90}
        days = days_map.get(time_window, 30)
        since_date = datetime.utcnow() - timedelta(days=days)

        # Query model usage from llm_usage_metrics materialized view
        query = """
        SELECT
            model,
            vendor,
            SUM(call_count) as call_count,
            SUM(total_tokens) as total_tokens,
            SUM(prompt_tokens) as prompt_tokens,
            SUM(completion_tokens) as completion_tokens,
            SUM(total_cost_usd) as total_cost_usd,
            AVG(avg_latency_ms) as avg_latency_ms,
            SUM(call_count) as success_count
        FROM llm_usage_metrics
        WHERE project_id = %(project_id)s
          AND date >= %(since_date)s
          AND (vendor = %(vendor)s OR %(vendor)s = 'all')
        GROUP BY model, vendor
        HAVING call_count > 0
        ORDER BY total_cost_usd DESC
        """

        result = client.query(
            query,
            parameters={
                "project_id": project_id,
                "since_date": since_date.strftime("%Y-%m-%d"),
                "vendor": vendor,
            },
        )

        # Convert to list of dicts
        usage_data = []
        for row in result.result_rows:
            usage_data.append({
                "model": row[0],
                "vendor": row[1],
                "call_count": row[2],
                "total_tokens": row[3],
                "prompt_tokens": row[4],
                "completion_tokens": row[5],
                "total_cost_usd": float(row[6]),
                "avg_latency_ms": float(row[7]),
                "success_count": row[8],
                "date_range_days": days,
            })

        # Analyze and generate recommendations
        recommender = ModelRecommender(
            min_calls_threshold=100,
            min_savings_threshold=10.0,
            latency_tolerance_pct=20.0,
        )

        recommendations = recommender.analyze_model_usage(usage_data)

        # Convert to response format
        recommendations_list = []
        for rec in recommendations:
            recommendations_list.append(
                ModelRecommendationResponse(
                    current_model=rec.current_model,
                    recommended_model=rec.recommended_model,
                    confidence=rec.confidence,
                    estimated_monthly_savings=rec.estimated_monthly_savings,
                    estimated_annual_savings=rec.estimated_annual_savings,
                    performance_impact=rec.performance_impact,
                    latency_change_pct=rec.latency_change_pct,
                    reasoning=rec.reasoning,
                    current_call_count=rec.current_stats.call_count,
                    current_avg_latency_ms=rec.current_stats.avg_latency_ms,
                ).dict()
            )

        total_annual_savings = sum(r.estimated_annual_savings for r in recommendations)

        return {
            "recommendations": recommendations_list,
            "count": len(recommendations_list),
            "total_potential_annual_savings": total_annual_savings,
            "time_window": time_window,
            "project_id": project_id,
        }

    except Exception as e:
        logger.error(f"Failed to get model recommendations: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to analyze model usage: {str(e)}"
        )


@router.get("/cache-recommendations", response_model=dict[str, Any])
async def get_cache_recommendations(
    project_id: str = Query(..., description="Project ID"),
    time_window: str = Query("30d", description="Time window (7d, 30d, 90d)"),
    min_cluster_size: int = Query(5, description="Minimum duplicate prompts"),
    user: dict = Depends(require_tier("lunch-money")),
) -> dict[str, Any]:
    """
    Get caching recommendations (P2.4.3).

    Identifies duplicate/similar prompts that could benefit from caching.
    """
    try:
        client = get_clickhouse_client()

        # Parse time window
        days_map = {"7d": 7, "30d": 30, "90d": 90}
        days = days_map.get(time_window, 30)
        since_date = datetime.utcnow() - timedelta(days=days)

        # Query prompts from spans table
        query = """
        SELECT
            JSONExtractString(attributes, 'llm.prompt') as prompt_text,
            JSONExtractUInt(attributes, 'llm.total_tokens') as tokens,
            JSONExtractFloat(attributes, 'llm.cost_usd') as cost,
            started_at as timestamp,
            JSONExtractString(attributes, 'llm.model') as model
        FROM spans
        WHERE project_id = %(project_id)s
          AND span_type = 'llm'
          AND started_at >= %(since_date)s
          AND JSONExtractString(attributes, 'llm.prompt') != ''
          AND JSONExtractUInt(attributes, 'llm.total_tokens') > 0
        ORDER BY started_at DESC
        LIMIT 10000
        """

        result = client.query(
            query,
            parameters={
                "project_id": project_id,
                "since_date": since_date.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

        # Convert to prompt list
        prompts = []
        for row in result.result_rows:
            prompt_text = row[0]
            tokens = row[1]
            cost = float(row[2]) if row[2] else 0.0
            timestamp = row[3]
            model = row[4]

            if prompt_text and tokens > 0:
                prompts.append({
                    "text": prompt_text,
                    "tokens": tokens,
                    "cost": cost,
                    "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                    "model": model,
                })

        if not prompts:
            return {
                "recommendations": [],
                "count": 0,
                "total_potential_annual_savings": 0.0,
                "time_window": time_window,
                "project_id": project_id,
                "message": "No prompts found in time window",
            }

        # Analyze and generate recommendations
        analyzer = CacheAnalyzer(
            min_cluster_size=min_cluster_size,
            min_frequency_per_day=2.0,
            min_monthly_savings=5.0,
        )

        recommendations = analyzer.analyze_caching_opportunities(prompts)

        # Convert to response format
        recommendations_list = []
        for rec in recommendations:
            recommendations_list.append(
                CacheRecommendationResponse(
                    cluster_id=rec.cluster_id,
                    representative_prompt=rec.representative_prompt,
                    duplicate_count=rec.duplicate_count,
                    frequency_per_day=rec.frequency_per_day,
                    avg_tokens_per_call=rec.avg_tokens_per_call,
                    current_monthly_cost=rec.current_monthly_cost,
                    estimated_monthly_savings=rec.estimated_monthly_savings,
                    estimated_annual_savings=rec.estimated_annual_savings,
                    cache_hit_rate=rec.cache_hit_rate,
                    reasoning=rec.reasoning,
                    confidence=rec.confidence,
                ).dict()
            )

        total_annual_savings = sum(r.estimated_annual_savings for r in recommendations)

        # Get storage estimates
        storage_estimate = analyzer.estimate_cache_storage(recommendations)

        return {
            "recommendations": recommendations_list,
            "count": len(recommendations_list),
            "total_potential_annual_savings": total_annual_savings,
            "storage_estimate": storage_estimate,
            "time_window": time_window,
            "project_id": project_id,
            "prompts_analyzed": len(prompts),
        }

    except Exception as e:
        logger.error(f"Failed to get cache recommendations: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to analyze caching opportunities: {str(e)}"
        )


@router.get("/analytics", response_model=dict[str, Any])
async def get_cost_analytics(
    project_id: str = Query(..., description="Project ID"),
    time_window: str = Query("30d", description="Time window (7d, 30d, 90d)"),
    user: dict = Depends(require_tier("lunch-money")),
) -> dict[str, Any]:
    """
    Get cost analytics overview.

    Provides high-level cost metrics and trends.
    """
    try:
        client = get_clickhouse_client()

        # Parse time window
        days_map = {"7d": 7, "30d": 30, "90d": 90}
        days = days_map.get(time_window, 30)
        since_date = datetime.utcnow() - timedelta(days=days)

        # Query cost metrics
        query = """
        SELECT
            SUM(call_count) as total_calls,
            SUM(total_tokens) as total_tokens,
            SUM(total_cost_usd) as total_cost
        FROM llm_usage_metrics
        WHERE project_id = %(project_id)s
          AND date >= %(since_date)s
        """

        result = client.query(
            query,
            parameters={
                "project_id": project_id,
                "since_date": since_date.strftime("%Y-%m-%d"),
            },
        )

        row = result.result_rows[0] if result.result_rows else (0, 0, 0.0)
        total_calls = row[0]
        total_tokens = row[1]
        total_cost = float(row[2])

        # Query cost by model
        query_by_model = """
        SELECT
            model,
            SUM(total_cost_usd) as cost
        FROM llm_usage_metrics
        WHERE project_id = %(project_id)s
          AND date >= %(since_date)s
        GROUP BY model
        ORDER BY cost DESC
        """

        result_by_model = client.query(
            query_by_model,
            parameters={
                "project_id": project_id,
                "since_date": since_date.strftime("%Y-%m-%d"),
            },
        )

        cost_by_model = {row[0]: float(row[1]) for row in result_by_model.result_rows}

        # Query cost by vendor
        query_by_vendor = """
        SELECT
            vendor,
            SUM(total_cost_usd) as cost
        FROM llm_usage_metrics
        WHERE project_id = %(project_id)s
          AND date >= %(since_date)s
        GROUP BY vendor
        ORDER BY cost DESC
        """

        result_by_vendor = client.query(
            query_by_vendor,
            parameters={
                "project_id": project_id,
                "since_date": since_date.strftime("%Y-%m-%d"),
            },
        )

        cost_by_vendor = {row[0]: float(row[1]) for row in result_by_vendor.result_rows}

        return CostAnalyticsResponse(
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            total_calls=total_calls,
            avg_cost_per_call=total_cost / total_calls if total_calls > 0 else 0.0,
            avg_tokens_per_call=total_tokens / total_calls if total_calls > 0 else 0.0,
            cost_by_model=cost_by_model,
            cost_by_vendor=cost_by_vendor,
            date_range_days=days,
        ).dict()

    except Exception as e:
        logger.error(f"Failed to get cost analytics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get cost analytics: {str(e)}"
        )
