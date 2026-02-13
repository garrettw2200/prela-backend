"""
Insights Dashboard API Router

Aggregates existing analyzers into a unified health score, top issues,
and cost insights view for the Analysis-First Dashboard (Sprint 3).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from shared import ModelRecommender, get_clickhouse_client
from shared.error_analyzer import ErrorAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter()


# Response Models


class HealthTrendPoint(BaseModel):
    date: str
    score: float


class TopIssue(BaseModel):
    category: str
    severity: str
    count: int
    latest_trace_id: str
    recommendation: str


class CostInsights(BaseModel):
    total_cost_usd: float
    total_calls: int
    potential_monthly_savings: float
    top_saving_opportunity: str
    cost_by_model: dict[str, float]


class SecuritySummaryData(BaseModel):
    total_findings: int
    by_severity: dict[str, int]
    by_type: dict[str, int]


class InsightsSummaryResponse(BaseModel):
    health_score: float = Field(ge=0, le=100)
    health_trend: list[HealthTrendPoint]
    top_issues: list[TopIssue]
    cost_insights: CostInsights
    security_summary: SecuritySummaryData
    trace_count: int
    error_rate: float
    time_window: str


# Helpers


def _compute_health_score(
    error_rate: float,
    cost_waste_pct: float,
    security_incident_rate: float = 0.0,
) -> float:
    """Compute health score from available metrics.

    Formula: 100 - (error_rate * 30) - (hallucination_rate * 25)
             - (cost_waste_pct * 15) - (latency_drift_pct * 15)
             - (security_incident_rate * 15)

    Sprint 5 (drift) term defaults to 0.
    Hallucination rate defaults to 0 (too expensive to compute on every request).
    """
    score = (
        100.0
        - (error_rate * 30)
        - (cost_waste_pct * 15)
        - (security_incident_rate * 15)
    )
    return max(0.0, min(100.0, round(score, 1)))


# Endpoint


@router.get("/summary", response_model=InsightsSummaryResponse)
async def get_insights_summary(
    project_id: str = Query(..., description="Project ID"),
    time_window: str = Query("7d", description="Time window (7d, 30d, 90d)"),
) -> InsightsSummaryResponse:
    """
    Get aggregated insights summary for the dashboard.

    Combines trace stats, error analysis, and cost optimization into
    a single response with health score, trend, top issues, and cost insights.
    """
    days_map = {"7d": 7, "30d": 30, "90d": 90}
    days = days_map.get(time_window, 7)
    since_date = datetime.utcnow() - timedelta(days=days)
    since_str = since_date.strftime("%Y-%m-%d %H:%M:%S")

    try:
        client = get_clickhouse_client()

        # --- Query 1: Trace stats (count + error rate) ---
        trace_stats = client.query(
            """
            SELECT
                count() AS total,
                countIf(status = 'error') AS errors
            FROM traces
            WHERE project_id = %(project_id)s
              AND started_at >= %(since)s
            """,
            parameters={"project_id": project_id, "since": since_str},
        )
        row = trace_stats.result_rows[0] if trace_stats.result_rows else (0, 0)
        trace_count = int(row[0])
        error_count = int(row[1])
        error_rate = error_count / trace_count if trace_count > 0 else 0.0

        # --- Query 2: Daily error rate (for health trend) ---
        daily_stats = client.query(
            """
            SELECT
                toDate(started_at) AS date,
                count() AS total,
                countIf(status = 'error') AS errors
            FROM traces
            WHERE project_id = %(project_id)s
              AND started_at >= %(since)s
            GROUP BY date
            ORDER BY date ASC
            """,
            parameters={"project_id": project_id, "since": since_str},
        )

        # --- Query 3: Cost by model (from materialized view) ---
        cost_result = client.query(
            """
            SELECT
                model,
                vendor,
                SUM(call_count) AS total_calls,
                SUM(total_tokens) AS total_tokens,
                SUM(prompt_tokens) AS prompt_tokens,
                SUM(completion_tokens) AS completion_tokens,
                SUM(total_cost_usd) AS total_cost_usd,
                AVG(avg_latency_ms) AS avg_latency_ms,
                SUM(call_count) AS success_count
            FROM llm_usage_metrics
            WHERE project_id = %(project_id)s
              AND date >= %(since_date)s
            GROUP BY model, vendor
            HAVING total_calls > 0
            ORDER BY total_cost_usd DESC
            """,
            parameters={
                "project_id": project_id,
                "since_date": since_date.strftime("%Y-%m-%d"),
            },
        )

        total_cost = 0.0
        total_calls = 0
        cost_by_model: dict[str, float] = {}
        usage_data: list[dict[str, Any]] = []

        for r in cost_result.result_rows:
            model_name = r[0]
            cost = float(r[6])
            calls = int(r[2])
            total_cost += cost
            total_calls += calls
            cost_by_model[model_name] = round(cost, 2)
            usage_data.append({
                "model": r[0],
                "vendor": r[1],
                "call_count": int(r[2]),
                "total_tokens": int(r[3]),
                "prompt_tokens": int(r[4]),
                "completion_tokens": int(r[5]),
                "total_cost_usd": float(r[6]),
                "avg_latency_ms": float(r[7]),
                "success_count": int(r[8]),
                "date_range_days": days,
            })

        # --- Query 4: Model recommendations ---
        potential_monthly_savings = 0.0
        top_saving_opportunity = "No optimization opportunities found"

        if usage_data:
            recommender = ModelRecommender(
                min_calls_threshold=50,
                min_savings_threshold=5.0,
                latency_tolerance_pct=20.0,
            )
            recommendations = recommender.analyze_model_usage(usage_data)
            if recommendations:
                potential_monthly_savings = sum(
                    r.estimated_monthly_savings for r in recommendations
                )
                best = recommendations[0]
                top_saving_opportunity = (
                    f"Switch {best.current_model} to {best.recommended_model} "
                    f"for ${best.estimated_monthly_savings:.0f}/mo savings"
                )

        # --- Query 5: Top error spans for issue categorization ---
        error_spans_result = client.query(
            """
            SELECT
                span_id,
                trace_id,
                name,
                span_type,
                attributes,
                events
            FROM spans
            WHERE project_id = %(project_id)s
              AND status = 'error'
              AND started_at >= %(since)s
            ORDER BY started_at DESC
            LIMIT 50
            """,
            parameters={"project_id": project_id, "since": since_str},
        )

        # Analyze errors and aggregate by category
        category_counts: Counter[str] = Counter()
        category_severity: dict[str, str] = {}
        category_trace_id: dict[str, str] = {}
        category_recommendation: dict[str, str] = {}

        for r in error_spans_result.result_rows:
            span_data = {
                "span_id": r[0],
                "name": r[2],
                "span_type": r[3],
                "attributes": json.loads(r[4]) if r[4] else {},
                "events": json.loads(r[5]) if r[5] else [],
            }
            try:
                analysis = ErrorAnalyzer.analyze_span_error(span_data)
                cat = analysis.category.value
                category_counts[cat] += 1
                # Keep highest severity and most recent trace_id per category
                if cat not in category_severity:
                    category_severity[cat] = analysis.severity.value
                    category_trace_id[cat] = r[1]
                    if analysis.recommendations:
                        category_recommendation[cat] = analysis.recommendations[0].title
                    else:
                        category_recommendation[cat] = "Review error details"
            except Exception:
                category_counts["unknown"] += 1

        top_issues: list[TopIssue] = []
        for cat, count in category_counts.most_common(5):
            top_issues.append(
                TopIssue(
                    category=cat,
                    severity=category_severity.get(cat, "MEDIUM"),
                    count=count,
                    latest_trace_id=category_trace_id.get(cat, ""),
                    recommendation=category_recommendation.get(cat, "Review error details"),
                )
            )

        # --- Query 6: Security findings from analysis_results ---
        security_by_severity: Counter[str] = Counter()
        security_by_type: Counter[str] = Counter()

        try:
            security_results = client.query(
                """
                SELECT result
                FROM analysis_results
                WHERE project_id = %(project_id)s
                  AND analysis_type = 'security'
                  AND created_at >= %(since)s
                LIMIT 200
                """,
                parameters={"project_id": project_id, "since": since_str},
            )

            for r in security_results.result_rows:
                try:
                    result_data = json.loads(r[0]) if r[0] else {}
                except (json.JSONDecodeError, TypeError):
                    continue
                for finding in result_data.get("findings", []):
                    security_by_severity[finding.get("severity", "MEDIUM")] += 1
                    security_by_type[finding.get("finding_type", "unknown")] += 1
        except Exception as sec_err:
            logger.warning(f"Security query failed (table may not exist yet): {sec_err}")

        security_total = sum(security_by_severity.values())

        # --- Compute health trend ---
        cost_waste_pct = (
            (potential_monthly_savings / (total_cost * 30 / days)) * 100
            if total_cost > 0 and days > 0
            else 0.0
        )

        # Security incident rate: findings / total LLM spans (capped at 1.0)
        security_incident_rate = min(security_total / trace_count, 1.0) if trace_count > 0 else 0.0

        health_trend: list[HealthTrendPoint] = []
        for r in daily_stats.result_rows:
            day_date = str(r[0])
            day_total = int(r[1])
            day_errors = int(r[2])
            day_error_rate = day_errors / day_total if day_total > 0 else 0.0
            day_score = _compute_health_score(day_error_rate, cost_waste_pct, security_incident_rate)
            health_trend.append(HealthTrendPoint(date=day_date, score=day_score))

        health_score = _compute_health_score(error_rate, cost_waste_pct, security_incident_rate)

        return InsightsSummaryResponse(
            health_score=health_score,
            health_trend=health_trend,
            top_issues=top_issues,
            cost_insights=CostInsights(
                total_cost_usd=round(total_cost, 2),
                total_calls=total_calls,
                potential_monthly_savings=round(potential_monthly_savings, 2),
                top_saving_opportunity=top_saving_opportunity,
                cost_by_model=cost_by_model,
            ),
            security_summary=SecuritySummaryData(
                total_findings=security_total,
                by_severity=dict(security_by_severity),
                by_type=dict(security_by_type),
            ),
            trace_count=trace_count,
            error_rate=round(error_rate, 4),
            time_window=time_window,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get insights summary: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate insights: {str(e)}"
        )
