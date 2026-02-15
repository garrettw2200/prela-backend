"""
Debug Agent API Router

Provides on-demand trace debugging via LLM analysis.
Checks the analysis_results cache first, then runs the DebugAgent
if no cached result exists.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from shared import get_clickhouse_client, query_spans, settings
from ..auth import require_tier
from ..middleware.ai_feature_limiter import check_ai_feature_limit
from shared.debug_agent import DebugAgent, TRACE_COLUMNS

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------


class TimelineEntryResponse(BaseModel):
    span_id: str
    name: str
    span_type: str
    started_at: str
    duration_ms: float
    status: str
    error_message: str | None = None
    parent_span_id: str | None = None


class FailureChainEntryResponse(BaseModel):
    span_id: str
    name: str
    span_type: str
    error_message: str
    is_root_cause: bool = False


class DebugAnalysisResponse(BaseModel):
    trace_id: str
    root_cause: str
    explanation: str
    fix_suggestions: list[str]
    execution_timeline: list[TimelineEntryResponse]
    failure_chain: list[FailureChainEntryResponse]
    confidence_score: float
    analyzed_at: str
    cached: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/traces/{trace_id}",
    response_model=DebugAnalysisResponse,
)
async def debug_trace(
    trace_id: str,
    project_id: str = Query(..., description="Project ID"),
    force: bool = Query(False, description="Force re-analysis (skip cache)"),
    user: dict = Depends(require_tier("pro")),
) -> DebugAnalysisResponse:
    """Debug a trace — returns root cause analysis and fix suggestions.

    Checks the analysis_results cache first. If no cached result exists
    (or force=true), runs the DebugAgent LLM analysis and caches the result.
    """
    try:
        client = get_clickhouse_client()

        # Check cache unless force refresh requested
        if not force:
            cached = _get_cached_result(client, trace_id, project_id)
            if cached:
                return cached

        # Check and increment debug session usage (only for non-cached analysis)
        await check_ai_feature_limit(user["user_id"], user["tier"], "debug")

        # Fetch trace
        trace_result = client.query(
            "SELECT * FROM traces WHERE trace_id = %(trace_id)s AND project_id = %(project_id)s LIMIT 1",
            parameters={"trace_id": trace_id, "project_id": project_id},
        )

        if not trace_result.result_rows:
            raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

        trace_row = trace_result.result_rows[0]

        # Fetch spans
        spans = await query_spans(client, trace_id)

        if not spans:
            raise HTTPException(status_code=404, detail=f"No spans found for trace {trace_id}")

        # Run debug analysis
        agent = DebugAgent()
        analysis = agent.analyze_trace(
            trace_data=trace_row,
            spans_data=spans,
            model=settings.debug_agent_model,
            max_tokens=settings.debug_agent_max_tokens,
        )

        # Cache result
        _cache_result(client, analysis, project_id)

        return _analysis_to_response(analysis, cached=False)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to debug trace {trace_id}: {e}")
        raise HTTPException(status_code=500, detail="Debug analysis failed")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _get_cached_result(
    client: Any, trace_id: str, project_id: str
) -> DebugAnalysisResponse | None:
    """Check analysis_results for a cached debug analysis."""
    try:
        result = client.query(
            """
            SELECT result, score, created_at
            FROM analysis_results
            WHERE trace_id = %(trace_id)s
              AND project_id = %(project_id)s
              AND analysis_type = 'debug'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            parameters={"trace_id": trace_id, "project_id": project_id},
        )

        if not result.result_rows:
            return None

        row = result.result_rows[0]
        result_data = json.loads(row[0]) if row[0] else None
        if not result_data:
            return None

        return DebugAnalysisResponse(
            trace_id=result_data.get("trace_id", trace_id),
            root_cause=result_data.get("root_cause", ""),
            explanation=result_data.get("explanation", ""),
            fix_suggestions=result_data.get("fix_suggestions", []),
            execution_timeline=[
                TimelineEntryResponse(**e) for e in result_data.get("execution_timeline", [])
            ],
            failure_chain=[
                FailureChainEntryResponse(**e) for e in result_data.get("failure_chain", [])
            ],
            confidence_score=result_data.get("confidence_score", 0.0),
            analyzed_at=result_data.get("analyzed_at", str(row[2])),
            cached=True,
        )

    except Exception as e:
        logger.warning(f"Failed to read debug cache for trace {trace_id}: {e}")
        return None


def _cache_result(client: Any, analysis: Any, project_id: str) -> None:
    """Store debug analysis in analysis_results table."""
    try:
        result_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        data = [
            [
                result_id,
                analysis.trace_id,
                project_id,
                "debug",
                json.dumps(analysis.to_dict()),
                analysis.confidence_score,
                now,
            ]
        ]

        client.insert(
            "analysis_results",
            data,
            column_names=[
                "result_id", "trace_id", "project_id",
                "analysis_type", "result", "score", "created_at",
            ],
        )
    except Exception as e:
        logger.warning(f"Failed to cache debug result for trace {analysis.trace_id}: {e}")


def _analysis_to_response(analysis: Any, cached: bool) -> DebugAnalysisResponse:
    """Convert DebugAnalysis dataclass to Pydantic response model."""
    return DebugAnalysisResponse(
        trace_id=analysis.trace_id,
        root_cause=analysis.root_cause,
        explanation=analysis.explanation,
        fix_suggestions=analysis.fix_suggestions,
        execution_timeline=[
            TimelineEntryResponse(
                span_id=e.span_id,
                name=e.name,
                span_type=e.span_type,
                started_at=e.started_at,
                duration_ms=e.duration_ms,
                status=e.status,
                error_message=e.error_message,
                parent_span_id=e.parent_span_id,
            )
            for e in analysis.execution_timeline
        ],
        failure_chain=[
            FailureChainEntryResponse(
                span_id=e.span_id,
                name=e.name,
                span_type=e.span_type,
                error_message=e.error_message,
                is_root_cause=e.is_root_cause,
            )
            for e in analysis.failure_chain
        ],
        confidence_score=analysis.confidence_score,
        analyzed_at=analysis.analyzed_at,
        cached=cached,
    )
