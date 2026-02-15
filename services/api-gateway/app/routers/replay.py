"""
Replay API Router

Provides endpoints for managing and executing trace replays.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..auth import require_tier, check_project_access_or_403
from shared import get_clickhouse_client
from shared.validation import InputValidator
from shared.utils import safe_json_parse

# Add SDK to Python path for replay functionality
sdk_path = Path(__file__).parent.parent.parent.parent.parent / "sdk"
if str(sdk_path) not in sys.path:
    sys.path.insert(0, str(sdk_path))

router = APIRouter()


# ============================================================================
# Helper Functions
# ============================================================================


async def load_trace_from_clickhouse(trace_id: str) -> Any:
    """Load trace and spans from ClickHouse and convert to Span objects."""
    try:
        from prela.core import Span, SpanType, SpanStatus
        from prela.replay.loader import TraceLoader
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"ReplayEngine not available: {e}. Install prela SDK in API Gateway.",
        )

    client = get_clickhouse_client()

    # Query trace
    trace_query = """
        SELECT trace_id, service_name, started_at, duration_ms, status, attributes
        FROM traces
        WHERE trace_id = %(trace_id)s
    """
    trace_result = client.query(trace_query, {"trace_id": trace_id})

    if not trace_result.result_rows:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

    # Query spans
    spans_query = """
        SELECT span_id, parent_span_id, name, span_type, started_at, ended_at,
               duration_ms, status, attributes, events, replay_snapshot
        FROM spans
        WHERE trace_id = %(trace_id)s
        ORDER BY started_at ASC
    """
    spans_result = client.query(spans_query, {"trace_id": trace_id})

    # Convert to Span objects
    span_objects = []
    for row in spans_result.result_rows:
        # Parse attributes and events with safe parsing
        attributes = safe_json_parse(row[8], default={}, field_name="span.attributes")
        events_data = safe_json_parse(row[9], default=[], field_name="span.events")
        replay_snapshot = safe_json_parse(row[10], default=None, field_name="span.replay_snapshot")

        # Create span
        span = Span(
            span_id=row[0],
            trace_id=trace_id,
            parent_span_id=row[1] if row[1] else None,
            name=row[2],
            span_type=SpanType(row[3]),
            started_at=row[4],
            ended_at=row[5] if row[5] else None,
            status=SpanStatus(row[7]) if row[7] else SpanStatus.PENDING,
            attributes=attributes,
        )

        # Set replay_snapshot if available
        if replay_snapshot:
            object.__setattr__(span, "replay_snapshot", replay_snapshot)

        span_objects.append(span)

    # Use TraceLoader to build trace tree
    trace = TraceLoader.from_spans(span_objects)
    return trace


async def execute_replay_in_background(
    execution_id: str,
    trace_id: str,
    project_id: str,
    parameters: dict[str, Any],
    batch_id: str | None = None,
) -> None:
    """Execute replay in background and store results in ClickHouse."""
    client = get_clickhouse_client()

    try:
        from prela.replay import ReplayEngine, compare_replays

        # Insert execution record with status running
        client.command(
            """
            INSERT INTO replay_executions (
                execution_id, trace_id, project_id, triggered_at, status, parameters, batch_id
            ) VALUES (
                %(execution_id)s, %(trace_id)s, %(project_id)s, now64(6), 'running', %(parameters)s, %(batch_id)s
            )
            """,
            {
                "execution_id": execution_id,
                "trace_id": trace_id,
                "project_id": project_id,
                "parameters": json.dumps(parameters),
                "batch_id": batch_id,
            },
        )

        # Load trace
        trace = await load_trace_from_clickhouse(trace_id)

        # Execute replay
        engine = ReplayEngine(trace)

        # Get original result (exact replay for comparison)
        original_result = engine.replay_exact()

        # Execute modified replay
        modified_result = engine.replay_with_modifications(
            model=parameters.get("model"),
            temperature=parameters.get("temperature"),
            system_prompt=parameters.get("system_prompt"),
            max_tokens=parameters.get("max_tokens"),
            stream=parameters.get("stream", False),
        )

        # Compare results
        comparison = compare_replays(original_result, modified_result)

        # Store results
        result_data = {
            "total_duration_ms": modified_result.total_duration_ms,
            "total_tokens": modified_result.total_tokens,
            "total_cost_usd": modified_result.total_cost_usd,
            "span_count": len(modified_result.spans),
        }

        comparison_data = {
            "original": {
                "total_duration_ms": original_result.total_duration_ms,
                "total_tokens": original_result.total_tokens,
                "total_cost_usd": original_result.total_cost_usd,
            },
            "modified": {
                "total_duration_ms": modified_result.total_duration_ms,
                "total_tokens": modified_result.total_tokens,
                "total_cost_usd": modified_result.total_cost_usd,
            },
            "differences": [
                {
                    "span_name": d.span_name,
                    "field": d.field,
                    "original_value": str(d.original_value)[:500],  # Truncate
                    "modified_value": str(d.modified_value)[:500],
                    "semantic_similarity": d.semantic_similarity,
                }
                for d in comparison.differences
            ],
            "summary": comparison.generate_summary(),
            "semantic_similarity_available": comparison.semantic_similarity_available,
        }

        # Update with completion
        client.command(
            """
            ALTER TABLE replay_executions
            UPDATE
                status = 'completed',
                completed_at = now64(6),
                result = %(result)s,
                comparison = %(comparison)s
            WHERE execution_id = %(execution_id)s
            """,
            {
                "execution_id": execution_id,
                "result": json.dumps(result_data),
                "comparison": json.dumps(comparison_data),
            },
        )

    except Exception as e:
        # Store error
        client.command(
            """
            ALTER TABLE replay_executions
            UPDATE
                status = 'failed',
                completed_at = now64(6),
                error = %(error)s
            WHERE execution_id = %(execution_id)s
            """,
            {"execution_id": execution_id, "error": str(e)},
        )


def _build_batch_summary(
    comparisons: list[dict[str, Any]],
    completed: int,
    failed: int,
    total: int,
) -> dict[str, Any]:
    """Build aggregated summary from individual replay comparisons."""
    if not comparisons:
        return {"completed": completed, "failed": failed, "total": total}

    total_original_cost = sum(
        c.get("original", {}).get("total_cost_usd", 0) for c in comparisons
    )
    total_modified_cost = sum(
        c.get("modified", {}).get("total_cost_usd", 0) for c in comparisons
    )
    total_original_tokens = sum(
        c.get("original", {}).get("total_tokens", 0) for c in comparisons
    )
    total_modified_tokens = sum(
        c.get("modified", {}).get("total_tokens", 0) for c in comparisons
    )
    avg_original_duration = (
        sum(c.get("original", {}).get("total_duration_ms", 0) for c in comparisons)
        / len(comparisons)
    )
    avg_modified_duration = (
        sum(c.get("modified", {}).get("total_duration_ms", 0) for c in comparisons)
        / len(comparisons)
    )

    return {
        "completed": completed,
        "failed": failed,
        "total": total,
        "cost_delta_usd": total_modified_cost - total_original_cost,
        "token_delta": total_modified_tokens - total_original_tokens,
        "avg_duration_delta_ms": avg_modified_duration - avg_original_duration,
        "total_original_cost_usd": total_original_cost,
        "total_modified_cost_usd": total_modified_cost,
    }


async def execute_batch_replay_in_background(
    batch_id: str,
    trace_ids: list[str],
    project_id: str,
    parameters: dict[str, Any],
) -> None:
    """Execute batch replay with concurrency control."""
    import logging

    logger = logging.getLogger(__name__)
    client = get_clickhouse_client()
    semaphore = asyncio.Semaphore(3)  # Max 3 concurrent replays

    # Update batch status to running
    client.command(
        """
        ALTER TABLE batch_replay_jobs
        UPDATE status = 'running', started_at = now64(6)
        WHERE batch_id = %(batch_id)s
        """,
        {"batch_id": batch_id},
    )

    completed = 0
    failed = 0
    all_comparisons: list[dict[str, Any]] = []

    async def run_single(trace_id: str) -> None:
        nonlocal completed, failed
        async with semaphore:
            execution_id = str(uuid.uuid4())
            try:
                await execute_replay_in_background(
                    execution_id=execution_id,
                    trace_id=trace_id,
                    project_id=project_id,
                    parameters=parameters,
                    batch_id=batch_id,
                )
                # Check if it succeeded by reading the record
                # Small delay to allow ClickHouse mutation to process
                await asyncio.sleep(1)
                result = client.query(
                    """
                    SELECT status, comparison FROM replay_executions
                    WHERE execution_id = %(eid)s
                    """,
                    {"eid": execution_id},
                )
                if result.result_rows and result.result_rows[0][0] == "completed":
                    completed += 1
                    comparison_data = safe_json_parse(
                        result.result_rows[0][1], default=None, field_name="comparison"
                    )
                    if comparison_data:
                        all_comparisons.append(comparison_data)
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Batch replay failed for trace {trace_id}: {e}")
                failed += 1

    # Run all replays with concurrency control
    tasks = [run_single(trace_id) for trace_id in trace_ids]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Build batch summary
    summary = _build_batch_summary(all_comparisons, completed, failed, len(trace_ids))

    # Determine final status
    if failed == 0:
        final_status = "completed"
    elif completed == 0:
        final_status = "failed"
    else:
        final_status = "partial"

    # Update batch job with final results
    client.command(
        """
        ALTER TABLE batch_replay_jobs
        UPDATE
            status = %(status)s,
            completed_traces = %(completed)s,
            failed_traces = %(failed)s,
            completed_at = now64(6),
            summary = %(summary)s
        WHERE batch_id = %(batch_id)s
        """,
        {
            "batch_id": batch_id,
            "status": final_status,
            "completed": completed,
            "failed": failed,
            "summary": json.dumps(summary),
        },
    )


# ============================================================================
# Pydantic Models
# ============================================================================


class ReplayCapableTrace(BaseModel):
    """Trace with replay capability information."""

    trace_id: str
    service_name: str
    started_at: str
    duration_ms: float
    total_tokens: int | None = None
    total_cost_usd: float | None = None
    has_replay_snapshot: bool
    span_count: int
    llm_span_count: int


class ReplayTracesResponse(BaseModel):
    """Response for listing replay-capable traces."""

    traces: list[ReplayCapableTrace]
    total: int
    page: int
    page_size: int


class ReplayTraceDetail(BaseModel):
    """Detailed trace with replay snapshot data."""

    trace_id: str
    service_name: str
    started_at: str
    duration_ms: float
    status: str
    attributes: dict[str, Any]
    spans: list[dict[str, Any]]
    replay_snapshot_summary: dict[str, Any]


class ReplayParameters(BaseModel):
    """Parameters for replay execution."""

    model: str | None = None
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    system_prompt: str | None = None
    max_tokens: int | None = Field(None, gt=0)
    stream: bool = False


class ReplayExecutionRequest(BaseModel):
    """Request to trigger replay execution."""

    trace_id: str
    parameters: ReplayParameters = Field(default_factory=ReplayParameters)


class ReplayExecutionResponse(BaseModel):
    """Response after triggering replay."""

    execution_id: str
    status: str
    started_at: str


class ReplayExecutionStatus(BaseModel):
    """Status of replay execution."""

    execution_id: str
    trace_id: str
    status: str
    started_at: str
    completed_at: str | None = None
    parameters: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None


class SpanDifference(BaseModel):
    """Difference between two span executions."""

    span_name: str
    field: str
    original_value: Any
    modified_value: Any
    semantic_similarity: float | None = None


class ReplayComparisonResponse(BaseModel):
    """Comparison results between original and replayed execution."""

    execution_id: str
    original: dict[str, Any]
    modified: dict[str, Any]
    differences: list[SpanDifference]
    summary: str
    semantic_similarity_available: bool


class ReplayHistoryItem(BaseModel):
    """Replay execution history item."""

    execution_id: str
    trace_id: str
    triggered_at: str
    completed_at: str | None
    status: str
    parameters: dict[str, Any]
    duration_ms: float | None = None
    cost_delta: float | None = None


class ReplayHistoryResponse(BaseModel):
    """Response for replay execution history."""

    executions: list[ReplayHistoryItem]
    total: int
    page: int
    page_size: int


# ============================================================================
# Batch Replay Models
# ============================================================================


class BatchReplayRequest(BaseModel):
    """Request to trigger batch replay for multiple traces."""

    trace_ids: list[str] = Field(..., min_length=1, max_length=50)
    parameters: ReplayParameters = Field(default_factory=ReplayParameters)


class BatchReplayResponse(BaseModel):
    """Response after triggering batch replay."""

    batch_id: str
    status: str
    total_traces: int
    created_at: str


class BatchStatusResponse(BaseModel):
    """Status of a batch replay job."""

    batch_id: str
    project_id: str
    status: str
    trace_ids: list[str]
    parameters: dict[str, Any]
    total_traces: int
    completed_traces: int
    failed_traces: int
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    summary: dict[str, Any] | None = None
    executions: list[ReplayHistoryItem] | None = None


class BatchListItem(BaseModel):
    """Summary of a batch replay job for list view."""

    batch_id: str
    status: str
    total_traces: int
    completed_traces: int
    failed_traces: int
    created_at: str
    completed_at: str | None = None
    parameters: dict[str, Any]


class BatchListResponse(BaseModel):
    """Response for listing batch replay jobs."""

    batches: list[BatchListItem]
    total: int
    page: int
    page_size: int


# ============================================================================
# API Endpoints
# ============================================================================


@router.get("/traces", response_model=ReplayTracesResponse)
async def list_replay_traces(
    project_id: str = Query(..., description="Project ID to filter traces"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    since: str | None = Query(None, description="ISO 8601 timestamp to filter traces"),
    user: dict = Depends(require_tier("lunch-money")),
) -> ReplayTracesResponse:
    """
    List traces with replay capability.

    Returns traces that have replay_snapshot data available.
    """
    client = get_clickhouse_client()

    # Validate inputs to prevent SQL injection
    project_id = InputValidator.validate_project_id(project_id)
    page, page_size = InputValidator.validate_pagination(page, page_size)

    # Enforce team-scoped project access
    await check_project_access_or_403(str(user.get("user_id", "")), project_id)

    # Build parameterized WHERE clause
    where_conditions = ["service_name LIKE %(project_pattern)s"]
    params = {"project_pattern": f"%{project_id}%"}

    if since:
        # Validate timestamp format
        since = InputValidator.validate_timestamp(since, "since")
        where_conditions.append("started_at >= %(since)s")
        params["since"] = since

    where_clause = " AND ".join(where_conditions)

    # Count total with parameterized query
    count_query = f"""
        SELECT count() as total
        FROM traces
        WHERE {where_clause}
    """

    count_result = client.query(count_query, parameters=params)
    total = count_result.result_rows[0][0] if count_result.result_rows else 0

    # Query traces with replay capability
    offset = (page - 1) * page_size

    # Add pagination parameters
    params["page_size"] = page_size
    params["offset"] = offset

    query = f"""
        SELECT
            trace_id,
            service_name,
            started_at,
            duration_ms,
            JSONExtractInt(attributes, 'total_tokens') as total_tokens,
            JSONExtractFloat(attributes, 'total_cost_usd') as total_cost_usd,
            1 as has_replay_snapshot,
            (SELECT count() FROM spans WHERE spans.trace_id = traces.trace_id) as span_count,
            (SELECT count() FROM spans WHERE spans.trace_id = traces.trace_id AND span_type = 'llm') as llm_span_count
        FROM traces
        WHERE {where_clause}
        ORDER BY started_at DESC
        LIMIT %(page_size)s OFFSET %(offset)s
    """

    result = client.query(query, parameters=params)

    traces = []
    for row in result.result_rows:
        traces.append(
            ReplayCapableTrace(
                trace_id=row[0],
                service_name=row[1],
                started_at=row[2].isoformat() if isinstance(row[2], datetime) else str(row[2]),
                duration_ms=float(row[3]) if row[3] else 0.0,
                total_tokens=int(row[4]) if row[4] else None,
                total_cost_usd=float(row[5]) if row[5] else None,
                has_replay_snapshot=bool(row[6]),
                span_count=int(row[7]) if row[7] else 0,
                llm_span_count=int(row[8]) if row[8] else 0,
            )
        )

    return ReplayTracesResponse(
        traces=traces,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/traces/{trace_id}", response_model=ReplayTraceDetail)
async def get_replay_trace_detail(trace_id: str, user: dict = Depends(require_tier("lunch-money"))) -> ReplayTraceDetail:
    """
    Get detailed trace information with replay snapshot data.
    """
    client = get_clickhouse_client()

    # Validate trace_id format
    trace_id = InputValidator.validate_uuid(trace_id, "trace_id")

    # Query trace (include project_id for access check)
    trace_query = """
        SELECT trace_id, service_name, started_at, duration_ms, status, attributes, project_id
        FROM traces
        WHERE trace_id = %(trace_id)s
    """

    trace_result = client.query(trace_query, {"trace_id": trace_id})

    if not trace_result.result_rows:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

    trace_row = trace_result.result_rows[0]

    # Enforce team-scoped project access
    trace_project_id = trace_row[6]
    if trace_project_id:
        await check_project_access_or_403(str(user.get("user_id", "")), trace_project_id)

    # Query spans
    spans_query = """
        SELECT span_id, name, span_type, started_at, ended_at, duration_ms,
               status, attributes, events, replay_snapshot
        FROM spans
        WHERE trace_id = %(trace_id)s
        ORDER BY started_at ASC
    """

    spans_result = client.query(spans_query, {"trace_id": trace_id})

    spans = []
    replay_snapshots = []

    for span_row in spans_result.result_rows:
        span = {
            "span_id": span_row[0],
            "name": span_row[1],
            "span_type": span_row[2],
            "started_at": span_row[3].isoformat() if isinstance(span_row[3], datetime) else str(span_row[3]),
            "ended_at": span_row[4].isoformat() if isinstance(span_row[4], datetime) else str(span_row[4]),
            "duration_ms": float(span_row[5]) if span_row[5] else 0.0,
            "status": span_row[6],
            "attributes": safe_json_parse(span_row[7], default={}, field_name="span.attributes"),
            "events": safe_json_parse(span_row[8], default=[], field_name="span.events"),
        }
        spans.append(span)

        # Extract replay snapshot with safe parsing
        if span_row[9]:
            replay_snapshot = safe_json_parse(span_row[9], default=None, field_name="span.replay_snapshot")
            if replay_snapshot:
                replay_snapshots.append(replay_snapshot)

    # Build replay snapshot summary
    replay_snapshot_summary = {
        "available": len(replay_snapshots) > 0,
        "snapshot_count": len(replay_snapshots),
        "models": list({s.get("llm_request", {}).get("model") for s in replay_snapshots if s.get("llm_request", {}).get("model")}),
    }

    return ReplayTraceDetail(
        trace_id=trace_row[0],
        service_name=trace_row[1],
        started_at=trace_row[2].isoformat() if isinstance(trace_row[2], datetime) else str(trace_row[2]),
        duration_ms=float(trace_row[3]) if trace_row[3] else 0.0,
        status=trace_row[4],
        attributes=safe_json_parse(trace_row[5], default={}, field_name="trace.attributes"),
        spans=spans,
        replay_snapshot_summary=replay_snapshot_summary,
    )


@router.post("/execute", response_model=ReplayExecutionResponse)
async def execute_replay(
    request: ReplayExecutionRequest, background_tasks: BackgroundTasks, user: dict = Depends(require_tier("lunch-money"))
) -> ReplayExecutionResponse:
    """
    Trigger a replay execution.

    This endpoint accepts a trace_id and parameters, then initiates
    a replay execution. The actual replay happens asynchronously in the background.
    """
    execution_id = str(uuid.uuid4())
    started_at = datetime.utcnow()

    # Enforce monthly replay limit for Lunch Money tier
    if user.get("tier") == "lunch-money":
        client = get_clickhouse_client()
        count_query = """
            SELECT count() FROM replay_executions
            WHERE project_id IN (
                SELECT project_id FROM traces WHERE trace_id = %(trace_id)s
            )
            AND triggered_at >= toStartOfMonth(now64(6))
        """
        count_result = client.query(count_query, {"trace_id": request.trace_id})
        monthly_count = count_result.result_rows[0][0] if count_result.result_rows else 0
        if monthly_count >= 100:
            raise HTTPException(
                status_code=429,
                detail="Monthly replay limit reached (100/month for Lunch Money tier). Upgrade to Pro for unlimited replays."
            )

    # Extract project_id from trace (query to get it)
    client = get_clickhouse_client()
    project_query = """
        SELECT project_id
        FROM traces
        WHERE trace_id = %(trace_id)s
    """
    project_result = client.query(project_query, {"trace_id": request.trace_id})

    if not project_result.result_rows:
        raise HTTPException(status_code=404, detail=f"Trace {request.trace_id} not found")

    project_id = project_result.result_rows[0][0]

    # Enforce team-scoped project access (member+ required to trigger replays)
    await check_project_access_or_403(str(user.get("user_id", "")), project_id, "member")

    # Convert parameters to dict
    parameters = {
        "model": request.parameters.model,
        "temperature": request.parameters.temperature,
        "system_prompt": request.parameters.system_prompt,
        "max_tokens": request.parameters.max_tokens,
        "stream": request.parameters.stream,
    }

    # Schedule background execution
    background_tasks.add_task(
        execute_replay_in_background,
        execution_id,
        request.trace_id,
        project_id,
        parameters,
    )

    return ReplayExecutionResponse(
        execution_id=execution_id,
        status="running",
        started_at=started_at.isoformat(),
    )


@router.get("/executions/{execution_id}", response_model=ReplayExecutionStatus)
async def get_replay_execution_status(execution_id: str, user: dict = Depends(require_tier("lunch-money"))) -> ReplayExecutionStatus:
    """
    Get the status of a replay execution.

    Returns the current status, parameters, and results (if completed).
    """
    client = get_clickhouse_client()

    # Validate execution_id format
    execution_id = InputValidator.validate_uuid(execution_id, "execution_id")

    query = """
        SELECT execution_id, trace_id, triggered_at, completed_at, status,
               parameters, result, error, project_id
        FROM replay_executions
        WHERE execution_id = %(execution_id)s
    """

    result = client.query(query, {"execution_id": execution_id})

    if not result.result_rows:
        raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")

    row = result.result_rows[0]

    # Enforce team-scoped project access
    exec_project_id = row[8]
    if exec_project_id:
        await check_project_access_or_403(str(user.get("user_id", "")), exec_project_id)

    # Parse JSON fields with safe parsing
    parameters = safe_json_parse(row[5], default={}, field_name="execution.parameters")
    result_data = safe_json_parse(row[6], default=None, field_name="execution.result")

    return ReplayExecutionStatus(
        execution_id=row[0],
        trace_id=row[1],
        started_at=row[2].isoformat() if isinstance(row[2], datetime) else str(row[2]),
        completed_at=row[3].isoformat() if row[3] and isinstance(row[3], datetime) else None,
        status=row[4],
        parameters=parameters,
        result=result_data,
        error=row[7] if row[7] else None,
    )


@router.get("/executions/{execution_id}/comparison", response_model=ReplayComparisonResponse)
async def get_replay_comparison(execution_id: str, user: dict = Depends(require_tier("lunch-money"))) -> ReplayComparisonResponse:
    """
    Get comparison results between original and replayed execution.

    Returns detailed differences, semantic similarity scores, and summary.
    """
    client = get_clickhouse_client()

    # Validate execution_id format
    execution_id = InputValidator.validate_uuid(execution_id, "execution_id")

    query = """
        SELECT comparison, status, project_id
        FROM replay_executions
        WHERE execution_id = %(execution_id)s
    """

    result = client.query(query, {"execution_id": execution_id})

    if not result.result_rows:
        raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")

    row = result.result_rows[0]

    # Enforce team-scoped project access
    exec_project_id = row[2]
    if exec_project_id:
        await check_project_access_or_403(str(user.get("user_id", "")), exec_project_id)

    # Check if completed
    if row[1] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Execution {execution_id} is {row[1]}, comparison not available yet"
        )

    # Parse comparison JSON with safe parsing
    if not row[0]:
        raise HTTPException(
            status_code=500,
            detail="Comparison data not found for completed execution"
        )

    comparison_data = safe_json_parse(row[0], default=None, field_name="execution.comparison")
    if not comparison_data:
        raise HTTPException(
            status_code=500,
            detail="Failed to parse comparison data"
        )

    # Convert differences to SpanDifference objects
    differences = [
        SpanDifference(
            span_name=d["span_name"],
            field=d["field"],
            original_value=d["original_value"],
            modified_value=d["modified_value"],
            semantic_similarity=d.get("semantic_similarity"),
        )
        for d in comparison_data.get("differences", [])
    ]

    return ReplayComparisonResponse(
        execution_id=execution_id,
        original=comparison_data["original"],
        modified=comparison_data["modified"],
        differences=differences,
        summary=comparison_data["summary"],
        semantic_similarity_available=comparison_data["semantic_similarity_available"],
    )


@router.get("/history", response_model=ReplayHistoryResponse)
async def list_replay_history(
    project_id: str = Query(..., description="Project ID to filter executions"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    user: dict = Depends(require_tier("lunch-money")),
) -> ReplayHistoryResponse:
    """
    List replay execution history.

    Returns past replay executions with their status and results.
    """
    client = get_clickhouse_client()

    # Validate inputs to prevent SQL injection
    project_id = InputValidator.validate_project_id(project_id)
    page, page_size = InputValidator.validate_pagination(page, page_size)

    # Enforce team-scoped project access
    await check_project_access_or_403(str(user.get("user_id", "")), project_id)

    # Count total
    count_query = """
        SELECT count() as total
        FROM replay_executions
        WHERE project_id = %(project_id)s
    """
    count_result = client.query(count_query, {"project_id": project_id})
    total = count_result.result_rows[0][0] if count_result.result_rows else 0

    # Query executions with parameterized pagination
    offset = (page - 1) * page_size

    query = """
        SELECT execution_id, trace_id, triggered_at, completed_at, status,
               parameters, result, comparison
        FROM replay_executions
        WHERE project_id = %(project_id)s
        ORDER BY triggered_at DESC
        LIMIT %(page_size)s OFFSET %(offset)s
    """

    result = client.query(
        query,
        {
            "project_id": project_id,
            "page_size": page_size,
            "offset": offset,
        }
    )

    executions = []
    for row in result.result_rows:
        parameters = safe_json_parse(row[5], default={}, field_name="execution.parameters")
        result_data = safe_json_parse(row[6], default=None, field_name="execution.result")
        comparison_data = safe_json_parse(row[7], default=None, field_name="execution.comparison")

        # Calculate duration and cost delta
        duration_ms = None
        cost_delta = None

        if result_data and comparison_data:
            duration_ms = result_data.get("total_duration_ms")
            original_cost = comparison_data["original"].get("total_cost_usd", 0)
            modified_cost = comparison_data["modified"].get("total_cost_usd", 0)
            cost_delta = modified_cost - original_cost

        executions.append(
            ReplayHistoryItem(
                execution_id=row[0],
                trace_id=row[1],
                triggered_at=row[2].isoformat() if isinstance(row[2], datetime) else str(row[2]),
                completed_at=row[3].isoformat() if row[3] and isinstance(row[3], datetime) else None,
                status=row[4],
                parameters=parameters,
                duration_ms=duration_ms,
                cost_delta=cost_delta,
            )
        )

    return ReplayHistoryResponse(
        executions=executions,
        total=total,
        page=page,
        page_size=page_size,
    )


# ============================================================================
# Batch Replay Endpoints
# ============================================================================


@router.post("/batch", response_model=BatchReplayResponse)
async def create_batch_replay(
    request: BatchReplayRequest,
    background_tasks: BackgroundTasks,
    project_id: str = Query(..., description="Project ID"),
    user: dict = Depends(require_tier("pro")),
) -> BatchReplayResponse:
    """
    Trigger a batch replay for multiple traces. Pro tier only.

    Accepts up to 50 trace IDs and shared replay parameters.
    Executes replays concurrently in the background.
    """
    # Validate project_id
    project_id = InputValidator.validate_project_id(project_id)

    # Enforce team-scoped project access (member+ required to trigger batch replays)
    await check_project_access_or_403(str(user.get("user_id", "")), project_id, "member")

    # Validate each trace_id
    for tid in request.trace_ids:
        InputValidator.validate_uuid(tid, "trace_id")

    # Deduplicate trace_ids
    unique_trace_ids = list(dict.fromkeys(request.trace_ids))

    # Verify all traces exist
    client = get_clickhouse_client()
    placeholders = ", ".join(f"%(t{i})s" for i in range(len(unique_trace_ids)))
    params: dict[str, Any] = {f"t{i}": tid for i, tid in enumerate(unique_trace_ids)}
    count_result = client.query(
        f"SELECT count() FROM traces WHERE trace_id IN ({placeholders})",
        parameters=params,
    )
    found_count = count_result.result_rows[0][0] if count_result.result_rows else 0
    if found_count != len(unique_trace_ids):
        raise HTTPException(
            status_code=404,
            detail=f"Some traces not found. Expected {len(unique_trace_ids)}, found {found_count}.",
        )

    batch_id = str(uuid.uuid4())
    created_at = datetime.utcnow()

    parameters = {
        "model": request.parameters.model,
        "temperature": request.parameters.temperature,
        "system_prompt": request.parameters.system_prompt,
        "max_tokens": request.parameters.max_tokens,
        "stream": request.parameters.stream,
    }

    # Insert batch job record
    client.command(
        """
        INSERT INTO batch_replay_jobs (
            batch_id, project_id, user_id, status, trace_ids, parameters,
            total_traces, created_at
        ) VALUES (
            %(batch_id)s, %(project_id)s, %(user_id)s, 'pending',
            %(trace_ids)s, %(parameters)s, %(total_traces)s, now64(6)
        )
        """,
        {
            "batch_id": batch_id,
            "project_id": project_id,
            "user_id": user.get("user_id", ""),
            "trace_ids": unique_trace_ids,
            "parameters": json.dumps(parameters),
            "total_traces": len(unique_trace_ids),
        },
    )

    # Schedule background execution
    background_tasks.add_task(
        execute_batch_replay_in_background,
        batch_id,
        unique_trace_ids,
        project_id,
        parameters,
    )

    return BatchReplayResponse(
        batch_id=batch_id,
        status="pending",
        total_traces=len(unique_trace_ids),
        created_at=created_at.isoformat(),
    )


@router.get("/batch/{batch_id}", response_model=BatchStatusResponse)
async def get_batch_status(
    batch_id: str,
    include_executions: bool = Query(False, description="Include individual execution details"),
    user: dict = Depends(require_tier("pro")),
) -> BatchStatusResponse:
    """
    Get the status of a batch replay job.

    Optionally includes individual execution details.
    """
    batch_id = InputValidator.validate_uuid(batch_id, "batch_id")
    client = get_clickhouse_client()

    query = """
        SELECT batch_id, project_id, status, trace_ids, parameters,
               total_traces, completed_traces, failed_traces,
               created_at, started_at, completed_at, error, summary
        FROM batch_replay_jobs
        WHERE batch_id = %(batch_id)s
    """
    result = client.query(query, {"batch_id": batch_id})

    if not result.result_rows:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

    row = result.result_rows[0]

    # Enforce team-scoped project access (project_id is row[1])
    batch_project_id = row[1]
    if batch_project_id:
        await check_project_access_or_403(str(user.get("user_id", "")), batch_project_id)

    executions = None
    if include_executions:
        exec_query = """
            SELECT execution_id, trace_id, triggered_at, completed_at, status,
                   parameters, result, comparison
            FROM replay_executions
            WHERE batch_id = %(batch_id)s
            ORDER BY triggered_at ASC
        """
        exec_result = client.query(exec_query, {"batch_id": batch_id})
        executions = []
        for erow in exec_result.result_rows:
            exec_params = safe_json_parse(erow[5], default={}, field_name="exec.parameters")
            result_data = safe_json_parse(erow[6], default=None, field_name="exec.result")
            comparison_data = safe_json_parse(erow[7], default=None, field_name="exec.comparison")
            duration_ms = None
            cost_delta = None
            if result_data and comparison_data:
                duration_ms = result_data.get("total_duration_ms")
                orig_cost = comparison_data.get("original", {}).get("total_cost_usd", 0)
                mod_cost = comparison_data.get("modified", {}).get("total_cost_usd", 0)
                cost_delta = mod_cost - orig_cost
            executions.append(ReplayHistoryItem(
                execution_id=erow[0],
                trace_id=erow[1],
                triggered_at=erow[2].isoformat() if isinstance(erow[2], datetime) else str(erow[2]),
                completed_at=erow[3].isoformat() if erow[3] and isinstance(erow[3], datetime) else None,
                status=erow[4],
                parameters=exec_params,
                duration_ms=duration_ms,
                cost_delta=cost_delta,
            ))

    return BatchStatusResponse(
        batch_id=row[0],
        project_id=row[1],
        status=row[2],
        trace_ids=row[3],
        parameters=safe_json_parse(row[4], default={}, field_name="batch.parameters"),
        total_traces=row[5],
        completed_traces=row[6],
        failed_traces=row[7],
        created_at=row[8].isoformat() if isinstance(row[8], datetime) else str(row[8]),
        started_at=row[9].isoformat() if row[9] and isinstance(row[9], datetime) else None,
        completed_at=row[10].isoformat() if row[10] and isinstance(row[10], datetime) else None,
        error=row[11] if row[11] else None,
        summary=safe_json_parse(row[12], default=None, field_name="batch.summary"),
        executions=executions,
    )


@router.get("/batch", response_model=BatchListResponse)
async def list_batch_jobs(
    project_id: str = Query(..., description="Project ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    user: dict = Depends(require_tier("pro")),
) -> BatchListResponse:
    """
    List batch replay jobs for a project.
    """
    project_id = InputValidator.validate_project_id(project_id)
    page, page_size = InputValidator.validate_pagination(page, page_size)

    # Enforce team-scoped project access
    await check_project_access_or_403(str(user.get("user_id", "")), project_id)

    client = get_clickhouse_client()

    count_query = """
        SELECT count() FROM batch_replay_jobs
        WHERE project_id = %(project_id)s
    """
    count_result = client.query(count_query, {"project_id": project_id})
    total = count_result.result_rows[0][0] if count_result.result_rows else 0

    offset = (page - 1) * page_size
    query = """
        SELECT batch_id, status, total_traces, completed_traces, failed_traces,
               created_at, completed_at, parameters
        FROM batch_replay_jobs
        WHERE project_id = %(project_id)s
        ORDER BY created_at DESC
        LIMIT %(page_size)s OFFSET %(offset)s
    """
    result = client.query(query, {
        "project_id": project_id,
        "page_size": page_size,
        "offset": offset,
    })

    batches = []
    for row in result.result_rows:
        batches.append(BatchListItem(
            batch_id=row[0],
            status=row[1],
            total_traces=row[2],
            completed_traces=row[3],
            failed_traces=row[4],
            created_at=row[5].isoformat() if isinstance(row[5], datetime) else str(row[5]),
            completed_at=row[6].isoformat() if row[6] and isinstance(row[6], datetime) else None,
            parameters=safe_json_parse(row[7], default={}, field_name="batch.parameters"),
        ))

    return BatchListResponse(
        batches=batches,
        total=total,
        page=page,
        page_size=page_size,
    )
