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

from ..auth import require_tier
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
) -> None:
    """Execute replay in background and store results in ClickHouse."""
    client = get_clickhouse_client()

    try:
        from prela.replay import ReplayEngine, compare_replays

        # Update status to running
        client.command(
            """
            INSERT INTO replay_executions (
                execution_id, trace_id, project_id, triggered_at, status, parameters
            ) VALUES (
                %(execution_id)s, %(trace_id)s, %(project_id)s, now64(6), 'running', %(parameters)s
            )
            """,
            {
                "execution_id": execution_id,
                "trace_id": trace_id,
                "project_id": project_id,
                "parameters": json.dumps(parameters),
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

    # Query trace
    trace_query = """
        SELECT trace_id, service_name, started_at, duration_ms, status, attributes
        FROM traces
        WHERE trace_id = %(trace_id)s
    """

    trace_result = client.query(trace_query, {"trace_id": trace_id})

    if not trace_result.result_rows:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

    trace_row = trace_result.result_rows[0]

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
               parameters, result, error
        FROM replay_executions
        WHERE execution_id = %(execution_id)s
    """

    result = client.query(query, {"execution_id": execution_id})

    if not result.result_rows:
        raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")

    row = result.result_rows[0]

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
        SELECT comparison, status
        FROM replay_executions
        WHERE execution_id = %(execution_id)s
    """

    result = client.query(query, {"execution_id": execution_id})

    if not result.result_rows:
        raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")

    row = result.result_rows[0]

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
