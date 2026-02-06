"""Trace query endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from shared import get_clickhouse_client, query_spans, query_traces

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/traces")
async def list_traces(
    project_id: str = Query(..., description="Project ID"),
    service_name: str | None = Query(None, description="Filter by service name"),
    start_time: str | None = Query(None, description="Start time (ISO format)"),
    end_time: str | None = Query(None, description="End time (ISO format)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of traces"),
) -> dict[str, Any]:
    """List traces with optional filters.

    Args:
        project_id: Project ID to filter traces.
        service_name: Filter by service name.
        start_time: Filter by start time (ISO format).
        end_time: Filter by end time (ISO format).
        limit: Maximum number of traces to return.

    Returns:
        Dictionary with traces list and metadata.
    """
    try:
        client = get_clickhouse_client()
        traces = await query_traces(
            client, service_name=service_name, start_time=start_time, end_time=end_time, limit=limit
        )
        return {"traces": traces, "count": len(traces), "limit": limit}
    except Exception as e:
        logger.error(f"Failed to query traces: {e}")
        raise HTTPException(status_code=500, detail="Failed to query traces")


@router.get("/traces/{trace_id}")
async def get_trace(
    trace_id: str,
) -> dict[str, Any]:
    """Get a single trace with all spans.

    Args:
        trace_id: Trace ID to retrieve.

    Returns:
        Dictionary with trace and spans data.
    """
    try:
        client = get_clickhouse_client()

        # Get trace metadata
        trace_query = "SELECT * FROM traces WHERE trace_id = %(trace_id)s LIMIT 1"
        trace_result = client.query(trace_query, parameters={"trace_id": trace_id})

        if not trace_result.result_rows:
            raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

        trace = trace_result.result_rows[0]

        # Get all spans for this trace
        spans = await query_spans(client, trace_id)

        return {
            "trace": trace,
            "spans": spans,
            "span_count": len(spans),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trace {trace_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve trace")


@router.get("/traces/{trace_id}/spans")
async def list_spans(
    trace_id: str,
) -> dict[str, Any]:
    """List all spans for a trace.

    Args:
        trace_id: Trace ID to query.

    Returns:
        Dictionary with spans list.
    """
    try:
        client = get_clickhouse_client()

        # Check if trace exists
        trace_query = "SELECT project_id FROM traces WHERE trace_id = %(trace_id)s LIMIT 1"
        trace_result = client.query(trace_query, parameters={"trace_id": trace_id})

        if not trace_result.result_rows:
            raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

        spans = await query_spans(client, trace_id)
        return {"spans": spans, "count": len(spans)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to query spans for trace {trace_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to query spans")


@router.get("/search")
async def search_traces(
    project_id: str = Query(..., description="Project ID"),
    query: str = Query(..., description="Search query"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of results"),
) -> dict[str, Any]:
    """Search traces by content.

    Args:
        project_id: Project ID to search within.
        query: Search query string.
        limit: Maximum number of results.

    Returns:
        Dictionary with matching traces.
    """
    try:
        client = get_clickhouse_client()

        # Search in span attributes and trace metadata, filtered by project
        search_query = """
            SELECT DISTINCT trace_id, service_name, started_at, status
            FROM spans
            WHERE project_id = %(project_id)s
              AND (attributes LIKE %(pattern)s OR name LIKE %(pattern)s)
            ORDER BY started_at DESC
            LIMIT %(limit)s
        """

        result = client.query(
            search_query, parameters={
                "project_id": project_id,
                "pattern": f"%{query}%",
                "limit": limit
            }
        )

        return {"results": result.result_rows, "count": len(result.result_rows)}
    except Exception as e:
        logger.error(f"Failed to search traces: {e}")
        raise HTTPException(status_code=500, detail="Failed to search traces")
