"""Langfuse API client and trace normalizer.

Fetches traces and observations from a Langfuse instance via its
REST API (Basic Auth) and normalizes them into Prela's ClickHouse
format (trace_rows, span_rows) with source='langfuse'.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Langfuse observation type → Prela span_type
_OBSERVATION_TYPE_MAP: dict[str, str | None] = {
    "GENERATION": "llm",
    "SPAN": "custom",
    "CHAIN": "custom",
    "AGENT": "custom",
    "RETRIEVER": "retrieval",
    "EMBEDDING": "embedding",
    "TOOL": "tool",
    "EVALUATOR": "custom",
    "GUARDRAIL": "custom",
    "EVENT": None,  # Skip — events are metadata, not spans
}

# Langfuse level → Prela status
_LEVEL_STATUS_MAP: dict[str, str] = {
    "DEBUG": "success",
    "DEFAULT": "success",
    "WARNING": "success",
    "ERROR": "error",
}


async def test_langfuse_connection(
    host: str,
    public_key: str,
    secret_key: str,
) -> bool:
    """Test that Langfuse credentials are valid.

    Hits GET /api/public/traces?limit=1 to validate authentication.

    Args:
        host: Langfuse instance URL (e.g. https://cloud.langfuse.com).
        public_key: Langfuse public key.
        secret_key: Langfuse secret key.

    Returns:
        True if connection is valid.

    Raises:
        httpx.HTTPStatusError: On auth failure or server error.
        httpx.ConnectError: On network failure.
    """
    url = f"{host.rstrip('/')}/api/public/traces"
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            url,
            params={"limit": 1},
            auth=(public_key, secret_key),
        )
        resp.raise_for_status()
        return True


async def fetch_langfuse_traces(
    host: str,
    public_key: str,
    secret_key: str,
    from_timestamp: str | None = None,
    limit_per_page: int = 100,
    max_pages: int = 50,
) -> list[dict[str, Any]]:
    """Fetch traces from Langfuse, paginated.

    Args:
        host: Langfuse instance URL.
        public_key: Langfuse public key.
        secret_key: Langfuse secret key.
        from_timestamp: ISO8601 timestamp — only fetch traces after this time.
        limit_per_page: Traces per page (max 100 for Langfuse).
        max_pages: Safety cap on pagination to avoid runaway fetches.

    Returns:
        List of raw Langfuse trace dicts.
    """
    url = f"{host.rstrip('/')}/api/public/traces"
    all_traces: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for page in range(1, max_pages + 1):
            params: dict[str, Any] = {"page": page, "limit": limit_per_page}
            if from_timestamp:
                params["fromTimestamp"] = from_timestamp

            resp = await client.get(
                url,
                params=params,
                auth=(public_key, secret_key),
            )
            resp.raise_for_status()
            body = resp.json()

            traces = body.get("data", [])
            if not traces:
                break

            all_traces.extend(traces)

            # If we got fewer than limit, we've reached the end
            if len(traces) < limit_per_page:
                break

            # Rate-limit politeness delay
            await asyncio.sleep(0.1)

    logger.info(f"Fetched {len(all_traces)} traces from Langfuse ({host})")
    return all_traces


async def fetch_langfuse_observations(
    host: str,
    public_key: str,
    secret_key: str,
    trace_id: str,
    limit_per_page: int = 100,
    max_pages: int = 20,
) -> list[dict[str, Any]]:
    """Fetch all observations for a single Langfuse trace.

    Args:
        host: Langfuse instance URL.
        public_key: Langfuse public key.
        secret_key: Langfuse secret key.
        trace_id: Langfuse trace ID.
        limit_per_page: Observations per page.
        max_pages: Safety cap.

    Returns:
        List of raw Langfuse observation dicts.
    """
    url = f"{host.rstrip('/')}/api/public/observations"
    all_observations: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for page in range(1, max_pages + 1):
            params: dict[str, Any] = {
                "traceId": trace_id,
                "page": page,
                "limit": limit_per_page,
            }

            resp = await client.get(
                url,
                params=params,
                auth=(public_key, secret_key),
            )
            resp.raise_for_status()
            body = resp.json()

            observations = body.get("data", [])
            if not observations:
                break

            all_observations.extend(observations)

            if len(observations) < limit_per_page:
                break

            await asyncio.sleep(0.1)

    return all_observations


def normalize_langfuse_traces(
    langfuse_traces: list[dict[str, Any]],
    observations_by_trace: dict[str, list[dict[str, Any]]],
    project_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert Langfuse traces + observations to Prela ClickHouse format.

    Args:
        langfuse_traces: Raw Langfuse trace dicts.
        observations_by_trace: Map of trace_id → list of observation dicts.
        project_id: Prela project ID for the imported data.

    Returns:
        Tuple of (trace_rows, span_rows) ready for ClickHouse insert.
    """
    all_trace_rows: list[dict[str, Any]] = []
    all_span_rows: list[dict[str, Any]] = []

    for lf_trace in langfuse_traces:
        trace_id = lf_trace.get("id", "")
        if not trace_id:
            continue

        observations = observations_by_trace.get(trace_id, [])
        trace_row, span_rows = _normalize_single_trace(
            lf_trace, observations, project_id
        )
        all_trace_rows.append(trace_row)
        all_span_rows.extend(span_rows)

    return all_trace_rows, all_span_rows


def _normalize_single_trace(
    lf_trace: dict[str, Any],
    observations: list[dict[str, Any]],
    project_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Normalize a single Langfuse trace and its observations.

    Args:
        lf_trace: Raw Langfuse trace dict.
        observations: Raw Langfuse observation dicts for this trace.
        project_id: Prela project ID.

    Returns:
        Tuple of (trace_row_dict, list_of_span_row_dicts).
    """
    trace_id = lf_trace["id"]

    # Build span rows from observations (skip EVENTs)
    span_rows: list[dict[str, Any]] = []
    for obs in observations:
        obs_type = obs.get("type", "SPAN")
        prela_type = _OBSERVATION_TYPE_MAP.get(obs_type)
        if prela_type is None:
            # EVENT type — skip
            continue

        span_row = _normalize_observation(obs, trace_id, project_id, prela_type)
        span_rows.append(span_row)

    # Build trace row
    started_at = _parse_iso_timestamp(lf_trace.get("timestamp"))

    # Compute timing from observations if available
    if span_rows:
        earliest = min(s["started_at"] for s in span_rows)
        latest = max(s["ended_at"] for s in span_rows if s["ended_at"])
        if not started_at:
            started_at = earliest
        completed_at = latest or started_at
    else:
        if not started_at:
            started_at = datetime.now(tz=timezone.utc)
        completed_at = started_at

    duration_ms = (completed_at - started_at).total_seconds() * 1000

    # Determine status
    has_error = any(s["status"] == "error" for s in span_rows)
    status = "error" if has_error else "success"

    # Find root span (no parent)
    root_span_id = ""
    for s in span_rows:
        if not s["parent_span_id"]:
            root_span_id = s["span_id"]
            break
    if not root_span_id and span_rows:
        root_span_id = span_rows[0]["span_id"]

    # Build trace attributes from Langfuse metadata
    trace_attrs: dict[str, Any] = {}
    if lf_trace.get("name"):
        trace_attrs["langfuse.trace.name"] = lf_trace["name"]
    if lf_trace.get("userId"):
        trace_attrs["langfuse.user_id"] = lf_trace["userId"]
    if lf_trace.get("sessionId"):
        trace_attrs["langfuse.session_id"] = lf_trace["sessionId"]
    if lf_trace.get("tags"):
        trace_attrs["langfuse.tags"] = lf_trace["tags"]
    if lf_trace.get("release"):
        trace_attrs["langfuse.release"] = lf_trace["release"]
    if lf_trace.get("version"):
        trace_attrs["langfuse.version"] = lf_trace["version"]
    if lf_trace.get("metadata"):
        trace_attrs["langfuse.metadata"] = lf_trace["metadata"]

    service_name = lf_trace.get("name", "langfuse-import")

    trace_row = {
        "trace_id": trace_id,
        "project_id": project_id,
        "service_name": service_name,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_ms": duration_ms,
        "status": status,
        "root_span_id": root_span_id,
        "span_count": len(span_rows),
        "attributes": json.dumps(trace_attrs),
        "source": "langfuse",
    }

    return trace_row, span_rows


def _normalize_observation(
    obs: dict[str, Any],
    trace_id: str,
    project_id: str,
    span_type: str,
) -> dict[str, Any]:
    """Convert a single Langfuse observation to a Prela span row.

    Args:
        obs: Raw Langfuse observation dict.
        trace_id: Parent trace ID.
        project_id: Prela project ID.
        span_type: Already-mapped Prela span type.

    Returns:
        Dict ready for ClickHouse spans table insert.
    """
    started_at = _parse_iso_timestamp(obs.get("startTime")) or datetime.now(tz=timezone.utc)
    ended_at = _parse_iso_timestamp(obs.get("endTime")) or started_at
    duration_ms = (ended_at - started_at).total_seconds() * 1000

    # Map Langfuse level to Prela status
    level = obs.get("level", "DEFAULT")
    status = _LEVEL_STATUS_MAP.get(level, "success")

    # Build attributes
    attributes: dict[str, Any] = {}

    if obs.get("name"):
        attributes["name"] = obs["name"]

    # LLM-specific attributes from GENERATION observations
    if obs.get("model"):
        attributes["gen_ai.request.model"] = obs["model"]
    if obs.get("modelParameters"):
        attributes["gen_ai.request.parameters"] = obs["modelParameters"]

    # Token usage
    usage = obs.get("usage") or {}
    if usage:
        if usage.get("promptTokens") is not None:
            attributes["gen_ai.usage.prompt_tokens"] = usage["promptTokens"]
        if usage.get("completionTokens") is not None:
            attributes["gen_ai.usage.completion_tokens"] = usage["completionTokens"]
        if usage.get("totalTokens") is not None:
            attributes["gen_ai.usage.total_tokens"] = usage["totalTokens"]

    # Input/output
    if obs.get("input") is not None:
        attributes["input"] = obs["input"]
    if obs.get("output") is not None:
        attributes["output"] = obs["output"]

    # Langfuse metadata
    if obs.get("metadata"):
        attributes["langfuse.metadata"] = obs["metadata"]
    if obs.get("statusMessage"):
        attributes["langfuse.status_message"] = obs["statusMessage"]
    if obs.get("version"):
        attributes["langfuse.version"] = obs["version"]

    # Parent observation → parent span
    parent_span_id = obs.get("parentObservationId", "") or ""

    return {
        "span_id": obs.get("id", str(uuid.uuid4())),
        "trace_id": trace_id,
        "project_id": project_id,
        "parent_span_id": parent_span_id,
        "name": obs.get("name", ""),
        "span_type": span_type,
        "service_name": "langfuse-import",
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_ms": duration_ms,
        "status": status,
        "attributes": json.dumps(attributes, default=str),
        "events": json.dumps([]),
        "replay_snapshot": json.dumps({}),
        "source": "langfuse",
    }


def _parse_iso_timestamp(value: Any) -> datetime | None:
    """Parse an ISO8601 timestamp string to a UTC datetime.

    Handles Langfuse's various timestamp formats:
    - "2026-02-13T10:30:00.000Z"
    - "2026-02-13T10:30:00+00:00"
    - "2026-02-13T10:30:00"

    Args:
        value: ISO8601 timestamp string, or None.

    Returns:
        UTC datetime, or None if parsing fails.
    """
    if not value or not isinstance(value, str):
        return None

    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        logger.warning(f"Failed to parse timestamp: {value}")
        return None
