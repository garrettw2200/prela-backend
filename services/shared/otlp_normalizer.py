"""OTLP JSON to Prela internal format normalizer.

Converts OpenTelemetry Protocol (OTLP) JSON payloads into the dict format
expected by Prela's ClickHouse insert paths. This is the reverse of what
sdk/prela/exporters/otlp.py._span_to_otlp() performs.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# OTLP status code → Prela status string
_STATUS_CODE_MAP = {
    0: "pending",   # UNSET
    1: "success",   # OK
    2: "error",     # ERROR
}


def normalize_otlp_traces(
    otlp_payload: dict[str, Any],
    project_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse an OTLP JSON payload and return Prela trace and span rows.

    Args:
        otlp_payload: The full OTLP JSON body (containing resourceSpans).
        project_id: The authenticated user's project/user ID.

    Returns:
        A tuple of (trace_rows, span_rows) ready for ClickHouse insert.
    """
    all_span_rows: list[dict[str, Any]] = []

    for resource_span in otlp_payload.get("resourceSpans", []):
        # Extract service name from resource attributes
        resource = resource_span.get("resource", {})
        resource_attrs = _parse_otlp_attributes(resource.get("attributes", []))
        service_name = resource_attrs.get("service.name", "unknown")

        for scope_span in resource_span.get("scopeSpans", []):
            for otlp_span in scope_span.get("spans", []):
                span_row = _normalize_span(otlp_span, project_id, service_name)
                all_span_rows.append(span_row)

    # Synthesize trace rows by grouping spans by trace_id
    trace_rows = _synthesize_traces(all_span_rows, project_id)

    return trace_rows, all_span_rows


def _normalize_span(
    otlp_span: dict[str, Any],
    project_id: str,
    service_name: str,
) -> dict[str, Any]:
    """Convert a single OTLP span dict to Prela ClickHouse span dict.

    Args:
        otlp_span: Single span from OTLP scopeSpans.spans.
        project_id: Project/user ID.
        service_name: Extracted from resource attributes.

    Returns:
        Dict ready for ClickHouse spans table insert.
    """
    # Parse timestamps (nanosecond strings → datetime)
    started_at = _nanos_to_datetime(otlp_span.get("startTimeUnixNano", "0"))
    ended_at = _nanos_to_datetime(otlp_span.get("endTimeUnixNano", "0"))
    duration_ms = (ended_at - started_at).total_seconds() * 1000

    # Parse attributes
    attributes = _parse_otlp_attributes(otlp_span.get("attributes", []))

    # Infer span type from kind + attributes
    kind = otlp_span.get("kind", 0)
    span_type = _infer_span_type(kind, attributes)

    # Parse status
    status_obj = otlp_span.get("status", {})
    status_code = status_obj.get("code", 0)
    status = _STATUS_CODE_MAP.get(status_code, "pending")

    # Parse events
    events = _parse_otlp_events(otlp_span.get("events", []))

    return {
        "span_id": otlp_span.get("spanId", ""),
        "trace_id": otlp_span.get("traceId", ""),
        "project_id": project_id,
        "parent_span_id": otlp_span.get("parentSpanId", ""),
        "name": otlp_span.get("name", ""),
        "span_type": span_type,
        "service_name": service_name,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_ms": duration_ms,
        "status": status,
        "attributes": json.dumps(attributes),
        "events": json.dumps(events),
        "replay_snapshot": json.dumps({}),
        "source": "otlp",
    }


def _synthesize_traces(
    span_rows: list[dict[str, Any]],
    project_id: str,
) -> list[dict[str, Any]]:
    """Synthesize trace rows by grouping spans by trace_id.

    OTLP has no explicit trace object — traces are implicit from spans
    sharing a traceId. We aggregate span data to build trace-level rows.

    Args:
        span_rows: All normalized span rows.
        project_id: Project/user ID.

    Returns:
        List of trace row dicts for ClickHouse traces table.
    """
    traces: dict[str, list[dict[str, Any]]] = {}
    for span in span_rows:
        trace_id = span["trace_id"]
        if trace_id not in traces:
            traces[trace_id] = []
        traces[trace_id].append(span)

    trace_rows = []
    for trace_id, spans in traces.items():
        # Find timing bounds
        started_at = min(s["started_at"] for s in spans)
        completed_at = max(s["ended_at"] for s in spans)
        duration_ms = (completed_at - started_at).total_seconds() * 1000

        # Status: error if any span errored
        has_error = any(s["status"] == "error" for s in spans)
        status = "error" if has_error else "success"

        # Root span: the one without a parent
        root_span_id = ""
        for s in spans:
            if not s["parent_span_id"]:
                root_span_id = s["span_id"]
                break
        # Fallback: use first span
        if not root_span_id and spans:
            root_span_id = spans[0]["span_id"]

        # Service name from root span or first span
        service_name = "unknown"
        for s in spans:
            if s["span_id"] == root_span_id:
                service_name = s["service_name"]
                break
        if service_name == "unknown" and spans:
            service_name = spans[0]["service_name"]

        trace_rows.append({
            "trace_id": trace_id,
            "project_id": project_id,
            "service_name": service_name,
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_ms": duration_ms,
            "status": status,
            "root_span_id": root_span_id,
            "span_count": len(spans),
            "attributes": json.dumps({}),
            "source": "otlp",
        })

    return trace_rows


def _infer_span_type(kind: int, attributes: dict[str, Any]) -> str:
    """Infer Prela span_type from OTLP span kind and attributes.

    Uses attribute-based heuristics first (more reliable), then falls
    back to OTLP span kind.

    Args:
        kind: OTLP span kind integer (0-5).
        attributes: Already-parsed flat attributes dict.

    Returns:
        Prela span_type string.
    """
    attr_keys = set(attributes.keys())

    # Check for LLM-related attributes
    if any(k.startswith("gen_ai.") or k.startswith("llm.") for k in attr_keys):
        return "llm"

    # Check for retrieval/database attributes
    if any(k.startswith("db.") or k.startswith("retrieval.") for k in attr_keys):
        return "retrieval"

    # Check for embedding attributes
    if any(k.startswith("embedding.") for k in attr_keys):
        return "embedding"

    # Check for tool attributes
    if any(k.startswith("tool.") for k in attr_keys):
        return "tool"

    # Fallback based on OTLP span kind
    # 0=UNSPECIFIED, 1=INTERNAL, 2=SERVER, 3=CLIENT, 4=PRODUCER, 5=CONSUMER
    kind_map = {
        2: "tool",    # SERVER — handling incoming requests
    }
    return kind_map.get(kind, "custom")


def _parse_otlp_attributes(
    otlp_attrs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert OTLP attribute list to flat Python dict.

    OTLP attributes format:
    [{"key": "foo", "value": {"stringValue": "bar"}}, ...]

    Args:
        otlp_attrs: List of OTLP attribute dicts.

    Returns:
        Flat Python dict of key-value pairs.
    """
    result: dict[str, Any] = {}

    for attr in otlp_attrs:
        key = attr.get("key", "")
        if not key:
            continue

        value_obj = attr.get("value", {})
        value = _extract_otlp_value(value_obj)
        if value is not None:
            result[key] = value

    return result


def _extract_otlp_value(value_obj: dict[str, Any]) -> Any:
    """Extract a typed value from an OTLP value object.

    Args:
        value_obj: OTLP value dict with a type key (stringValue, intValue, etc.)

    Returns:
        Python value, or None if unrecognized.
    """
    if "stringValue" in value_obj:
        return value_obj["stringValue"]
    elif "intValue" in value_obj:
        return int(value_obj["intValue"])
    elif "doubleValue" in value_obj:
        return float(value_obj["doubleValue"])
    elif "boolValue" in value_obj:
        return value_obj["boolValue"]
    elif "arrayValue" in value_obj:
        values = value_obj["arrayValue"].get("values", [])
        return [_extract_otlp_value(v) for v in values]
    elif "bytesValue" in value_obj:
        return value_obj["bytesValue"]
    return None


def _parse_otlp_events(
    otlp_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert OTLP events to Prela event format.

    OTLP event: {"timeUnixNano": "...", "name": "...", "attributes": [...]}
    Prela event: {"timestamp": "ISO8601", "name": "...", "attributes": {...}}

    Args:
        otlp_events: List of OTLP event dicts.

    Returns:
        List of Prela event dicts.
    """
    events = []
    for otlp_event in otlp_events:
        timestamp = _nanos_to_datetime(otlp_event.get("timeUnixNano", "0"))
        attributes = _parse_otlp_attributes(otlp_event.get("attributes", []))

        events.append({
            "timestamp": timestamp.isoformat(),
            "name": otlp_event.get("name", ""),
            "attributes": attributes,
        })

    return events


def _nanos_to_datetime(nanos_str: str) -> datetime:
    """Convert OTLP nanosecond timestamp string to datetime.

    Args:
        nanos_str: Nanosecond timestamp as string (e.g. "1706000000000000000").

    Returns:
        UTC datetime (microsecond precision — last 3 digits of nanos are lost,
        which matches ClickHouse DateTime64(6)).
    """
    try:
        nanos = int(nanos_str)
        seconds = nanos / 1_000_000_000
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (ValueError, OSError):
        return datetime.now(tz=timezone.utc)
