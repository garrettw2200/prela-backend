"""ClickHouse client utilities."""

import logging
import threading
import time
from typing import Any, Optional

import clickhouse_connect
from clickhouse_connect.driver import Client

from .config import settings

logger = logging.getLogger(__name__)

_client: Optional[Client] = None
_client_lock = threading.Lock()

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 5  # seconds


def get_clickhouse_client() -> Client:
    """Get or create a singleton ClickHouse client.

    Uses a module-level singleton so the expensive init query
    (SELECT system.settings LIMIT 10000) runs only once per process.
    Thread-safe via a lock. Retries with exponential backoff on failure.
    """
    global _client

    # Fast path: client already exists and is usable
    if _client is not None:
        try:
            _client.command("SELECT 1")
            return _client
        except Exception:
            logger.warning("Existing ClickHouse client is stale, reconnecting")
            _client = None

    with _client_lock:
        # Double-check after acquiring lock
        if _client is not None:
            try:
                _client.command("SELECT 1")
                return _client
            except Exception:
                _client = None

        last_error = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                client = clickhouse_connect.get_client(
                    host=settings.clickhouse_host,
                    port=settings.clickhouse_port,
                    username=settings.clickhouse_user,
                    password=settings.clickhouse_password,
                    database=settings.clickhouse_database,
                    secure=settings.clickhouse_secure,
                    connect_timeout=30,
                    send_receive_timeout=300,
                )
                logger.info(
                    f"Connected to ClickHouse: {settings.clickhouse_host}:{settings.clickhouse_port} "
                    f"(attempt {attempt}/{_MAX_RETRIES})"
                )
                _client = client
                return _client
            except Exception as e:
                last_error = e
                wait = _RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    f"ClickHouse connection attempt {attempt}/{_MAX_RETRIES} failed: {e}. "
                    f"Retrying in {wait}s..."
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(wait)

        logger.error(f"Failed to connect to ClickHouse after {_MAX_RETRIES} attempts: {last_error}")
        raise last_error


async def init_clickhouse_schema(client: Client) -> None:
    """Initialize ClickHouse schema with required tables and materialized views.

    Args:
        client: ClickHouse client instance.
    """
    # Projects table
    client.command(
        """
        CREATE TABLE IF NOT EXISTS projects (
            project_id String,
            name String,
            description String,
            webhook_url String,
            created_at DateTime64(6) DEFAULT now64(6),
            updated_at DateTime64(6) DEFAULT now64(6)
        )
        ENGINE = MergeTree()
        ORDER BY (project_id, created_at)
        TTL toDateTime(created_at) + INTERVAL 365 DAY
        """
    )

    # Traces table
    client.command(
        """
        CREATE TABLE IF NOT EXISTS traces (
            trace_id String,
            project_id String,
            service_name String,
            started_at DateTime64(6),
            completed_at DateTime64(6),
            duration_ms Float64,
            status String,
            root_span_id String,
            span_count UInt32,
            attributes String,
            source String DEFAULT 'native',
            created_at DateTime64(6) DEFAULT now64(6)
        )
        ENGINE = MergeTree()
        ORDER BY (project_id, service_name, started_at, trace_id)
        PARTITION BY (project_id, toYYYYMM(started_at))
        TTL toDateTime(started_at) + INTERVAL 90 DAY
        """
    )

    # Spans table
    client.command(
        """
        CREATE TABLE IF NOT EXISTS spans (
            span_id String,
            trace_id String,
            project_id String,
            parent_span_id String,
            name String,
            span_type String,
            service_name String,
            started_at DateTime64(6),
            ended_at DateTime64(6),
            duration_ms Float64,
            status String,
            attributes String,
            events String,
            replay_snapshot String,
            source String DEFAULT 'native',
            created_at DateTime64(6) DEFAULT now64(6)
        )
        ENGINE = MergeTree()
        ORDER BY (project_id, trace_id, started_at, span_id)
        PARTITION BY (project_id, toYYYYMM(started_at))
        TTL toDateTime(started_at) + INTERVAL 90 DAY
        """
    )

    # Span attributes table for fast filtering
    client.command(
        """
        CREATE TABLE IF NOT EXISTS span_attributes (
            span_id String,
            trace_id String,
            project_id String,
            key String,
            value String,
            started_at DateTime64(6)
        )
        ENGINE = MergeTree()
        ORDER BY (project_id, key, value, trace_id, span_id)
        PARTITION BY (project_id, toYYYYMM(started_at))
        TTL toDateTime(started_at) + INTERVAL 90 DAY
        """
    )

    # Replay executions table
    client.command(
        """
        CREATE TABLE IF NOT EXISTS replay_executions (
            execution_id String,
            trace_id String,
            project_id String,
            triggered_at DateTime64(6),
            completed_at Nullable(DateTime64(6)),
            status Enum8('running' = 1, 'completed' = 2, 'failed' = 3),
            parameters String,
            result String,
            comparison String,
            error Nullable(String),
            created_at DateTime64(6) DEFAULT now64(6)
        )
        ENGINE = MergeTree()
        ORDER BY (project_id, triggered_at, execution_id)
        PARTITION BY (project_id, toYYYYMM(triggered_at))
        TTL toDateTime(triggered_at) + INTERVAL 90 DAY
        """
    )

    # n8n workflow metrics materialized view
    client.command(
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS n8n_workflow_metrics
        ENGINE = SummingMergeTree()
        PARTITION BY (project_id, toYYYYMM(date))
        ORDER BY (project_id, workflow_id, date)
        AS SELECT
            project_id,
            JSONExtractString(attributes, 'n8n.workflow.id') AS workflow_id,
            JSONExtractString(attributes, 'n8n.workflow.name') AS workflow_name,
            toDate(started_at) AS date,
            count() AS execution_count,
            countIf(status = 'error') AS error_count,
            countIf(status = 'success') AS success_count,
            avg(duration_ms) AS avg_duration_ms,
            quantile(0.95)(duration_ms) AS p95_duration_ms,
            quantile(0.99)(duration_ms) AS p99_duration_ms
        FROM traces
        WHERE JSONHas(attributes, 'n8n.workflow.id')
        GROUP BY project_id, workflow_id, workflow_name, date
        """
    )

    # n8n AI node metrics materialized view
    client.command(
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS n8n_ai_node_metrics
        ENGINE = SummingMergeTree()
        PARTITION BY (project_id, toYYYYMM(date))
        ORDER BY (project_id, workflow_id, node_name, model, date)
        AS SELECT
            project_id,
            JSONExtractString(attributes, 'n8n.workflow.id') AS workflow_id,
            name AS node_name,
            JSONExtractString(attributes, 'llm.model') AS model,
            JSONExtractString(attributes, 'llm.vendor') AS vendor,
            toDate(started_at) AS date,
            count() AS call_count,
            sum(JSONExtractUInt(attributes, 'llm.prompt_tokens')) AS prompt_tokens,
            sum(JSONExtractUInt(attributes, 'llm.completion_tokens')) AS completion_tokens,
            sum(JSONExtractUInt(attributes, 'llm.total_tokens')) AS total_tokens,
            avg(JSONExtractFloat(attributes, 'llm.latency_ms')) AS avg_latency_ms,
            sum(JSONExtractFloat(attributes, 'llm.cost_usd')) AS total_cost_usd
        FROM spans
        WHERE span_type = 'llm' AND JSONHas(attributes, 'n8n.workflow.id')
        GROUP BY project_id, workflow_id, node_name, model, vendor, date
        """
    )

    # General LLM usage metrics
    client.command(
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS llm_usage_metrics
        ENGINE = SummingMergeTree()
        PARTITION BY (project_id, toYYYYMM(date))
        ORDER BY (project_id, vendor, model, date)
        AS SELECT
            project_id,
            JSONExtractString(attributes, 'llm.vendor') AS vendor,
            JSONExtractString(attributes, 'llm.model') AS model,
            toDate(started_at) AS date,
            count() AS call_count,
            sum(JSONExtractUInt(attributes, 'llm.prompt_tokens')) AS prompt_tokens,
            sum(JSONExtractUInt(attributes, 'llm.completion_tokens')) AS completion_tokens,
            sum(JSONExtractUInt(attributes, 'llm.total_tokens')) AS total_tokens,
            avg(JSONExtractFloat(attributes, 'llm.latency_ms')) AS avg_latency_ms,
            sum(JSONExtractFloat(attributes, 'llm.cost_usd')) AS total_cost_usd
        FROM spans
        WHERE span_type = 'llm'
        GROUP BY project_id, vendor, model, date
        """
    )

    # Agent baselines table for drift detection
    client.command(
        """
        CREATE TABLE IF NOT EXISTS agent_baselines (
            baseline_id String,
            project_id String,
            agent_name String,
            service_name String,
            window_start DateTime64(6),
            window_end DateTime64(6),
            sample_size UInt32,

            -- Duration metrics
            duration_mean Float64,
            duration_stddev Float64,
            duration_p50 Float64,
            duration_p95 Float64,
            duration_p99 Float64,
            duration_min Float64,
            duration_max Float64,

            -- Token usage metrics
            token_usage_mean Float64,
            token_usage_stddev Float64,
            token_usage_p50 Float64,
            token_usage_p95 Float64,

            -- Tool call metrics
            tool_calls_mean Float64,
            tool_calls_stddev Float64,

            -- Response metrics
            response_length_mean Float64,
            response_length_stddev Float64,

            -- Success metrics
            success_rate Float64,
            error_count UInt32,

            -- Cost metrics
            cost_mean Float64,
            cost_total Float64,

            -- Metadata
            created_at DateTime64(6) DEFAULT now64(6),
            updated_at DateTime64(6) DEFAULT now64(6)
        )
        ENGINE = ReplacingMergeTree(updated_at)
        ORDER BY (project_id, agent_name, service_name, window_start)
        PARTITION BY (project_id, toYYYYMM(window_start))
        TTL toDateTime(window_start) + INTERVAL 90 DAY
        """
    )

    # Drift alerts table
    client.command(
        """
        CREATE TABLE IF NOT EXISTS drift_alerts (
            alert_id String,
            project_id String,
            agent_name String,
            service_name String,
            baseline_id String,
            detected_at DateTime64(6),
            severity Enum8('low' = 1, 'medium' = 2, 'high' = 3, 'critical' = 4),
            status Enum8('active' = 1, 'acknowledged' = 2, 'dismissed' = 3, 'muted' = 4),
            anomalies String,  -- JSON array of anomaly objects
            root_causes String,  -- JSON array of root cause objects
            acknowledged_by Nullable(String),
            acknowledged_at Nullable(DateTime64(6)),
            dismissed_by Nullable(String),
            dismissed_at Nullable(DateTime64(6)),
            mute_until Nullable(DateTime64(6)),
            notes Nullable(String),
            created_at DateTime64(6) DEFAULT now64(6),
            updated_at DateTime64(6) DEFAULT now64(6)
        )
        ENGINE = ReplacingMergeTree(updated_at)
        ORDER BY (project_id, severity, detected_at, alert_id)
        PARTITION BY (project_id, toYYYYMM(detected_at))
        TTL toDateTime(detected_at) + INTERVAL 90 DAY
        """
    )

    # Alert rules table
    client.command(
        """
        CREATE TABLE IF NOT EXISTS alert_rules (
            rule_id String,
            project_id String,
            name String,
            description Nullable(String),
            enabled Boolean DEFAULT true,

            -- Rule conditions
            agent_name Nullable(String),  -- Null = all agents
            metric_name Nullable(String),  -- Null = all metrics
            severity_threshold Enum8('low' = 1, 'medium' = 2, 'high' = 3, 'critical' = 4),
            change_percent_min Nullable(Float64),  -- Minimum % change to trigger

            -- Notification configuration
            notify_email Boolean DEFAULT false,
            email_addresses Array(String),  -- List of email addresses
            notify_slack Boolean DEFAULT false,
            slack_webhook_url Nullable(String),
            slack_channel Nullable(String),

            -- Metadata
            created_by String,
            created_at DateTime64(6) DEFAULT now64(6),
            updated_at DateTime64(6) DEFAULT now64(6)
        )
        ENGINE = ReplacingMergeTree(updated_at)
        ORDER BY (project_id, enabled, rule_id)
        PARTITION BY project_id
        """
    )

    # Analysis results table (security scanning, hallucination, drift)
    client.command(
        """
        CREATE TABLE IF NOT EXISTS analysis_results (
            result_id String,
            trace_id String,
            project_id String,
            analysis_type String,
            result String,
            score Float64,
            created_at DateTime64(6) DEFAULT now64(6)
        )
        ENGINE = ReplacingMergeTree(created_at)
        ORDER BY (project_id, analysis_type, trace_id)
        TTL toDateTime(created_at) + INTERVAL 90 DAY
        """
    )

    # Batch replay jobs table
    client.command(
        """
        CREATE TABLE IF NOT EXISTS batch_replay_jobs (
            batch_id String,
            project_id String,
            user_id String,
            status Enum8('pending' = 1, 'running' = 2, 'completed' = 3, 'failed' = 4, 'partial' = 5),
            trace_ids Array(String),
            parameters String,
            total_traces UInt32,
            completed_traces UInt32 DEFAULT 0,
            failed_traces UInt32 DEFAULT 0,
            created_at DateTime64(6) DEFAULT now64(6),
            started_at Nullable(DateTime64(6)),
            completed_at Nullable(DateTime64(6)),
            error Nullable(String),
            summary String DEFAULT '{}'
        )
        ENGINE = MergeTree()
        ORDER BY (project_id, created_at, batch_id)
        PARTITION BY (project_id, toYYYYMM(created_at))
        TTL toDateTime(created_at) + INTERVAL 90 DAY
        """
    )

    # Migration: Add source column to existing tables
    try:
        client.command(
            "ALTER TABLE traces ADD COLUMN IF NOT EXISTS source String DEFAULT 'native'"
        )
        client.command(
            "ALTER TABLE spans ADD COLUMN IF NOT EXISTS source String DEFAULT 'native'"
        )
        logger.info("Source column migration applied (or already present)")
    except Exception as e:
        logger.warning(f"Source column migration skipped: {e}")

    # Migration: Add batch_id column to replay_executions for batch replay support
    try:
        client.command(
            "ALTER TABLE replay_executions ADD COLUMN IF NOT EXISTS batch_id Nullable(String)"
        )
        logger.info("batch_id column migration applied (or already present)")
    except Exception as e:
        logger.warning(f"batch_id column migration skipped: {e}")

    logger.info("ClickHouse schema initialized successfully")


async def query_traces(
    client: Client,
    project_id: str | None = None,
    service_name: str | None = None,
    agent_name: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Query traces with optional filters.

    Args:
        client: ClickHouse client instance.
        project_id: Filter by project ID.
        service_name: Filter by service name.
        agent_name: Filter by agent name (stored in attributes JSON as 'agent.name').
        start_time: Filter by start time (ISO format).
        end_time: Filter by end time (ISO format).
        status: Filter by trace status (e.g. 'success', 'error', 'failed', 'running').
        limit: Maximum number of traces to return.
        offset: Number of traces to skip (for pagination).

    Returns:
        List of trace records as dicts.
    """
    conditions = []
    params = {}

    if project_id:
        conditions.append("project_id = %(project_id)s")
        params["project_id"] = project_id

    if service_name:
        conditions.append("service_name = %(service_name)s")
        params["service_name"] = service_name

    if agent_name:
        # Match on trace-level attributes OR any span's attributes (many traces
        # store agent.name only on spans, not on the trace row itself)
        conditions.append(
            "(JSONExtractString(attributes, 'agent.name') = %(agent_name)s"
            " OR trace_id IN ("
            "SELECT DISTINCT trace_id FROM spans"
            " WHERE JSONExtractString(attributes, 'agent.name') = %(agent_name)s"
            "))"
        )
        params["agent_name"] = agent_name

    if start_time:
        conditions.append("started_at >= %(start_time)s")
        params["start_time"] = start_time

    if end_time:
        conditions.append("started_at <= %(end_time)s")
        params["end_time"] = end_time

    if status:
        conditions.append("status = %(status)s")
        params["status"] = status

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
        SELECT
            trace_id, project_id, service_name,
            started_at, completed_at, duration_ms,
            status, root_span_id, span_count,
            attributes, source, created_at
        FROM traces
        {where_clause}
        ORDER BY started_at DESC
        LIMIT %(limit)s OFFSET %(offset)s
    """
    params["limit"] = limit
    params["offset"] = offset

    result = client.query(query, parameters=params)
    columns = ["trace_id", "project_id", "service_name", "started_at", "completed_at",
               "duration_ms", "status", "root_span_id", "span_count", "attributes",
               "source", "created_at"]
    return [dict(zip(columns, row)) for row in result.result_rows]


async def query_spans(
    client: Client, trace_id: str
) -> list[dict[str, Any]]:
    """Query all spans for a given trace.

    Args:
        client: ClickHouse client instance.
        trace_id: Trace ID to query.

    Returns:
        List of span records as dicts.
    """
    query = """
        SELECT
            span_id, trace_id, project_id, parent_span_id,
            name, span_type, service_name,
            started_at, ended_at, duration_ms,
            status, attributes, events, replay_snapshot,
            source, created_at
        FROM spans
        WHERE trace_id = %(trace_id)s
        ORDER BY started_at ASC
    """

    result = client.query(query, parameters={"trace_id": trace_id})
    columns = ["span_id", "trace_id", "project_id", "parent_span_id",
               "name", "span_type", "service_name", "started_at", "ended_at",
               "duration_ms", "status", "attributes", "events", "replay_snapshot",
               "source", "created_at"]
    return [dict(zip(columns, row)) for row in result.result_rows]


async def insert_trace(client: Client, trace_data: dict[str, Any]) -> None:
    """Insert a trace record into ClickHouse.

    Args:
        client: ClickHouse client instance.
        trace_data: Trace data dictionary.
    """
    import json
    from datetime import datetime

    # Parse or use current timestamp
    started_at = trace_data.get("started_at")
    if isinstance(started_at, str):
        started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    elif started_at is None:
        started_at = datetime.utcnow()

    completed_at = trace_data.get("ended_at") or trace_data.get("completed_at")
    if isinstance(completed_at, str):
        completed_at = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))

    # Calculate duration if not provided
    duration_ms = trace_data.get("duration_ms", 0)
    if duration_ms == 0 and completed_at:
        duration_ms = (completed_at - started_at).total_seconds() * 1000

    client.insert(
        "traces",
        [[
            trace_data.get("trace_id", ""),
            trace_data.get("project_id", ""),
            trace_data.get("service_name", trace_data.get("attributes", {}).get("service.name", "")),
            started_at,
            completed_at,
            duration_ms,
            trace_data.get("status", "running"),
            trace_data.get("span_id", ""),  # root_span_id
            trace_data.get("span_count", 1),
            json.dumps(trace_data.get("attributes", {})),
        ]],
        column_names=[
            "trace_id", "project_id", "service_name", "started_at",
            "completed_at", "duration_ms", "status", "root_span_id",
            "span_count", "attributes"
        ]
    )
    logger.debug(f"Inserted trace: {trace_data.get('trace_id')}")


async def insert_span(client: Client, span_data: dict[str, Any]) -> None:
    """Insert a span record into ClickHouse.

    Args:
        client: ClickHouse client instance.
        span_data: Span data dictionary.
    """
    import json
    from datetime import datetime

    # Parse or use current timestamp
    started_at = span_data.get("started_at")
    if isinstance(started_at, str):
        started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    elif started_at is None:
        started_at = datetime.utcnow()

    ended_at = span_data.get("ended_at")
    if isinstance(ended_at, str):
        ended_at = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
    elif ended_at is None:
        ended_at = started_at

    # Calculate duration if not provided
    duration_ms = span_data.get("duration_ms", 0)
    if duration_ms == 0 and ended_at:
        duration_ms = (ended_at - started_at).total_seconds() * 1000

    client.insert(
        "spans",
        [[
            span_data.get("span_id", ""),
            span_data.get("trace_id", ""),
            span_data.get("project_id", ""),
            span_data.get("parent_span_id", ""),
            span_data.get("name", ""),
            span_data.get("span_type", "custom"),
            span_data.get("service_name", span_data.get("attributes", {}).get("service.name", "")),
            started_at,
            ended_at,
            duration_ms,
            span_data.get("status", "completed"),
            json.dumps(span_data.get("attributes", {})),
            json.dumps(span_data.get("events", [])),
            json.dumps(span_data.get("replay_snapshot", {})) if span_data.get("replay_snapshot") else "",
        ]],
        column_names=[
            "span_id", "trace_id", "project_id", "parent_span_id", "name",
            "span_type", "service_name", "started_at", "ended_at",
            "duration_ms", "status", "attributes", "events", "replay_snapshot"
        ]
    )
    logger.debug(f"Inserted span: {span_data.get('span_id')}")
