"""ClickHouse Cloud client utilities."""

import logging
from typing import Any

import clickhouse_connect
from clickhouse_connect.driver import Client

from .config import settings

logger = logging.getLogger(__name__)


def get_clickhouse_client() -> Client:
    """Get ClickHouse Cloud client with secure connection.

    Returns:
        ClickHouse client instance configured for ClickHouse Cloud.

    Raises:
        Exception: If connection fails.
    """
    try:
        client = clickhouse_connect.get_client(
            host=settings.clickhouse_host,
            port=settings.clickhouse_port,
            username=settings.clickhouse_user,
            password=settings.clickhouse_password,
            database=settings.clickhouse_database,
            secure=True,  # Required for ClickHouse Cloud
        )
        logger.info(f"Connected to ClickHouse Cloud: {settings.clickhouse_host}")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to ClickHouse: {e}")
        raise


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
        TTL created_at + INTERVAL 365 DAY
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
            created_at DateTime64(6) DEFAULT now64(6)
        )
        ENGINE = MergeTree()
        ORDER BY (project_id, service_name, started_at, trace_id)
        PARTITION BY (project_id, toYYYYMM(started_at))
        TTL started_at + INTERVAL 90 DAY
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
            created_at DateTime64(6) DEFAULT now64(6)
        )
        ENGINE = MergeTree()
        ORDER BY (project_id, trace_id, started_at, span_id)
        PARTITION BY (project_id, toYYYYMM(started_at))
        TTL started_at + INTERVAL 90 DAY
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
        TTL started_at + INTERVAL 90 DAY
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
        TTL triggered_at + INTERVAL 90 DAY
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
            countIf(status = 'completed') AS success_count,
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
        TTL window_start + INTERVAL 90 DAY
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
        TTL detected_at + INTERVAL 90 DAY
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
            severity_threshold Enum8('low' = .1, 'medium' = 2, 'high' = 3, 'critical' = 4),
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

    logger.info("ClickHouse schema initialized successfully")


async def query_traces(
    client: Client,
    service_name: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Query traces with optional filters.

    Args:
        client: ClickHouse client instance.
        service_name: Filter by service name.
        start_time: Filter by start time (ISO format).
        end_time: Filter by end time (ISO format).
        limit: Maximum number of traces to return.

    Returns:
        List of trace records.
    """
    conditions = []
    params = {}

    if service_name:
        conditions.append("service_name = %(service_name)s")
        params["service_name"] = service_name

    if start_time:
        conditions.append("started_at >= %(start_time)s")
        params["start_time"] = start_time

    if end_time:
        conditions.append("started_at <= %(end_time)s")
        params["end_time"] = end_time

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
        SELECT *
        FROM traces
        {where_clause}
        ORDER BY started_at DESC
        LIMIT %(limit)s
    """
    params["limit"] = limit

    result = client.query(query, parameters=params)
    return result.result_rows


async def query_spans(
    client: Client, trace_id: str
) -> list[dict[str, Any]]:
    """Query all spans for a given trace.

    Args:
        client: ClickHouse client instance.
        trace_id: Trace ID to query.

    Returns:
        List of span records.
    """
    query = """
        SELECT *
        FROM spans
        WHERE trace_id = %(trace_id)s
        ORDER BY started_at ASC
    """

    result = client.query(query, parameters={"trace_id": trace_id})
    return result.result_rows
