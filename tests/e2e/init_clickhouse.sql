-- ClickHouse schema for E2E tests
-- Core tables only (no materialized views for test speed)

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
ORDER BY (project_id, service_name, started_at, trace_id);

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
ORDER BY (project_id, trace_id, started_at, span_id);

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
ORDER BY (project_id, analysis_type, trace_id);

CREATE TABLE IF NOT EXISTS agent_baselines (
    baseline_id String,
    project_id String,
    agent_name String,
    service_name String,
    window_start DateTime64(6),
    window_end DateTime64(6),
    sample_size UInt32,
    duration_mean Float64,
    duration_stddev Float64,
    duration_p50 Float64,
    duration_p95 Float64,
    duration_p99 Float64,
    duration_min Float64,
    duration_max Float64,
    token_usage_mean Float64,
    token_usage_stddev Float64,
    token_usage_p50 Float64,
    token_usage_p95 Float64,
    tool_calls_mean Float64,
    tool_calls_stddev Float64,
    response_length_mean Float64,
    response_length_stddev Float64,
    success_rate Float64,
    error_count UInt32,
    cost_mean Float64,
    cost_total Float64,
    created_at DateTime64(6) DEFAULT now64(6),
    updated_at DateTime64(6) DEFAULT now64(6)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (project_id, agent_name, service_name, window_start);

CREATE TABLE IF NOT EXISTS drift_alerts (
    alert_id String,
    project_id String,
    agent_name String,
    service_name String,
    baseline_id String,
    detected_at DateTime64(6),
    severity Enum8('low' = 1, 'medium' = 2, 'high' = 3, 'critical' = 4),
    status Enum8('active' = 1, 'acknowledged' = 2, 'dismissed' = 3, 'muted' = 4),
    anomalies String,
    root_causes String,
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
PARTITION BY (project_id, toYYYYMM(toDateTime(detected_at)))
TTL toDateTime(detected_at) + INTERVAL 90 DAY;

CREATE TABLE IF NOT EXISTS alert_rules (
    rule_id String,
    project_id String,
    name String,
    description Nullable(String),
    enabled Boolean DEFAULT true,
    agent_name Nullable(String),
    metric_name Nullable(String),
    severity_threshold Enum8('low' = 1, 'medium' = 2, 'high' = 3, 'critical' = 4),
    change_percent_min Nullable(Float64),
    notify_email Boolean DEFAULT false,
    email_addresses Array(String),
    notify_slack Boolean DEFAULT false,
    slack_webhook_url Nullable(String),
    slack_channel Nullable(String),
    created_by String,
    created_at DateTime64(6) DEFAULT now64(6),
    updated_at DateTime64(6) DEFAULT now64(6)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (project_id, enabled, rule_id)
PARTITION BY project_id;
