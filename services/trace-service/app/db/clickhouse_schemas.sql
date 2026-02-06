-- ClickHouse Schema for Prela Observability Platform
-- This file contains table definitions and materialized views for trace storage and analytics

-- =============================================================================
-- CORE TABLES
-- =============================================================================

-- Traces table (workflow/execution level)
CREATE TABLE IF NOT EXISTS traces (
    trace_id String,
    project_id String,
    service_name String,
    started_at DateTime64(6),
    completed_at DateTime64(6),
    duration_ms Float64,
    status String,  -- 'running', 'completed', 'error'
    root_span_id String,
    span_count UInt32,
    attributes String,  -- JSON string with flexible metadata
    created_at DateTime64(6) DEFAULT now64(6)
)
ENGINE = MergeTree()
ORDER BY (project_id, service_name, started_at, trace_id)
PARTITION BY (project_id, toYYYYMM(started_at))
TTL started_at + INTERVAL 90 DAY
COMMENT 'Top-level traces representing complete workflow executions';

-- Spans table (operation level)
CREATE TABLE IF NOT EXISTS spans (
    span_id String,
    trace_id String,
    project_id String,
    parent_span_id String,
    name String,
    span_type String,  -- 'agent', 'llm', 'tool', 'retrieval', 'embedding', 'custom'
    service_name String,
    started_at DateTime64(6),
    ended_at DateTime64(6),
    duration_ms Float64,
    status String,  -- 'pending', 'completed', 'error'
    attributes String,  -- JSON string with span-specific attributes
    events String,  -- JSON array of timestamped events
    created_at DateTime64(6) DEFAULT now64(6)
)
ENGINE = MergeTree()
ORDER BY (project_id, trace_id, started_at, span_id)
PARTITION BY (project_id, toYYYYMM(started_at))
TTL started_at + INTERVAL 90 DAY
COMMENT 'Individual operations within traces (LLM calls, tool usage, etc.)';

-- Span attributes table (for fast filtering by specific attributes)
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
COMMENT 'Extracted key-value pairs from span attributes for fast filtering';

-- =============================================================================
-- N8N MATERIALIZED VIEWS
-- =============================================================================

-- Materialized view for n8n workflow metrics
-- Aggregates workflow-level performance metrics by day
CREATE MATERIALIZED VIEW IF NOT EXISTS n8n_workflow_metrics
ENGINE = SummingMergeTree()
PARTITION BY (project_id, toYYYYMM(date))
ORDER BY (project_id, workflow_id, date)
COMMENT 'Daily aggregated metrics for n8n workflows'
AS SELECT
    project_id,
    JSONExtractString(attributes, 'n8n.workflow.id') AS workflow_id,
    JSONExtractString(attributes, 'n8n.workflow.name') AS workflow_name,
    toDate(started_at) AS date,
    count() AS execution_count,
    countIf(status = 'error') AS error_count,
    countIf(status = 'completed') AS success_count,
    avg(duration_ms) AS avg_duration_ms,
    quantile(0.50)(duration_ms) AS p50_duration_ms,
    quantile(0.95)(duration_ms) AS p95_duration_ms,
    quantile(0.99)(duration_ms) AS p99_duration_ms,
    max(duration_ms) AS max_duration_ms,
    min(duration_ms) AS min_duration_ms
FROM traces
WHERE JSONHas(attributes, 'n8n.workflow.id')
GROUP BY project_id, workflow_id, workflow_name, date;

-- Materialized view for n8n AI node usage
-- Aggregates AI/LLM call metrics by workflow, node, and model
CREATE MATERIALIZED VIEW IF NOT EXISTS n8n_ai_node_metrics
ENGINE = SummingMergeTree()
PARTITION BY (project_id, toYYYYMM(date))
ORDER BY (project_id, workflow_id, node_name, model, date)
COMMENT 'Daily aggregated metrics for AI nodes in n8n workflows'
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
    quantile(0.95)(JSONExtractFloat(attributes, 'llm.latency_ms')) AS p95_latency_ms,
    sum(JSONExtractFloat(attributes, 'llm.cost_usd')) AS total_cost_usd,
    countIf(status = 'error') AS error_count
FROM spans
WHERE span_type = 'llm' AND JSONHas(attributes, 'n8n.workflow.id')
GROUP BY project_id, workflow_id, node_name, model, vendor, date;

-- Materialized view for n8n node execution metrics
-- Aggregates metrics for all node types (tools, retrievals, custom)
CREATE MATERIALIZED VIEW IF NOT EXISTS n8n_node_metrics
ENGINE = SummingMergeTree()
PARTITION BY (project_id, toYYYYMM(date))
ORDER BY (project_id, workflow_id, node_name, span_type, date)
COMMENT 'Daily aggregated metrics for all n8n nodes'
AS SELECT
    project_id,
    JSONExtractString(attributes, 'n8n.workflow.id') AS workflow_id,
    JSONExtractString(attributes, 'n8n.node.name') AS node_name,
    JSONExtractString(attributes, 'n8n.node.type') AS node_type,
    span_type,
    toDate(started_at) AS date,
    count() AS execution_count,
    avg(duration_ms) AS avg_duration_ms,
    quantile(0.95)(duration_ms) AS p95_duration_ms,
    countIf(status = 'error') AS error_count,
    countIf(status = 'completed') AS success_count
FROM spans
WHERE JSONHas(attributes, 'n8n.workflow.id') AND span_type != 'llm'
GROUP BY project_id, workflow_id, node_name, node_type, span_type, date;

-- =============================================================================
-- GENERAL METRICS VIEWS
-- =============================================================================

-- Daily span metrics by service and type
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_span_metrics
ENGINE = SummingMergeTree()
PARTITION BY (project_id, toYYYYMM(date))
ORDER BY (project_id, service_name, span_type, date)
COMMENT 'Daily aggregated metrics for spans by service and type'
AS SELECT
    project_id,
    service_name,
    span_type,
    toDate(started_at) AS date,
    count() AS span_count,
    avg(duration_ms) AS avg_duration_ms,
    quantile(0.95)(duration_ms) AS p95_duration_ms,
    countIf(status = 'error') AS error_count
FROM spans
GROUP BY project_id, service_name, span_type, date;

-- LLM usage metrics across all services
CREATE MATERIALIZED VIEW IF NOT EXISTS llm_usage_metrics
ENGINE = SummingMergeTree()
PARTITION BY (project_id, toYYYYMM(date))
ORDER BY (project_id, vendor, model, date)
COMMENT 'Daily aggregated LLM usage and cost metrics'
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
GROUP BY project_id, vendor, model, date;

-- =============================================================================
-- EXAMPLE QUERIES
-- =============================================================================

-- Query: Get n8n workflow performance over the last 7 days
-- SELECT
--     workflow_name,
--     date,
--     execution_count,
--     error_count,
--     round(error_count / execution_count * 100, 2) AS error_rate_pct,
--     round(avg_duration_ms, 2) AS avg_duration_ms,
--     round(p95_duration_ms, 2) AS p95_duration_ms
-- FROM n8n_workflow_metrics
-- WHERE project_id = 'my-project' AND date >= today() - 7
-- ORDER BY date DESC, workflow_name;

-- Query: Get AI node usage by model for a specific workflow
-- SELECT
--     model,
--     vendor,
--     sum(call_count) AS total_calls,
--     sum(prompt_tokens) AS total_prompt_tokens,
--     sum(completion_tokens) AS total_completion_tokens,
--     sum(total_tokens) AS total_tokens,
--     round(sum(total_cost_usd), 4) AS total_cost_usd,
--     round(avg(avg_latency_ms), 2) AS avg_latency_ms
-- FROM n8n_ai_node_metrics
-- WHERE project_id = 'my-project'
--   AND workflow_id = 'wf-123'
--   AND date >= today() - 30
-- GROUP BY model, vendor
-- ORDER BY total_cost_usd DESC;

-- Query: Get most expensive workflows by AI usage
-- SELECT
--     workflow_name,
--     sum(call_count) AS ai_calls,
--     sum(total_tokens) AS tokens_used,
--     sum(total_cost_usd) AS total_cost_usd
-- FROM n8n_ai_node_metrics m
-- JOIN (
--     SELECT DISTINCT workflow_id, any(workflow_name) AS workflow_name
--     FROM n8n_workflow_metrics
--     WHERE project_id = 'my-project'
--     GROUP BY workflow_id
-- ) w ON m.workflow_id = w.workflow_id
-- WHERE m.project_id = 'my-project'
--   AND m.date >= today() - 30
-- GROUP BY workflow_name
-- ORDER BY total_cost_usd DESC
-- LIMIT 10;

-- Query: Get error rate trends for workflows
-- SELECT
--     workflow_name,
--     date,
--     execution_count,
--     error_count,
--     round(error_count / execution_count * 100, 2) AS error_rate_pct
-- FROM n8n_workflow_metrics
-- WHERE project_id = 'my-project'
--   AND date >= today() - 30
-- ORDER BY date DESC, error_rate_pct DESC;

-- Query: Get slowest n8n nodes
-- SELECT
--     node_name,
--     node_type,
--     span_type,
--     sum(execution_count) AS total_executions,
--     round(avg(avg_duration_ms), 2) AS avg_duration_ms,
--     round(avg(p95_duration_ms), 2) AS p95_duration_ms
-- FROM n8n_node_metrics
-- WHERE project_id = 'my-project'
--   AND date >= today() - 7
-- GROUP BY node_name, node_type, span_type
-- ORDER BY avg_duration_ms DESC
-- LIMIT 20;

-- Query: Compare LLM vendors by cost and latency
-- SELECT
--     vendor,
--     count(DISTINCT model) AS model_count,
--     sum(call_count) AS total_calls,
--     sum(total_tokens) AS total_tokens,
--     round(sum(total_cost_usd), 2) AS total_cost_usd,
--     round(avg(avg_latency_ms), 2) AS avg_latency_ms
-- FROM llm_usage_metrics
-- WHERE project_id = 'my-project'
--   AND date >= today() - 30
-- GROUP BY vendor
-- ORDER BY total_cost_usd DESC;

-- =============================================================================
-- MAINTENANCE
-- =============================================================================

-- Drop all tables and views (use with caution!)
-- DROP VIEW IF EXISTS n8n_workflow_metrics;
-- DROP VIEW IF EXISTS n8n_ai_node_metrics;
-- DROP VIEW IF EXISTS n8n_node_metrics;
-- DROP VIEW IF EXISTS daily_span_metrics;
-- DROP VIEW IF EXISTS llm_usage_metrics;
-- DROP TABLE IF EXISTS span_attributes;
-- DROP TABLE IF EXISTS spans;
-- DROP TABLE IF EXISTS traces;

-- Optimize tables (run periodically to improve query performance)
-- OPTIMIZE TABLE traces FINAL;
-- OPTIMIZE TABLE spans FINAL;
-- OPTIMIZE TABLE span_attributes FINAL;
