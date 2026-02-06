# ClickHouse Database Schemas

This directory contains the ClickHouse schema definitions for the Prela observability platform.

## Files

- **`clickhouse_schemas.sql`** - Complete SQL schema with tables, materialized views, and example queries

## Schema Overview

### Core Tables

#### 1. `traces` Table
Top-level traces representing complete workflow executions.

**Columns:**
- `trace_id` - Unique trace identifier
- `project_id` - Project/tenant identifier
- `service_name` - Service generating the trace (e.g., "n8n", "my-agent")
- `started_at` - Trace start timestamp (microsecond precision)
- `completed_at` - Trace completion timestamp
- `duration_ms` - Total duration in milliseconds
- `status` - Trace status: 'running', 'completed', 'error'
- `root_span_id` - ID of the root span
- `span_count` - Number of child spans
- `attributes` - JSON string with flexible metadata
- `created_at` - Record creation timestamp

**Partitioning:** By `project_id` and month (`toYYYYMM(started_at)`)
**TTL:** 90 days from `started_at`

#### 2. `spans` Table
Individual operations within traces (LLM calls, tool usage, etc.).

**Columns:**
- `span_id` - Unique span identifier
- `trace_id` - Parent trace identifier
- `project_id` - Project identifier
- `parent_span_id` - Parent span ID (for hierarchy)
- `name` - Span name (e.g., "openai.chat.completions.create")
- `span_type` - Type: 'agent', 'llm', 'tool', 'retrieval', 'embedding', 'custom'
- `service_name` - Service name
- `started_at` - Span start timestamp
- `ended_at` - Span end timestamp
- `duration_ms` - Duration in milliseconds
- `status` - Span status: 'pending', 'completed', 'error'
- `attributes` - JSON string with span-specific attributes
- `events` - JSON array of timestamped events
- `created_at` - Record creation timestamp

**Partitioning:** By `project_id` and month
**TTL:** 90 days from `started_at`

#### 3. `span_attributes` Table
Extracted key-value pairs from span attributes for fast filtering.

**Columns:**
- `span_id`, `trace_id`, `project_id`
- `key` - Attribute key
- `value` - Attribute value (as string)
- `started_at` - Timestamp for partitioning

**Partitioning:** By `project_id` and month
**TTL:** 90 days

---

## Materialized Views

### n8n-Specific Views

#### 1. `n8n_workflow_metrics`
Daily aggregated metrics for n8n workflows.

**Columns:**
- `project_id`, `workflow_id`, `workflow_name`, `date`
- `execution_count` - Total executions
- `error_count` - Failed executions
- `success_count` - Successful executions
- `avg_duration_ms` - Average duration
- `p50_duration_ms`, `p95_duration_ms`, `p99_duration_ms` - Percentiles
- `max_duration_ms`, `min_duration_ms`

**Engine:** SummingMergeTree (aggregates on query)

**Example Query:**
```sql
SELECT
    workflow_name,
    date,
    execution_count,
    round(error_count / execution_count * 100, 2) AS error_rate_pct,
    round(avg_duration_ms, 2) AS avg_duration_ms
FROM n8n_workflow_metrics
WHERE project_id = 'my-project' AND date >= today() - 7
ORDER BY date DESC;
```

#### 2. `n8n_ai_node_metrics`
Daily aggregated metrics for AI nodes in n8n workflows.

**Columns:**
- `project_id`, `workflow_id`, `node_name`, `model`, `vendor`, `date`
- `call_count` - Number of AI calls
- `prompt_tokens`, `completion_tokens`, `total_tokens` - Token usage
- `avg_latency_ms`, `p95_latency_ms` - Latency metrics
- `total_cost_usd` - Total cost in USD
- `error_count` - Failed calls

**Engine:** SummingMergeTree

**Example Query:**
```sql
SELECT
    model,
    vendor,
    sum(call_count) AS total_calls,
    sum(total_tokens) AS total_tokens,
    round(sum(total_cost_usd), 4) AS total_cost_usd
FROM n8n_ai_node_metrics
WHERE project_id = 'my-project'
  AND workflow_id = 'wf-123'
  AND date >= today() - 30
GROUP BY model, vendor
ORDER BY total_cost_usd DESC;
```

#### 3. `n8n_node_metrics`
Daily aggregated metrics for all n8n nodes (non-LLM).

**Columns:**
- `project_id`, `workflow_id`, `node_name`, `node_type`, `span_type`, `date`
- `execution_count` - Total executions
- `avg_duration_ms`, `p95_duration_ms` - Duration metrics
- `error_count`, `success_count` - Success/failure counts

### General Metrics Views

#### 4. `daily_span_metrics`
Daily span metrics by service and type.

#### 5. `llm_usage_metrics`
Daily LLM usage and cost metrics across all services.

---

## Schema Initialization

### Automatic (via Python)

The schema is automatically initialized when the Trace Service starts:

```python
from shared import get_clickhouse_client, init_clickhouse_schema

client = get_clickhouse_client()
await init_clickhouse_schema(client)
```

This creates all tables and materialized views defined in `backend/services/shared/clickhouse.py`.

### Manual (via SQL)

To manually create the schema:

```bash
# Connect to ClickHouse
clickhouse-client --host your-host --secure --user default --password your-password

# Load and execute schema
SOURCE /path/to/clickhouse_schemas.sql;
```

---

## Example Queries

### 1. Get n8n Workflow Performance

```sql
SELECT
    workflow_name,
    date,
    execution_count,
    error_count,
    round(error_count / execution_count * 100, 2) AS error_rate_pct,
    round(avg_duration_ms, 2) AS avg_duration_ms,
    round(p95_duration_ms, 2) AS p95_duration_ms
FROM n8n_workflow_metrics
WHERE project_id = 'my-project' AND date >= today() - 7
ORDER BY date DESC, workflow_name;
```

### 2. Get Most Expensive Workflows

```sql
SELECT
    workflow_name,
    sum(call_count) AS ai_calls,
    sum(total_tokens) AS tokens_used,
    round(sum(total_cost_usd), 2) AS total_cost_usd
FROM n8n_ai_node_metrics
WHERE project_id = 'my-project'
  AND date >= today() - 30
GROUP BY workflow_name
ORDER BY total_cost_usd DESC
LIMIT 10;
```

### 3. Compare LLM Vendors

```sql
SELECT
    vendor,
    count(DISTINCT model) AS model_count,
    sum(call_count) AS total_calls,
    sum(total_tokens) AS total_tokens,
    round(sum(total_cost_usd), 2) AS total_cost_usd,
    round(avg(avg_latency_ms), 2) AS avg_latency_ms
FROM llm_usage_metrics
WHERE project_id = 'my-project'
  AND date >= today() - 30
GROUP BY vendor
ORDER BY total_cost_usd DESC;
```

### 4. Get Slowest n8n Nodes

```sql
SELECT
    node_name,
    node_type,
    span_type,
    sum(execution_count) AS total_executions,
    round(avg(avg_duration_ms), 2) AS avg_duration_ms,
    round(avg(p95_duration_ms), 2) AS p95_duration_ms
FROM n8n_node_metrics
WHERE project_id = 'my-project'
  AND date >= today() - 7
GROUP BY node_name, node_type, span_type
ORDER BY avg_duration_ms DESC
LIMIT 20;
```

---

## Data Retention

All tables use TTL (Time-To-Live) to automatically delete old data:

- **Retention Period:** 90 days
- **Based On:** `started_at` timestamp
- **Automatic:** ClickHouse handles cleanup in background

To change retention:

```sql
-- Extend to 180 days
ALTER TABLE traces MODIFY TTL started_at + INTERVAL 180 DAY;
ALTER TABLE spans MODIFY TTL started_at + INTERVAL 180 DAY;
```

---

## Performance Optimization

### 1. Regular Optimization

Run periodically to improve query performance:

```sql
OPTIMIZE TABLE traces FINAL;
OPTIMIZE TABLE spans FINAL;
OPTIMIZE TABLE span_attributes FINAL;
```

### 2. Materialized Views

Materialized views are automatically updated as data is inserted. No manual refresh needed.

### 3. Partitioning

Tables are partitioned by:
- `project_id` - For multi-tenant isolation
- `toYYYYMM(started_at)` - Monthly partitions

This enables:
- Fast queries filtering by project
- Efficient data deletion (drop old partitions)
- Parallel query execution

---

## Maintenance

### Drop and Recreate

**⚠️ WARNING: This deletes ALL data!**

```sql
-- Drop materialized views first
DROP VIEW IF EXISTS n8n_workflow_metrics;
DROP VIEW IF EXISTS n8n_ai_node_metrics;
DROP VIEW IF EXISTS n8n_node_metrics;
DROP VIEW IF EXISTS daily_span_metrics;
DROP VIEW IF EXISTS llm_usage_metrics;

-- Drop tables
DROP TABLE IF EXISTS span_attributes;
DROP TABLE IF EXISTS spans;
DROP TABLE IF EXISTS traces;

-- Recreate (via Python init or manual SQL)
```

### Check Table Sizes

```sql
SELECT
    table,
    formatReadableSize(sum(bytes)) AS size,
    sum(rows) AS rows
FROM system.parts
WHERE database = 'prela'
GROUP BY table
ORDER BY sum(bytes) DESC;
```

### Check Partition Info

```sql
SELECT
    table,
    partition,
    sum(rows) AS rows,
    formatReadableSize(sum(bytes)) AS size
FROM system.parts
WHERE database = 'prela'
GROUP BY table, partition
ORDER BY table, partition;
```

---

## Development

### Adding New Materialized Views

1. Define view in `clickhouse_schemas.sql`
2. Add creation command to `init_clickhouse_schema()` in `shared/clickhouse.py`
3. Test with sample data
4. Deploy to production

### Schema Migration

ClickHouse doesn't support complex schema migrations. To modify tables:

1. Create new table with desired schema
2. Copy data: `INSERT INTO new_table SELECT * FROM old_table`
3. Drop old table
4. Rename new table

For adding columns (simpler):
```sql
ALTER TABLE traces ADD COLUMN new_field String DEFAULT '';
```

---

## Monitoring

### Query Performance

```sql
SELECT
    query,
    query_duration_ms,
    read_rows,
    formatReadableSize(read_bytes) AS read_size,
    result_rows
FROM system.query_log
WHERE type = 'QueryFinish'
  AND query_duration_ms > 1000
ORDER BY query_duration_ms DESC
LIMIT 10;
```

### Slow Queries

```sql
SELECT
    query_id,
    user,
    query,
    query_duration_ms,
    memory_usage
FROM system.query_log
WHERE query_duration_ms > 5000
ORDER BY query_start_time DESC
LIMIT 20;
```

---

## Best Practices

1. **Always filter by `project_id`** - Leverages partitioning
2. **Use date ranges** - Avoids scanning entire dataset
3. **Aggregate before filtering** - Use materialized views when possible
4. **Avoid SELECT *** - Specify columns explicitly
5. **Use appropriate ORDER BY** - Matches table ORDER BY for best performance

---

## Support

For questions or issues with the schema:
- Check `clickhouse_schemas.sql` for full definitions
- Review example queries in SQL file
- Contact backend team for schema changes