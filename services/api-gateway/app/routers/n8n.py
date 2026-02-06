"""n8n integration routes for Prela API Gateway."""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from shared import cache_delete, cache_get, cache_set, settings
from shared.clickhouse import get_clickhouse_client, insert_span, insert_trace
from shared.validation import InputValidator
from shared.webhook_rate_limiter import get_webhook_rate_limiter

router = APIRouter()
logger = logging.getLogger(__name__)


# Authentication dependency
async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """Verify API key from request header.

    Args:
        x_api_key: API key from X-API-Key header.

    Returns:
        API key if valid.

    Raises:
        HTTPException: If API key is invalid or missing.
    """
    # TODO: Implement proper API key validation against database
    # For now, accept any non-empty key
    if not x_api_key or len(x_api_key) < 10:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# Request/Response Models


class N8nTraceStartRequest(BaseModel):
    """Request to start a new n8n workflow trace."""

    project_id: str = Field(..., description="Project ID for organizing traces")
    workflow_id: str = Field(..., description="n8n workflow ID")
    workflow_name: str = Field(..., description="Human-readable workflow name")
    execution_id: str = Field(..., description="n8n execution ID")
    trace_name: Optional[str] = Field(None, description="Custom trace name")
    attributes: Optional[dict[str, Any]] = Field(None, description="Additional metadata")


class N8nTraceEndRequest(BaseModel):
    """Request to end an n8n workflow trace."""

    project_id: str = Field(..., description="Project ID")
    execution_id: str = Field(..., description="n8n execution ID")
    status: str = Field(..., description="Execution status: 'success' or 'error'")
    error_message: Optional[str] = Field(None, description="Error message if status is error")


class N8nSpanRequest(BaseModel):
    """Request to log a custom span within an n8n trace."""

    project_id: str = Field(..., description="Project ID")
    execution_id: str = Field(..., description="n8n execution ID")
    span_name: str = Field(..., description="Span name")
    span_type: str = Field(..., description="Span type (agent, tool, custom, etc.)")
    input_data: Optional[Any] = Field(None, description="Input data for the span")
    output_data: Optional[Any] = Field(None, description="Output data from the span")
    attributes: Optional[dict[str, Any]] = Field(None, description="Additional span attributes")


class N8nAICallRequest(BaseModel):
    """Request to log an AI/LLM call from n8n."""

    project_id: str = Field(..., description="Project ID")
    execution_id: str = Field(..., description="n8n execution ID")
    model: str = Field(..., description="Model name (e.g., 'gpt-4', 'claude-3-sonnet')")
    provider: str = Field(..., description="Provider name (e.g., 'openai', 'anthropic')")
    prompt: Any = Field(..., description="Prompt sent to the LLM")
    response: Any = Field(..., description="Response from the LLM")
    token_usage: Optional[dict] = Field(None, description="Token usage statistics")
    latency_ms: Optional[float] = Field(None, description="Call latency in milliseconds")
    attributes: Optional[dict[str, Any]] = Field(None, description="Additional attributes")


class N8nWebhookPayload(BaseModel):
    """Standard n8n webhook payload format."""

    workflow: dict = Field(..., description="Workflow metadata from $workflow")
    execution: dict = Field(..., description="Execution metadata from $execution")
    node: Optional[dict] = Field(None, description="Node metadata from $node")
    data: Optional[list] = Field(None, description="Node output data from $json")


# Analytics Response Models


class N8nWorkflow(BaseModel):
    """n8n workflow summary with 24h metrics."""

    workflow_id: str
    workflow_name: str
    last_execution: str
    execution_count_24h: int
    success_rate_24h: float
    avg_duration_ms: float
    total_ai_calls_24h: int
    total_tokens_24h: int
    total_cost_24h: float


class N8nExecution(BaseModel):
    """n8n workflow execution details."""

    execution_id: str
    started_at: str
    ended_at: str
    status: str
    duration_ms: float
    total_tokens: int
    cost_usd: float


class N8nAINode(BaseModel):
    """n8n AI node usage metrics."""

    node_name: str
    model: str
    vendor: str
    call_count: int
    prompt_tokens: int
    completion_tokens: int
    avg_latency_ms: float


class WorkflowListResponse(BaseModel):
    """Response for workflow list endpoint."""

    workflows: List[N8nWorkflow]


class WorkflowDetailResponse(BaseModel):
    """Response for workflow detail endpoint."""

    workflow: N8nWorkflow
    executions: List[N8nExecution]
    ai_nodes: List[N8nAINode]


class ExecutionListResponse(BaseModel):
    """Response for execution list endpoint."""

    executions: List[N8nExecution]


class AINodeListResponse(BaseModel):
    """Response for AI node metrics endpoint."""

    nodes: List[N8nAINode]


class TimelineNode(BaseModel):
    """Timeline node execution data."""

    node_id: str
    node_name: str
    node_type: str
    start_offset_ms: float
    duration_ms: float
    status: str
    is_ai_node: bool


class ExecutionTimelineResponse(BaseModel):
    """Response for execution timeline endpoint."""

    execution_id: str
    total_duration_ms: float
    nodes: List[TimelineNode]


# Routes


@router.post("/traces/start")
async def start_n8n_trace(
    request: N8nTraceStartRequest,
    api_key: str = Depends(verify_api_key),
):
    """Start a new trace for an n8n workflow execution.

    Creates the root span for the workflow and stores the mapping between
    execution_id and trace_id for later lookups.

    Returns:
        Dictionary with trace_id, span_id, and status.
    """
    try:
        trace_id = f"n8n-{request.execution_id}"
        span_id = str(uuid.uuid4())

        trace_data = {
            "trace_id": trace_id,
            "span_id": span_id,
            "project_id": request.project_id,
            "name": request.trace_name or request.workflow_name,
            "span_type": "agent",
            "started_at": datetime.utcnow().isoformat(),
            "attributes": {
                "n8n.workflow.id": request.workflow_id,
                "n8n.workflow.name": request.workflow_name,
                "n8n.execution.id": request.execution_id,
                "service.name": "n8n",
                **(request.attributes or {}),
            },
            "status": "running",
        }

        # Store execution_id -> (trace_id, span_id) mapping in Redis (24h TTL)
        mapping = json.dumps({"trace_id": trace_id, "span_id": span_id})
        await cache_set(f"n8n:execution:{request.execution_id}", mapping, ttl=86400)

        # Insert trace directly to ClickHouse
        client = get_clickhouse_client()
        await insert_trace(client, trace_data)

        logger.info(f"n8n trace started: {trace_id} (execution: {request.execution_id})")

        return {
            "trace_id": trace_id,
            "span_id": span_id,
            "status": "started",
        }

    except Exception as e:
        logger.error(f"Failed to start n8n trace: {e}")
        raise HTTPException(status_code=500, detail="Failed to start trace")


@router.post("/traces/end")
async def end_n8n_trace(
    request: N8nTraceEndRequest,
    api_key: str = Depends(verify_api_key),
):
    """End an n8n workflow trace.

    Updates the root span with completion status and cleans up the execution mapping.

    Returns:
        Dictionary with status.
    """
    try:
        # Look up trace_id from execution_id
        mapping_str = await cache_get(f"n8n:execution:{request.execution_id}")
        if not mapping_str:
            raise HTTPException(
                status_code=404,
                detail=f"Trace not found for execution {request.execution_id}",
            )

        mapping = json.loads(mapping_str)

        update_data = {
            "trace_id": mapping["trace_id"],
            "span_id": mapping["span_id"],
            "project_id": request.project_id,
            "ended_at": datetime.utcnow().isoformat(),
            "status": "completed" if request.status == "success" else "error",
        }

        if request.error_message:
            update_data["error_message"] = request.error_message

        # Insert span update to ClickHouse
        client = get_clickhouse_client()
        await insert_span(client, update_data)

        # Clean up mapping after a delay (keep for 1 more hour for late spans)
        await cache_set(
            f"n8n:execution:{request.execution_id}",
            mapping_str,
            ttl=3600,
        )

        logger.info(
            f"n8n trace ended: {mapping['trace_id']} "
            f"(execution: {request.execution_id}, status: {request.status})"
        )

        return {"status": "ended", "trace_id": mapping["trace_id"]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to end n8n trace: {e}")
        raise HTTPException(status_code=500, detail="Failed to end trace")


@router.post("/spans")
async def log_n8n_span(
    request: N8nSpanRequest,
    api_key: str = Depends(verify_api_key),
):
    """Log a custom span within an n8n trace.

    Creates a child span under the workflow root span.

    Returns:
        Dictionary with span_id.
    """
    try:
        # Look up trace_id from execution_id
        mapping_str = await cache_get(f"n8n:execution:{request.execution_id}")
        if not mapping_str:
            raise HTTPException(
                status_code=404,
                detail=f"Trace not found for execution {request.execution_id}",
            )

        mapping = json.loads(mapping_str)
        span_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        span_data = {
            "trace_id": mapping["trace_id"],
            "span_id": span_id,
            "parent_span_id": mapping["span_id"],  # Child of root workflow span
            "project_id": request.project_id,
            "name": request.span_name,
            "span_type": request.span_type,
            "started_at": now,
            "ended_at": now,  # Instant span
            "attributes": {
                "n8n.execution.id": request.execution_id,
                "service.name": "n8n",
                **(request.attributes or {}),
            },
            "status": "completed",
        }

        # Add input/output as events if provided
        events = []
        if request.input_data is not None:
            events.append({
                "name": "input",
                "timestamp": now,
                "attributes": {"data": request.input_data},
            })
        if request.output_data is not None:
            events.append({
                "name": "output",
                "timestamp": now,
                "attributes": {"data": request.output_data},
            })

        if events:
            span_data["events"] = events

        # Insert span to ClickHouse
        client = get_clickhouse_client()
        await insert_span(client, span_data)

        logger.debug(f"n8n span logged: {span_id} ({request.span_name})")

        return {"span_id": span_id, "trace_id": mapping["trace_id"]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to log n8n span: {e}")
        raise HTTPException(status_code=500, detail="Failed to log span")


@router.post("/ai-calls")
async def log_n8n_ai_call(
    request: N8nAICallRequest,
    api_key: str = Depends(verify_api_key),
):
    """Log an AI/LLM call from an n8n workflow.

    Creates an LLM-type span with full details including prompts, responses,
    and token usage.

    Returns:
        Dictionary with span_id.
    """
    try:
        # Look up trace_id from execution_id
        mapping_str = await cache_get(f"n8n:execution:{request.execution_id}")
        if not mapping_str:
            raise HTTPException(
                status_code=404,
                detail=f"Trace not found for execution {request.execution_id}",
            )

        mapping = json.loads(mapping_str)
        span_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        # Calculate tokens if provided
        prompt_tokens = 0
        completion_tokens = 0
        if request.token_usage:
            prompt_tokens = request.token_usage.get("promptTokens", 0)
            completion_tokens = request.token_usage.get("completionTokens", 0)

        span_data = {
            "trace_id": mapping["trace_id"],
            "span_id": span_id,
            "parent_span_id": mapping["span_id"],
            "project_id": request.project_id,
            "name": f"{request.provider}.{request.model}",
            "span_type": "llm",
            "started_at": now,
            "ended_at": now,
            "attributes": {
                "llm.vendor": request.provider,
                "llm.model": request.model,
                "llm.prompt_tokens": prompt_tokens,
                "llm.completion_tokens": completion_tokens,
                "llm.total_tokens": prompt_tokens + completion_tokens,
                "n8n.execution.id": request.execution_id,
                "service.name": "n8n",
                **(request.attributes or {}),
            },
            "events": [
                {
                    "name": "llm.request",
                    "timestamp": now,
                    "attributes": {"messages": request.prompt},
                },
                {
                    "name": "llm.response",
                    "timestamp": now,
                    "attributes": {"content": request.response},
                },
            ],
            "status": "completed",
        }

        if request.latency_ms:
            span_data["attributes"]["llm.latency_ms"] = request.latency_ms

        # Insert span to ClickHouse
        client = get_clickhouse_client()
        await insert_span(client, span_data)

        logger.debug(
            f"n8n AI call logged: {span_id} "
            f"({request.provider}/{request.model})"
        )

        return {"span_id": span_id, "trace_id": mapping["trace_id"]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to log n8n AI call: {e}")
        raise HTTPException(status_code=500, detail="Failed to log AI call")


@router.post("/webhook")
async def n8n_webhook_ingest(
    payload: N8nWebhookPayload,
    request: Request,
    project: str = Query(default="default", description="Project ID"),
    x_prela_project: str | None = Header(default=None, alias="X-Prela-Project"),
    api_key: str = Depends(verify_api_key),
):
    """Receive traces directly from n8n webhook nodes.

    Parses the n8n webhook format and creates appropriate spans.
    This is an alternative to the explicit start/end/span APIs.

    Project ID can be provided in 3 ways (priority order):
    1. Query parameter: ?project=my-project
    2. Header: X-Prela-Project: my-project
    3. Defaults to "default"

    Returns:
        Dictionary with spans_created count.
    """
    try:
        # Security: Check webhook rate limit (100 requests/minute per API key)
        rate_limiter = get_webhook_rate_limiter(limit_per_minute=100)
        await rate_limiter.check_rate_limit(api_key, endpoint="n8n_webhook")

        # Security: Check request body size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.max_webhook_payload_size:
            logger.warning(
                f"Webhook payload too large: {content_length} bytes "
                f"(max: {settings.max_webhook_payload_size})"
            )
            raise HTTPException(
                status_code=413,
                detail=f"Webhook payload too large. Maximum: {settings.max_webhook_payload_size} bytes"
            )

        # Extract project_id with priority: query param > header > default
        project_id = project if project != "default" else (x_prela_project or "default")

        # Validate project_id format
        if project_id != "default":
            project_id = InputValidator.validate_project_id(project_id)

        # Extract execution info
        execution_id = payload.execution.get("id")
        workflow_id = payload.workflow.get("id")
        workflow_name = payload.workflow.get("name", "Unknown Workflow")

        if not execution_id or not workflow_id:
            raise HTTPException(
                status_code=400,
                detail="Missing execution ID or workflow ID in webhook payload",
            )

        # Security: Validate execution_id and workflow_id formats
        execution_id = InputValidator.validate_execution_id(str(execution_id))
        workflow_id = InputValidator.validate_execution_id(str(workflow_id))

        # Security: Validate data array size if present
        if payload.data and isinstance(payload.data, list):
            if len(payload.data) > settings.max_webhook_data_items:
                logger.warning(
                    f"Webhook data array too large: {len(payload.data)} items "
                    f"(max: {settings.max_webhook_data_items})"
                )
                raise HTTPException(
                    status_code=413,
                    detail=f"Too many data items. Maximum: {settings.max_webhook_data_items}"
                )

        # Create trace_id
        trace_id = f"n8n-{execution_id}"
        root_span_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        spans_created = []

        # Create root workflow span
        workflow_span = {
            "trace_id": trace_id,
            "span_id": root_span_id,
            "project_id": project_id,
            "name": workflow_name,
            "span_type": "agent",
            "started_at": now,
            "ended_at": now,
            "attributes": {
                "n8n.workflow.id": workflow_id,
                "n8n.workflow.name": workflow_name,
                "n8n.execution.id": execution_id,
                "service.name": "n8n",
            },
            "status": "completed",
        }
        spans_created.append(workflow_span)

        # If node data is provided, create a node span
        if payload.node:
            node_span_id = str(uuid.uuid4())
            node_name = payload.node.get("name", "Unknown Node")
            node_type = payload.node.get("type", "custom")

            node_span = {
                "trace_id": trace_id,
                "span_id": node_span_id,
                "parent_span_id": root_span_id,
                "project_id": project_id,
                "name": node_name,
                "span_type": _map_n8n_node_type(node_type),
                "started_at": now,
                "ended_at": now,
                "attributes": {
                    "n8n.node.name": node_name,
                    "n8n.node.type": node_type,
                    "n8n.execution.id": execution_id,
                    "service.name": "n8n",
                },
                "status": "completed",
            }

            # Add node data as event if provided
            if payload.data:
                node_span["events"] = [
                    {
                        "name": "node.output",
                        "timestamp": now,
                        "attributes": {"data": payload.data},
                    }
                ]

            spans_created.append(node_span)

        # Insert all spans to ClickHouse
        client = get_clickhouse_client()
        for span in spans_created:
            await insert_span(client, span)

        logger.info(
            f"n8n webhook processed: {trace_id} "
            f"({len(spans_created)} spans created)"
        )

        return {
            "trace_id": trace_id,
            "spans_created": len(spans_created),
            "status": "accepted",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process n8n webhook: {e}")
        raise HTTPException(status_code=500, detail="Failed to process webhook")


# Helper functions


def _map_n8n_node_type(node_type: str) -> str:
    """Map n8n node type to Prela span type.

    Args:
        node_type: n8n node type string.

    Returns:
        Prela span type.
    """
    # Map common n8n AI nodes to appropriate span types
    ai_nodes = {
        "n8n-nodes-langchain.agent": "agent",
        "n8n-nodes-langchain.chainLlm": "llm",
        "n8n-nodes-langchain.chatOpenAi": "llm",
        "n8n-nodes-langchain.chatAnthropic": "llm",
        "n8n-nodes-langchain.openAi": "llm",
        "n8n-nodes-langchain.vectorStoreQdrant": "retrieval",
        "n8n-nodes-langchain.vectorStorePinecone": "retrieval",
        "n8n-nodes-langchain.memoryBufferWindow": "memory",
        "n8n-nodes-langchain.toolCalculator": "tool",
        "n8n-nodes-langchain.toolCode": "tool",
    }

    return ai_nodes.get(node_type, "custom")


# Analytics Endpoints


@router.get("/workflows", response_model=WorkflowListResponse)
async def list_workflows(
    project_id: str = Query(..., description="Project ID"),
    api_key: str = Depends(verify_api_key),
) -> WorkflowListResponse:
    """
    List all n8n workflows with 24h metrics.

    Returns workflow summary with execution counts, success rates,
    duration, AI usage, and costs.
    """
    client = get_clickhouse_client()

    # Calculate 24h window
    now = datetime.utcnow()
    yesterday = now - timedelta(days=1)

    query = """
    SELECT
        JSONExtractString(attributes, 'n8n.workflow.id') AS workflow_id,
        JSONExtractString(attributes, 'n8n.workflow.name') AS workflow_name,
        max(started_at) AS last_execution,
        count(*) AS execution_count_24h,
        countIf(status = 'success') * 100.0 / count(*) AS success_rate_24h,
        avg(duration_ms) AS avg_duration_ms,
        sum(JSONExtractUInt(attributes, 'n8n.total_ai_calls')) AS total_ai_calls_24h,
        sum(JSONExtractUInt(attributes, 'llm.total_tokens')) AS total_tokens_24h,
        sum(JSONExtractFloat(attributes, 'llm.cost_usd')) AS total_cost_24h
    FROM traces
    WHERE
        project_id = %(project_id)s
        AND JSONHas(attributes, 'n8n.workflow.id')
        AND started_at >= %(start_time)s
    GROUP BY workflow_id, workflow_name
    ORDER BY last_execution DESC
    """

    result = client.execute(
        query,
        {
            "project_id": project_id,
            "start_time": yesterday,
        },
    )

    workflows = [
        N8nWorkflow(
            workflow_id=row[0],
            workflow_name=row[1],
            last_execution=row[2].isoformat(),
            execution_count_24h=row[3],
            success_rate_24h=round(row[4], 1),
            avg_duration_ms=round(row[5], 1),
            total_ai_calls_24h=row[6] or 0,
            total_tokens_24h=row[7] or 0,
            total_cost_24h=round(row[8] or 0, 2),
        )
        for row in result
    ]

    return WorkflowListResponse(workflows=workflows)


@router.get("/workflows/{workflow_id}", response_model=WorkflowDetailResponse)
async def get_workflow_detail(
    workflow_id: str,
    project_id: str = Query(..., description="Project ID"),
    api_key: str = Depends(verify_api_key),
) -> WorkflowDetailResponse:
    """
    Get detailed information for a specific workflow.

    Returns workflow summary, recent executions, and AI node metrics.
    """
    client = get_clickhouse_client()

    # Calculate 24h window
    now = datetime.utcnow()
    yesterday = now - timedelta(days=1)

    # Get workflow summary
    summary_query = """
    SELECT
        JSONExtractString(attributes, 'n8n.workflow.id') AS workflow_id,
        JSONExtractString(attributes, 'n8n.workflow.name') AS workflow_name,
        max(started_at) AS last_execution,
        count(*) AS execution_count_24h,
        countIf(status = 'success') * 100.0 / count(*) AS success_rate_24h,
        avg(duration_ms) AS avg_duration_ms,
        sum(JSONExtractUInt(attributes, 'n8n.total_ai_calls')) AS total_ai_calls_24h,
        sum(JSONExtractUInt(attributes, 'llm.total_tokens')) AS total_tokens_24h,
        sum(JSONExtractFloat(attributes, 'llm.cost_usd')) AS total_cost_24h
    FROM traces
    WHERE
        project_id = %(project_id)s
        AND JSONExtractString(attributes, 'n8n.workflow.id') = %(workflow_id)s
        AND started_at >= %(start_time)s
    GROUP BY workflow_id, workflow_name
    """

    summary_result = client.execute(
        summary_query,
        {
            "project_id": project_id,
            "workflow_id": workflow_id,
            "start_time": yesterday,
        },
    )

    if not summary_result:
        raise HTTPException(status_code=404, detail="Workflow not found")

    row = summary_result[0]
    workflow = N8nWorkflow(
        workflow_id=row[0],
        workflow_name=row[1],
        last_execution=row[2].isoformat(),
        execution_count_24h=row[3],
        success_rate_24h=round(row[4], 1),
        avg_duration_ms=round(row[5], 1),
        total_ai_calls_24h=row[6] or 0,
        total_tokens_24h=row[7] or 0,
        total_cost_24h=round(row[8] or 0, 2),
    )

    # Get recent executions
    executions_query = """
    SELECT
        JSONExtractString(attributes, 'n8n.execution.id') AS execution_id,
        started_at,
        ended_at,
        status,
        duration_ms,
        JSONExtractUInt(attributes, 'llm.total_tokens') AS total_tokens,
        JSONExtractFloat(attributes, 'llm.cost_usd') AS cost_usd
    FROM traces
    WHERE
        project_id = %(project_id)s
        AND JSONExtractString(attributes, 'n8n.workflow.id') = %(workflow_id)s
    ORDER BY started_at DESC
    LIMIT 50
    """

    executions_result = client.execute(
        executions_query,
        {
            "project_id": project_id,
            "workflow_id": workflow_id,
        },
    )

    executions = [
        N8nExecution(
            execution_id=row[0],
            started_at=row[1].isoformat(),
            ended_at=row[2].isoformat() if row[2] else "",
            status=row[3],
            duration_ms=round(row[4], 1),
            total_tokens=row[5] or 0,
            cost_usd=round(row[6] or 0, 4),
        )
        for row in executions_result
    ]

    # Get AI node metrics
    nodes_query = """
    SELECT
        name AS node_name,
        JSONExtractString(attributes, 'llm.model') AS model,
        JSONExtractString(attributes, 'llm.vendor') AS vendor,
        count(*) AS call_count,
        sum(JSONExtractUInt(attributes, 'llm.prompt_tokens')) AS prompt_tokens,
        sum(JSONExtractUInt(attributes, 'llm.completion_tokens')) AS completion_tokens,
        avg(JSONExtractFloat(attributes, 'llm.latency_ms')) AS avg_latency_ms
    FROM spans
    WHERE
        project_id = %(project_id)s
        AND span_type = 'llm'
        AND JSONExtractString(attributes, 'n8n.workflow.id') = %(workflow_id)s
        AND started_at >= %(start_time)s
    GROUP BY node_name, model, vendor
    ORDER BY call_count DESC
    """

    nodes_result = client.execute(
        nodes_query,
        {
            "project_id": project_id,
            "workflow_id": workflow_id,
            "start_time": yesterday,
        },
    )

    ai_nodes = [
        N8nAINode(
            node_name=row[0],
            model=row[1] or "unknown",
            vendor=row[2] or "unknown",
            call_count=row[3],
            prompt_tokens=row[4] or 0,
            completion_tokens=row[5] or 0,
            avg_latency_ms=round(row[6] or 0, 1),
        )
        for row in nodes_result
    ]

    return WorkflowDetailResponse(
        workflow=workflow,
        executions=executions,
        ai_nodes=ai_nodes,
    )


@router.get("/workflows/{workflow_id}/executions", response_model=ExecutionListResponse)
async def list_executions(
    workflow_id: str,
    project_id: str = Query(..., description="Project ID"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of executions"),
    api_key: str = Depends(verify_api_key),
) -> ExecutionListResponse:
    """List recent executions for a workflow."""
    client = get_clickhouse_client()

    query = """
    SELECT
        JSONExtractString(attributes, 'n8n.execution.id') AS execution_id,
        started_at,
        ended_at,
        status,
        duration_ms,
        JSONExtractUInt(attributes, 'llm.total_tokens') AS total_tokens,
        JSONExtractFloat(attributes, 'llm.cost_usd') AS cost_usd
    FROM traces
    WHERE
        project_id = %(project_id)s
        AND JSONExtractString(attributes, 'n8n.workflow.id') = %(workflow_id)s
    ORDER BY started_at DESC
    LIMIT %(limit)s
    """

    result = client.execute(
        query,
        {
            "project_id": project_id,
            "workflow_id": workflow_id,
            "limit": limit,
        },
    )

    executions = [
        N8nExecution(
            execution_id=row[0],
            started_at=row[1].isoformat(),
            ended_at=row[2].isoformat() if row[2] else "",
            status=row[3],
            duration_ms=round(row[4], 1),
            total_tokens=row[5] or 0,
            cost_usd=round(row[6] or 0, 4),
        )
        for row in result
    ]

    return ExecutionListResponse(executions=executions)


@router.get("/workflows/{workflow_id}/ai-nodes", response_model=AINodeListResponse)
async def list_ai_nodes(
    workflow_id: str,
    project_id: str = Query(..., description="Project ID"),
    api_key: str = Depends(verify_api_key),
) -> AINodeListResponse:
    """
    Get AI node usage metrics for a workflow.

    Returns aggregated metrics for each AI node including model,
    token usage, and latency.
    """
    client = get_clickhouse_client()

    # Calculate 24h window
    now = datetime.utcnow()
    yesterday = now - timedelta(days=1)

    query = """
    SELECT
        name AS node_name,
        JSONExtractString(attributes, 'llm.model') AS model,
        JSONExtractString(attributes, 'llm.vendor') AS vendor,
        count(*) AS call_count,
        sum(JSONExtractUInt(attributes, 'llm.prompt_tokens')) AS prompt_tokens,
        sum(JSONExtractUInt(attributes, 'llm.completion_tokens')) AS completion_tokens,
        avg(JSONExtractFloat(attributes, 'llm.latency_ms')) AS avg_latency_ms
    FROM spans
    WHERE
        project_id = %(project_id)s
        AND span_type = 'llm'
        AND JSONExtractString(attributes, 'n8n.workflow.id') = %(workflow_id)s
        AND started_at >= %(start_time)s
    GROUP BY node_name, model, vendor
    ORDER BY call_count DESC
    """

    result = client.execute(
        query,
        {
            "project_id": project_id,
            "workflow_id": workflow_id,
            "start_time": yesterday,
        },
    )

    nodes = [
        N8nAINode(
            node_name=row[0],
            model=row[1] or "unknown",
            vendor=row[2] or "unknown",
            call_count=row[3],
            prompt_tokens=row[4] or 0,
            completion_tokens=row[5] or 0,
            avg_latency_ms=round(row[6] or 0, 1),
        )
        for row in result
    ]

    return AINodeListResponse(nodes=nodes)


@router.get(
    "/executions/{execution_id}/timeline", response_model=ExecutionTimelineResponse
)
async def get_execution_timeline(
    execution_id: str,
    project_id: str = Query(..., description="Project ID"),
    api_key: str = Depends(verify_api_key),
) -> ExecutionTimelineResponse:
    """
    Get execution timeline showing node execution order and timing.

    Returns timeline data for visualizing node execution sequence,
    including start times, durations, and AI node detection.
    """
    client = get_clickhouse_client()

    # Get trace-level data
    trace_query = """
    SELECT
        JSONExtractString(attributes, 'n8n.execution.id') AS execution_id,
        started_at AS trace_start,
        duration_ms AS total_duration
    FROM traces
    WHERE
        project_id = %(project_id)s
        AND JSONExtractString(attributes, 'n8n.execution.id') = %(execution_id)s
    LIMIT 1
    """

    trace_result = client.execute(
        trace_query,
        {
            "project_id": project_id,
            "execution_id": execution_id,
        },
    )

    if not trace_result:
        raise HTTPException(status_code=404, detail="Execution not found")

    trace_start = trace_result[0][1]
    total_duration = trace_result[0][2]

    # Get span-level node data
    spans_query = """
    SELECT
        span_id AS node_id,
        name AS node_name,
        span_type AS node_type,
        started_at,
        duration_ms,
        status,
        JSONExtractString(attributes, 'n8n.node.type') AS n8n_node_type
    FROM spans
    WHERE
        project_id = %(project_id)s
        AND JSONExtractString(attributes, 'n8n.execution.id') = %(execution_id)s
    ORDER BY started_at ASC
    """

    spans_result = client.execute(
        spans_query,
        {
            "project_id": project_id,
            "execution_id": execution_id,
        },
    )

    # Process spans into timeline nodes
    nodes = []
    for row in spans_result:
        node_id = row[0]
        node_name = row[1]
        node_type = row[2]
        started_at = row[3]
        duration_ms = row[4]
        status = row[5]
        n8n_node_type = row[6] or ""

        # Calculate start offset from trace start
        start_offset_ms = (started_at - trace_start).total_seconds() * 1000

        # Determine if this is an AI node
        is_ai_node = node_type in ("llm", "embedding") or any(
            ai_type in n8n_node_type.lower()
            for ai_type in ["openai", "anthropic", "langchain", "vector", "embedding"]
        )

        nodes.append(
            TimelineNode(
                node_id=node_id,
                node_name=node_name,
                node_type=node_type,
                start_offset_ms=round(start_offset_ms, 1),
                duration_ms=round(duration_ms, 1),
                status=status,
                is_ai_node=is_ai_node,
            )
        )

    return ExecutionTimelineResponse(
        execution_id=execution_id,
        total_duration_ms=round(total_duration, 1),
        nodes=nodes,
    )
