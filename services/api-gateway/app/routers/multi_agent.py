"""Multi-agent framework routes for Prela API Gateway.

Provides endpoints for querying multi-agent execution data from CrewAI, AutoGen,
LangGraph, and Swarm frameworks.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from shared import get_clickhouse_client

router = APIRouter()
logger = logging.getLogger(__name__)


# Response Models


class ExecutionSummary(BaseModel):
    """Summary of a multi-agent execution."""

    execution_id: str = Field(..., description="Unique execution ID")
    framework: str = Field(..., description="Framework name (crewai, autogen, langgraph, swarm)")
    trace_id: str = Field(..., description="Root trace ID")
    service_name: str = Field(..., description="Service name from trace")
    status: str = Field(..., description="Execution status (success, error)")
    started_at: datetime = Field(..., description="Execution start time")
    duration_ms: float = Field(..., description="Total execution duration in milliseconds")
    num_agents: int = Field(0, description="Number of agents involved")
    num_tasks: int = Field(0, description="Number of tasks executed")
    num_messages: int = Field(0, description="Number of messages exchanged")


class ExecutionDetail(BaseModel):
    """Detailed information about a multi-agent execution."""

    execution_id: str
    framework: str
    trace_id: str
    service_name: str
    status: str
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    agents: list[dict[str, Any]] = Field(default_factory=list, description="List of agents")
    tasks: list[dict[str, Any]] = Field(default_factory=list, description="List of tasks")
    messages: list[dict[str, Any]] = Field(default_factory=list, description="Message history")
    attributes: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AgentPerformance(BaseModel):
    """Performance metrics for an individual agent."""

    agent_id: str = Field(..., description="Agent ID")
    agent_name: str = Field(..., description="Agent name")
    framework: str = Field(..., description="Framework name")
    total_executions: int = Field(..., description="Number of times agent executed")
    avg_duration_ms: float = Field(..., description="Average execution duration")
    success_rate: float = Field(..., description="Success rate (0.0-1.0)")
    total_tokens: int = Field(0, description="Total tokens consumed")
    total_cost_usd: float = Field(0.0, description="Total cost in USD")


class CommunicationGraph(BaseModel):
    """Agent communication graph for visualization."""

    nodes: list[dict[str, Any]] = Field(
        default_factory=list, description="Nodes (agents) in the graph"
    )
    edges: list[dict[str, Any]] = Field(
        default_factory=list, description="Edges (communications) between agents"
    )


# Endpoints


@router.get("/executions", response_model=dict[str, Any])
async def list_executions(
    project_id: str = Query(..., description="Project ID to filter by"),
    framework: Optional[str] = Query(None, description="Filter by framework (crewai, autogen, etc.)"),
    status: Optional[str] = Query(None, description="Filter by status (success, error)"),
    since: Optional[datetime] = Query(None, description="Start time filter (ISO format)"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of results"),
) -> dict[str, Any]:
    """List multi-agent crew/swarm executions.

    Args:
        project_id: Project ID to filter traces by.
        framework: Optional framework filter.
        status: Optional status filter.
        since: Optional start time filter.
        limit: Maximum number of results.

    Returns:
        Dictionary with executions list and metadata.
    """
    try:
        if since is None:
            since = datetime.utcnow() - timedelta(days=7)

        client = get_clickhouse_client()

        # Build WHERE clauses
        where_clauses = ["JSONExtractString(attributes, 'framework') != ''"]
        params: dict[str, Any] = {"since": since.isoformat(), "limit": limit}

        if framework:
            where_clauses.append("JSONExtractString(attributes, 'framework') = %(framework)s")
            params["framework"] = framework

        if status:
            where_clauses.append("status = %(status)s")
            params["status"] = status

        where_sql = " AND ".join(where_clauses)

        # Query for multi-agent executions
        # We identify multi-agent traces by the presence of 'framework' attribute
        query = f"""
            SELECT
                trace_id,
                JSONExtractString(attributes, 'execution_id') as execution_id,
                JSONExtractString(attributes, 'framework') as framework,
                service_name,
                status,
                started_at,
                duration_ms,
                JSONExtractInt(attributes, 'num_agents') as num_agents,
                JSONExtractInt(attributes, 'num_tasks') as num_tasks,
                JSONExtractInt(attributes, 'num_messages') as num_messages
            FROM traces
            WHERE {where_sql}
              AND started_at >= %(since)s
            ORDER BY started_at DESC
            LIMIT %(limit)s
        """

        result = client.query(query, parameters=params)

        executions = []
        for row in result.result_rows:
            executions.append(
                {
                    "trace_id": row[0],
                    "execution_id": row[1] or row[0],  # Fallback to trace_id
                    "framework": row[2],
                    "service_name": row[3],
                    "status": row[4],
                    "started_at": row[5],
                    "duration_ms": row[6],
                    "num_agents": row[7] or 0,
                    "num_tasks": row[8] or 0,
                    "num_messages": row[9] or 0,
                }
            )

        return {
            "executions": executions,
            "count": len(executions),
            "filters": {
                "project_id": project_id,
                "framework": framework,
                "status": status,
                "since": since.isoformat(),
            },
        }

    except Exception as e:
        logger.error(f"Failed to list executions: {e}")
        raise HTTPException(status_code=500, detail="Failed to query executions")


@router.get("/executions/{execution_id}", response_model=dict[str, Any])
async def get_execution(execution_id: str) -> dict[str, Any]:
    """Get detailed info about a crew execution.

    Args:
        execution_id: Execution ID or trace ID to retrieve.

    Returns:
        Dictionary with execution details including agents, tasks, messages.
    """
    try:
        client = get_clickhouse_client()

        # Get trace metadata
        trace_query = """
            SELECT
                trace_id,
                service_name,
                status,
                started_at,
                ended_at,
                duration_ms,
                attributes
            FROM traces
            WHERE trace_id = %(execution_id)s
               OR JSONExtractString(attributes, 'execution_id') = %(execution_id)s
            LIMIT 1
        """
        trace_result = client.query(trace_query, parameters={"execution_id": execution_id})

        if not trace_result.result_rows:
            raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")

        trace_row = trace_result.result_rows[0]
        trace_id = trace_row[0]

        # Parse attributes JSON
        import json

        attributes = json.loads(trace_row[6]) if trace_row[6] else {}

        # Get all spans for this trace
        spans_query = """
            SELECT
                span_id,
                name,
                span_type,
                started_at,
                ended_at,
                duration_ms,
                status,
                attributes
            FROM spans
            WHERE trace_id = %(trace_id)s
            ORDER BY started_at ASC
        """
        spans_result = client.query(spans_query, parameters={"trace_id": trace_id})

        # Organize spans by type
        agents = []
        tasks = []
        messages = []

        for span_row in spans_result.result_rows:
            span_attrs = json.loads(span_row[7]) if span_row[7] else {}

            # Identify agent spans
            if "agent.id" in span_attrs or "agent.name" in span_attrs:
                agents.append(
                    {
                        "span_id": span_row[0],
                        "agent_id": span_attrs.get("agent.id", ""),
                        "agent_name": span_attrs.get("agent.name", span_row[1]),
                        "role": span_attrs.get("agent.role", ""),
                        "duration_ms": span_row[5],
                        "status": span_row[6],
                    }
                )

            # Identify task spans
            if "task.id" in span_attrs or "task.description" in span_attrs:
                tasks.append(
                    {
                        "span_id": span_row[0],
                        "task_id": span_attrs.get("task.id", ""),
                        "description": span_attrs.get("task.description", ""),
                        "expected_output": span_attrs.get("task.expected_output", ""),
                        "duration_ms": span_row[5],
                        "status": span_row[6],
                    }
                )

            # Identify message/communication spans
            if "message.content" in span_attrs or span_row[2] == "custom":
                messages.append(
                    {
                        "span_id": span_row[0],
                        "content": span_attrs.get("message.content", ""),
                        "sender": span_attrs.get("sender.name", ""),
                        "recipient": span_attrs.get("recipient.name", ""),
                        "timestamp": span_row[3],
                    }
                )

        return {
            "execution_id": execution_id,
            "trace_id": trace_id,
            "framework": attributes.get("framework", ""),
            "service_name": trace_row[1],
            "status": trace_row[2],
            "started_at": trace_row[3],
            "ended_at": trace_row[4],
            "duration_ms": trace_row[5],
            "agents": agents,
            "tasks": tasks,
            "messages": messages,
            "attributes": attributes,
            "span_count": len(spans_result.result_rows),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get execution {execution_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve execution")


@router.get("/executions/{execution_id}/graph", response_model=CommunicationGraph)
async def get_communication_graph(execution_id: str) -> CommunicationGraph:
    """Get the agent communication graph for visualization.

    Args:
        execution_id: Execution ID or trace ID.

    Returns:
        Graph with nodes (agents) and edges (communications).
    """
    try:
        client = get_clickhouse_client()

        # Get trace_id
        trace_query = """
            SELECT trace_id
            FROM traces
            WHERE trace_id = %(execution_id)s
               OR JSONExtractString(attributes, 'execution_id') = %(execution_id)s
            LIMIT 1
        """
        trace_result = client.query(trace_query, parameters={"execution_id": execution_id})

        if not trace_result.result_rows:
            raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")

        trace_id = trace_result.result_rows[0][0]

        # Get all spans with agent information
        spans_query = """
            SELECT attributes
            FROM spans
            WHERE trace_id = %(trace_id)s
            ORDER BY started_at ASC
        """
        spans_result = client.query(spans_query, parameters={"trace_id": trace_id})

        import json

        # Extract agents and communications
        agents_map = {}  # agent_id -> agent info
        edges = []

        for span_row in spans_result.result_rows:
            span_attrs = json.loads(span_row[0]) if span_row[0] else {}

            # Extract agent node
            agent_id = span_attrs.get("agent.id") or span_attrs.get("agent.name")
            if agent_id and agent_id not in agents_map:
                agents_map[agent_id] = {
                    "id": agent_id,
                    "name": span_attrs.get("agent.name", agent_id),
                    "role": span_attrs.get("agent.role", ""),
                    "type": "agent",
                }

            # Extract communication edges
            sender = span_attrs.get("sender.name") or span_attrs.get("sender")
            recipient = span_attrs.get("recipient.name") or span_attrs.get("recipient")

            if sender and recipient:
                edges.append(
                    {
                        "source": sender,
                        "target": recipient,
                        "message": span_attrs.get("message.content", "")[:100],  # Truncate
                    }
                )

        return CommunicationGraph(nodes=list(agents_map.values()), edges=edges)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get communication graph for {execution_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve graph")


@router.get("/executions/{execution_id}/tasks", response_model=dict[str, Any])
async def get_execution_tasks(
    execution_id: str,
    status: Optional[str] = Query(None, description="Filter by task status"),
) -> dict[str, Any]:
    """Get tasks from a crew execution.

    Args:
        execution_id: Execution ID or trace ID.
        status: Optional status filter.

    Returns:
        Dictionary with tasks list.
    """
    try:
        client = get_clickhouse_client()

        # Get trace_id
        trace_query = """
            SELECT trace_id
            FROM traces
            WHERE trace_id = %(execution_id)s
               OR JSONExtractString(attributes, 'execution_id') = %(execution_id)s
            LIMIT 1
        """
        trace_result = client.query(trace_query, parameters={"execution_id": execution_id})

        if not trace_result.result_rows:
            raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")

        trace_id = trace_result.result_rows[0][0]

        # Query task spans
        where_clause = "trace_id = %(trace_id)s"
        params = {"trace_id": trace_id}

        if status:
            where_clause += " AND status = %(status)s"
            params["status"] = status

        tasks_query = f"""
            SELECT
                span_id,
                name,
                started_at,
                ended_at,
                duration_ms,
                status,
                attributes
            FROM spans
            WHERE {where_clause}
              AND (
                  JSONExtractString(attributes, 'task.id') != ''
                  OR JSONExtractString(attributes, 'task.description') != ''
              )
            ORDER BY started_at ASC
        """
        tasks_result = client.query(tasks_query, parameters=params)

        import json

        tasks = []
        for task_row in tasks_result.result_rows:
            task_attrs = json.loads(task_row[6]) if task_row[6] else {}
            tasks.append(
                {
                    "span_id": task_row[0],
                    "name": task_row[1],
                    "task_id": task_attrs.get("task.id", ""),
                    "description": task_attrs.get("task.description", ""),
                    "expected_output": task_attrs.get("task.expected_output", ""),
                    "agent": task_attrs.get("agent.name", ""),
                    "started_at": task_row[2],
                    "ended_at": task_row[3],
                    "duration_ms": task_row[4],
                    "status": task_row[5],
                }
            )

        return {"tasks": tasks, "count": len(tasks)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tasks for execution {execution_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve tasks")


@router.get("/analytics/agent-performance", response_model=dict[str, Any])
async def get_agent_performance(
    project_id: str = Query(..., description="Project ID"),
    framework: Optional[str] = Query(None, description="Filter by framework"),
    since: Optional[datetime] = Query(None, description="Start time filter"),
) -> dict[str, Any]:
    """Get per-agent performance metrics.

    Args:
        project_id: Project ID to analyze.
        framework: Optional framework filter.
        since: Optional start time filter.

    Returns:
        Dictionary with agent performance metrics.
    """
    try:
        if since is None:
            since = datetime.utcnow() - timedelta(days=7)

        client = get_clickhouse_client()

        # Build WHERE clause
        where_clauses = ["started_at >= %(since)s"]
        params: dict[str, Any] = {"since": since.isoformat()}

        if framework:
            where_clauses.append("JSONExtractString(attributes, 'framework') = %(framework)s")
            params["framework"] = framework

        where_sql = " AND ".join(where_clauses)

        # Query agent performance from spans
        query = f"""
            SELECT
                JSONExtractString(attributes, 'agent.id') as agent_id,
                JSONExtractString(attributes, 'agent.name') as agent_name,
                JSONExtractString(attributes, 'framework') as framework,
                COUNT(*) as total_executions,
                AVG(duration_ms) as avg_duration_ms,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) / COUNT(*) as success_rate,
                SUM(JSONExtractInt(attributes, 'total_tokens')) as total_tokens
            FROM spans
            WHERE {where_sql}
              AND JSONExtractString(attributes, 'agent.id') != ''
            GROUP BY agent_id, agent_name, framework
            ORDER BY total_executions DESC
        """

        result = client.query(query, parameters=params)

        agents = []
        for row in result.result_rows:
            agents.append(
                {
                    "agent_id": row[0],
                    "agent_name": row[1],
                    "framework": row[2],
                    "total_executions": row[3],
                    "avg_duration_ms": round(row[4], 2) if row[4] else 0,
                    "success_rate": round(row[5], 3) if row[5] else 0,
                    "total_tokens": row[6] or 0,
                    "total_cost_usd": 0.0,  # TODO: Calculate based on model and tokens
                }
            )

        return {
            "agents": agents,
            "count": len(agents),
            "period": {"since": since.isoformat(), "until": datetime.utcnow().isoformat()},
        }

    except Exception as e:
        logger.error(f"Failed to get agent performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve agent performance")
