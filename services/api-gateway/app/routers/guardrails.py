"""Guardrails configuration and violation log API endpoints.

Provides management of guardrail configurations per project
and queryable violation history from guardrail enforcement.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..auth import require_tier
from shared import get_clickhouse_client

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class GuardrailConfigRequest(BaseModel):
    """Request to create or update a guardrail configuration."""

    name: str = Field(..., min_length=1, max_length=200)
    guard_type: str = Field(..., description="pii, injection, content_filter, max_tokens, custom")
    enabled: bool = True
    action: str = Field(default="block", description="block, redact, log")
    config: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class ReportViolationRequest(BaseModel):
    """Request to report a guardrail violation from the SDK."""

    guard_name: str
    phase: str = Field(..., description="input or output")
    action_taken: str
    message: str = ""
    trace_id: str | None = None
    span_id: str | None = None
    agent_name: str | None = None
    model: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------

_SCHEMA_INITIALIZED = False


def _ensure_guardrail_tables() -> None:
    """Create guardrail tables in ClickHouse if they don't exist."""
    global _SCHEMA_INITIALIZED
    if _SCHEMA_INITIALIZED:
        return

    client = get_clickhouse_client()

    client.command("""
        CREATE TABLE IF NOT EXISTS guardrail_configs (
            config_id String,
            project_id String,
            name String,
            guard_type String,
            enabled UInt8 DEFAULT 1,
            action String DEFAULT 'block',
            config String DEFAULT '{}',
            description String DEFAULT '',
            created_at DateTime64(3) DEFAULT now64(),
            updated_at DateTime64(3) DEFAULT now64()
        ) ENGINE = ReplacingMergeTree(updated_at)
        ORDER BY (project_id, config_id)
    """)

    client.command("""
        CREATE TABLE IF NOT EXISTS guardrail_violations (
            violation_id String,
            project_id String,
            guard_name String,
            phase String,
            action_taken String,
            message String DEFAULT '',
            trace_id Nullable(String),
            span_id Nullable(String),
            agent_name Nullable(String),
            model Nullable(String),
            details String DEFAULT '{}',
            created_at DateTime64(3) DEFAULT now64()
        ) ENGINE = MergeTree()
        PARTITION BY (project_id, toYYYYMM(created_at))
        ORDER BY (project_id, created_at)
        TTL toDateTime(created_at) + INTERVAL 90 DAY
    """)

    _SCHEMA_INITIALIZED = True


# ---------------------------------------------------------------------------
# Configuration endpoints
# ---------------------------------------------------------------------------


@router.get("/projects/{project_id}/configs")
async def list_guardrail_configs(
    project_id: str,
    user=Depends(require_tier("free")),
):
    """List all guardrail configurations for a project."""
    _ensure_guardrail_tables()
    client = get_clickhouse_client()

    result = client.query("""
        SELECT config_id, name, guard_type, enabled, action, config,
               description, created_at, updated_at
        FROM guardrail_configs
        WHERE project_id = %(project_id)s
        ORDER BY name
    """, parameters={"project_id": project_id})

    configs = []
    for row in result.result_rows:
        configs.append({
            "config_id": row[0],
            "name": row[1],
            "guard_type": row[2],
            "enabled": bool(row[3]),
            "action": row[4],
            "config": row[5],
            "description": row[6],
            "created_at": row[7].isoformat() if row[7] else None,
            "updated_at": row[8].isoformat() if row[8] else None,
        })

    return {"configs": configs, "count": len(configs)}


@router.post("/projects/{project_id}/configs", status_code=201)
async def create_guardrail_config(
    project_id: str,
    request: GuardrailConfigRequest,
    user=Depends(require_tier("free")),
):
    """Create a new guardrail configuration."""
    _ensure_guardrail_tables()
    client = get_clickhouse_client()

    config_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    import json
    client.insert("guardrail_configs", [[
        config_id,
        project_id,
        request.name,
        request.guard_type,
        1 if request.enabled else 0,
        request.action,
        json.dumps(request.config),
        request.description,
        now,
        now,
    ]], column_names=[
        "config_id", "project_id", "name", "guard_type", "enabled",
        "action", "config", "description", "created_at", "updated_at",
    ])

    return {
        "config_id": config_id,
        "name": request.name,
        "guard_type": request.guard_type,
        "created_at": now.isoformat(),
    }


@router.put("/projects/{project_id}/configs/{config_id}")
async def update_guardrail_config(
    project_id: str,
    config_id: str,
    request: GuardrailConfigRequest,
    user=Depends(require_tier("free")),
):
    """Update an existing guardrail configuration."""
    _ensure_guardrail_tables()
    client = get_clickhouse_client()

    now = datetime.now(timezone.utc)

    import json
    client.insert("guardrail_configs", [[
        config_id,
        project_id,
        request.name,
        request.guard_type,
        1 if request.enabled else 0,
        request.action,
        json.dumps(request.config),
        request.description,
        now,  # created_at (kept from original via ReplacingMergeTree)
        now,  # updated_at
    ]], column_names=[
        "config_id", "project_id", "name", "guard_type", "enabled",
        "action", "config", "description", "created_at", "updated_at",
    ])

    return {
        "config_id": config_id,
        "name": request.name,
        "updated_at": now.isoformat(),
    }


@router.delete("/projects/{project_id}/configs/{config_id}")
async def delete_guardrail_config(
    project_id: str,
    config_id: str,
    user=Depends(require_tier("free")),
):
    """Delete a guardrail configuration."""
    _ensure_guardrail_tables()
    client = get_clickhouse_client()

    client.command("""
        ALTER TABLE guardrail_configs
        DELETE WHERE project_id = %(project_id)s AND config_id = %(config_id)s
    """, parameters={"project_id": project_id, "config_id": config_id})

    return {"deleted": True, "config_id": config_id}


# ---------------------------------------------------------------------------
# Violation endpoints
# ---------------------------------------------------------------------------


@router.post("/projects/{project_id}/violations", status_code=201)
async def report_violation(
    project_id: str,
    request: ReportViolationRequest,
    user=Depends(require_tier("free")),
):
    """Report a guardrail violation (called by SDK or backend scanners)."""
    _ensure_guardrail_tables()
    client = get_clickhouse_client()

    violation_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    import json
    client.insert("guardrail_violations", [[
        violation_id,
        project_id,
        request.guard_name,
        request.phase,
        request.action_taken,
        request.message,
        request.trace_id,
        request.span_id,
        request.agent_name,
        request.model,
        json.dumps(request.details),
        now,
    ]], column_names=[
        "violation_id", "project_id", "guard_name", "phase",
        "action_taken", "message", "trace_id", "span_id",
        "agent_name", "model", "details", "created_at",
    ])

    return {"violation_id": violation_id, "created_at": now.isoformat()}


@router.get("/projects/{project_id}/violations")
async def list_violations(
    project_id: str,
    guard_name: str | None = Query(None),
    phase: str | None = Query(None),
    action_taken: str | None = Query(None),
    hours: int = Query(default=24, ge=1, le=720),
    limit: int = Query(default=100, ge=1, le=1000),
    user=Depends(require_tier("free")),
):
    """List guardrail violations with filters."""
    _ensure_guardrail_tables()
    client = get_clickhouse_client()

    where_parts = [
        "project_id = %(project_id)s",
        "created_at >= now() - INTERVAL %(hours)s HOUR",
    ]
    params: dict[str, Any] = {"project_id": project_id, "hours": hours}

    if guard_name:
        where_parts.append("guard_name = %(guard_name)s")
        params["guard_name"] = guard_name
    if phase:
        where_parts.append("phase = %(phase)s")
        params["phase"] = phase
    if action_taken:
        where_parts.append("action_taken = %(action_taken)s")
        params["action_taken"] = action_taken

    where_clause = " AND ".join(where_parts)

    result = client.query(f"""
        SELECT violation_id, guard_name, phase, action_taken, message,
               trace_id, span_id, agent_name, model, details, created_at
        FROM guardrail_violations
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT %(limit)s
    """, parameters={**params, "limit": limit})

    violations = []
    for row in result.result_rows:
        violations.append({
            "violation_id": row[0],
            "guard_name": row[1],
            "phase": row[2],
            "action_taken": row[3],
            "message": row[4],
            "trace_id": row[5],
            "span_id": row[6],
            "agent_name": row[7],
            "model": row[8],
            "details": row[9],
            "created_at": row[10].isoformat() if row[10] else None,
        })

    return {"violations": violations, "count": len(violations)}


@router.get("/projects/{project_id}/violations/summary")
async def violation_summary(
    project_id: str,
    hours: int = Query(default=24, ge=1, le=720),
    user=Depends(require_tier("free")),
):
    """Get a summary of guardrail violations."""
    _ensure_guardrail_tables()
    client = get_clickhouse_client()

    result = client.query("""
        SELECT
            guard_name,
            phase,
            action_taken,
            count() as violation_count
        FROM guardrail_violations
        WHERE project_id = %(project_id)s
          AND created_at >= now() - INTERVAL %(hours)s HOUR
        GROUP BY guard_name, phase, action_taken
        ORDER BY violation_count DESC
    """, parameters={"project_id": project_id, "hours": hours})

    summary = []
    total = 0
    for row in result.result_rows:
        count = row[3]
        total += count
        summary.append({
            "guard_name": row[0],
            "phase": row[1],
            "action_taken": row[2],
            "count": count,
        })

    return {"summary": summary, "total_violations": total, "hours": hours}
