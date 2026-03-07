"""Prompt management API endpoints.

Provides CRUD for prompt templates with version history, tagging,
and promotion to deployment stages (production, staging, canary).
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


class CreatePromptRequest(BaseModel):
    """Request to create a new prompt template."""

    name: str = Field(..., min_length=1, max_length=200)
    template: str = Field(..., min_length=1)
    model: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    change_note: str = ""


class UpdatePromptRequest(BaseModel):
    """Request to create a new version of a prompt (new version is auto-incremented)."""

    template: str = Field(..., min_length=1)
    model: str | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
    change_note: str = ""


class PromotePromptRequest(BaseModel):
    """Request to promote a prompt version to a stage."""

    version: int
    stage: str = Field(default="production", min_length=1, max_length=50)


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------

_SCHEMA_INITIALIZED = False


def _ensure_prompt_tables() -> None:
    """Create prompt tables in ClickHouse if they don't exist."""
    global _SCHEMA_INITIALIZED
    if _SCHEMA_INITIALIZED:
        return

    client = get_clickhouse_client()

    client.command("""
        CREATE TABLE IF NOT EXISTS prompt_templates (
            prompt_id String,
            project_id String,
            name String,
            template String,
            version UInt32,
            model Nullable(String),
            tags Array(String),
            metadata String DEFAULT '{}',
            change_note String DEFAULT '',
            promoted_stages Array(String) DEFAULT [],
            created_at DateTime64(3) DEFAULT now64(),
            updated_at DateTime64(3) DEFAULT now64()
        ) ENGINE = ReplacingMergeTree(updated_at)
        ORDER BY (project_id, name, version)
    """)

    _SCHEMA_INITIALIZED = True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/projects/{project_id}/prompts")
async def list_prompts(
    project_id: str,
    tag: str | None = Query(None),
    name: str | None = Query(None),
    user=Depends(require_tier("free")),
):
    """List prompt templates (latest versions)."""
    _ensure_prompt_tables()
    client = get_clickhouse_client()

    where_parts = ["project_id = %(project_id)s"]
    params: dict[str, Any] = {"project_id": project_id}

    if tag:
        where_parts.append("has(tags, %(tag)s)")
        params["tag"] = tag
    if name:
        where_parts.append("name LIKE %(name)s")
        params["name"] = f"%{name}%"

    where_clause = " AND ".join(where_parts)

    # Get latest version of each prompt
    result = client.query(f"""
        SELECT
            prompt_id, name, template, version, model,
            tags, metadata, change_note, promoted_stages,
            created_at
        FROM prompt_templates
        WHERE {where_clause}
        AND (name, version) IN (
            SELECT name, max(version)
            FROM prompt_templates
            WHERE project_id = %(project_id)s
            GROUP BY name
        )
        ORDER BY name
    """, parameters=params)

    prompts = []
    for row in result.result_rows:
        prompts.append({
            "prompt_id": row[0],
            "name": row[1],
            "template": row[2],
            "version": row[3],
            "model": row[4],
            "tags": row[5],
            "metadata": row[6],
            "change_note": row[7],
            "promoted_stages": row[8],
            "created_at": row[9].isoformat() if row[9] else None,
        })

    return {"prompts": prompts, "count": len(prompts)}


@router.get("/projects/{project_id}/prompts/{prompt_name}")
async def get_prompt(
    project_id: str,
    prompt_name: str,
    version: int | None = Query(None),
    stage: str | None = Query(None),
    user=Depends(require_tier("free")),
):
    """Get a specific prompt template."""
    _ensure_prompt_tables()
    client = get_clickhouse_client()

    params: dict[str, Any] = {
        "project_id": project_id,
        "name": prompt_name,
    }

    if stage:
        # Get version promoted to this stage
        query = """
            SELECT
                prompt_id, name, template, version, model,
                tags, metadata, change_note, promoted_stages,
                created_at
            FROM prompt_templates
            WHERE project_id = %(project_id)s
              AND name = %(name)s
              AND has(promoted_stages, %(stage)s)
            ORDER BY version DESC
            LIMIT 1
        """
        params["stage"] = stage
    elif version:
        query = """
            SELECT
                prompt_id, name, template, version, model,
                tags, metadata, change_note, promoted_stages,
                created_at
            FROM prompt_templates
            WHERE project_id = %(project_id)s
              AND name = %(name)s
              AND version = %(version)s
            LIMIT 1
        """
        params["version"] = version
    else:
        # Get latest version
        query = """
            SELECT
                prompt_id, name, template, version, model,
                tags, metadata, change_note, promoted_stages,
                created_at
            FROM prompt_templates
            WHERE project_id = %(project_id)s
              AND name = %(name)s
            ORDER BY version DESC
            LIMIT 1
        """

    result = client.query(query, parameters=params)

    if not result.result_rows:
        raise HTTPException(status_code=404, detail="Prompt not found")

    row = result.result_rows[0]
    return {
        "prompt_id": row[0],
        "name": row[1],
        "template": row[2],
        "version": row[3],
        "model": row[4],
        "tags": row[5],
        "metadata": row[6],
        "change_note": row[7],
        "promoted_stages": row[8],
        "created_at": row[9].isoformat() if row[9] else None,
    }


@router.get("/projects/{project_id}/prompts/{prompt_name}/history")
async def get_prompt_history(
    project_id: str,
    prompt_name: str,
    user=Depends(require_tier("free")),
):
    """Get all versions of a prompt template."""
    _ensure_prompt_tables()
    client = get_clickhouse_client()

    result = client.query("""
        SELECT
            prompt_id, name, template, version, model,
            tags, metadata, change_note, promoted_stages,
            created_at
        FROM prompt_templates
        WHERE project_id = %(project_id)s AND name = %(name)s
        ORDER BY version ASC
    """, parameters={"project_id": project_id, "name": prompt_name})

    versions = []
    for row in result.result_rows:
        versions.append({
            "prompt_id": row[0],
            "name": row[1],
            "template": row[2],
            "version": row[3],
            "model": row[4],
            "tags": row[5],
            "metadata": row[6],
            "change_note": row[7],
            "promoted_stages": row[8],
            "created_at": row[9].isoformat() if row[9] else None,
        })

    if not versions:
        raise HTTPException(status_code=404, detail="Prompt not found")

    return {"name": prompt_name, "versions": versions, "count": len(versions)}


@router.post("/projects/{project_id}/prompts", status_code=201)
async def create_prompt(
    project_id: str,
    request: CreatePromptRequest,
    user=Depends(require_tier("free")),
):
    """Create a new prompt template (version 1)."""
    _ensure_prompt_tables()
    client = get_clickhouse_client()

    # Check if prompt with this name already exists
    existing = client.query("""
        SELECT count() FROM prompt_templates
        WHERE project_id = %(project_id)s AND name = %(name)s
    """, parameters={"project_id": project_id, "name": request.name})

    if existing.result_rows and existing.result_rows[0][0] > 0:
        raise HTTPException(
            status_code=409,
            detail=f"Prompt '{request.name}' already exists. Use PUT to create a new version.",
        )

    prompt_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    import json
    client.insert("prompt_templates", [[
        prompt_id,
        project_id,
        request.name,
        request.template,
        1,  # version
        request.model,
        request.tags,
        json.dumps(request.metadata),
        request.change_note,
        [],  # promoted_stages
        now,
        now,
    ]], column_names=[
        "prompt_id", "project_id", "name", "template", "version",
        "model", "tags", "metadata", "change_note", "promoted_stages",
        "created_at", "updated_at",
    ])

    return {
        "prompt_id": prompt_id,
        "name": request.name,
        "version": 1,
        "created_at": now.isoformat(),
    }


@router.put("/projects/{project_id}/prompts/{prompt_name}")
async def create_prompt_version(
    project_id: str,
    prompt_name: str,
    request: UpdatePromptRequest,
    user=Depends(require_tier("free")),
):
    """Create a new version of an existing prompt template."""
    _ensure_prompt_tables()
    client = get_clickhouse_client()

    # Get current latest version
    result = client.query("""
        SELECT max(version) FROM prompt_templates
        WHERE project_id = %(project_id)s AND name = %(name)s
    """, parameters={"project_id": project_id, "name": prompt_name})

    if not result.result_rows or result.result_rows[0][0] is None or result.result_rows[0][0] == 0:
        raise HTTPException(status_code=404, detail="Prompt not found")

    new_version = result.result_rows[0][0] + 1
    prompt_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    # Get previous version for defaults
    prev = client.query("""
        SELECT tags, metadata, model FROM prompt_templates
        WHERE project_id = %(project_id)s AND name = %(name)s
        ORDER BY version DESC LIMIT 1
    """, parameters={"project_id": project_id, "name": prompt_name})

    prev_tags = prev.result_rows[0][0] if prev.result_rows else []
    prev_metadata = prev.result_rows[0][1] if prev.result_rows else "{}"
    prev_model = prev.result_rows[0][2] if prev.result_rows else None

    import json
    client.insert("prompt_templates", [[
        prompt_id,
        project_id,
        prompt_name,
        request.template,
        new_version,
        request.model if request.model is not None else prev_model,
        request.tags if request.tags is not None else prev_tags,
        json.dumps(request.metadata) if request.metadata is not None else prev_metadata,
        request.change_note,
        [],  # promoted_stages
        now,
        now,
    ]], column_names=[
        "prompt_id", "project_id", "name", "template", "version",
        "model", "tags", "metadata", "change_note", "promoted_stages",
        "created_at", "updated_at",
    ])

    return {
        "prompt_id": prompt_id,
        "name": prompt_name,
        "version": new_version,
        "created_at": now.isoformat(),
    }


@router.post("/projects/{project_id}/prompts/{prompt_name}/promote")
async def promote_prompt(
    project_id: str,
    prompt_name: str,
    request: PromotePromptRequest,
    user=Depends(require_tier("free")),
):
    """Promote a prompt version to a deployment stage."""
    _ensure_prompt_tables()
    client = get_clickhouse_client()

    # Verify version exists
    result = client.query("""
        SELECT prompt_id, promoted_stages FROM prompt_templates
        WHERE project_id = %(project_id)s
          AND name = %(name)s
          AND version = %(version)s
        LIMIT 1
    """, parameters={
        "project_id": project_id,
        "name": prompt_name,
        "version": request.version,
    })

    if not result.result_rows:
        raise HTTPException(status_code=404, detail="Prompt version not found")

    prompt_id = result.result_rows[0][0]
    current_stages = list(result.result_rows[0][1])

    # Remove this stage from any other version of this prompt
    all_versions = client.query("""
        SELECT prompt_id, version, promoted_stages FROM prompt_templates
        WHERE project_id = %(project_id)s AND name = %(name)s
    """, parameters={"project_id": project_id, "name": prompt_name})

    now = datetime.now(timezone.utc)
    for row in all_versions.result_rows:
        v_id, v_num, v_stages = row[0], row[1], list(row[2])
        if request.stage in v_stages:
            v_stages.remove(request.stage)
            client.command(f"""
                ALTER TABLE prompt_templates
                UPDATE promoted_stages = {v_stages}, updated_at = now64()
                WHERE prompt_id = '{v_id}'
            """)

    # Add stage to the target version
    if request.stage not in current_stages:
        current_stages.append(request.stage)

    client.command(f"""
        ALTER TABLE prompt_templates
        UPDATE promoted_stages = {current_stages}, updated_at = now64()
        WHERE prompt_id = '{prompt_id}'
    """)

    return {
        "name": prompt_name,
        "version": request.version,
        "stage": request.stage,
        "promoted_stages": current_stages,
    }


@router.delete("/projects/{project_id}/prompts/{prompt_name}")
async def delete_prompt(
    project_id: str,
    prompt_name: str,
    version: int | None = Query(None),
    user=Depends(require_tier("free")),
):
    """Delete a prompt or a specific version."""
    _ensure_prompt_tables()
    client = get_clickhouse_client()

    params: dict[str, Any] = {
        "project_id": project_id,
        "name": prompt_name,
    }

    if version is not None:
        params["version"] = version
        client.command("""
            ALTER TABLE prompt_templates
            DELETE WHERE project_id = %(project_id)s
              AND name = %(name)s
              AND version = %(version)s
        """, parameters=params)
    else:
        client.command("""
            ALTER TABLE prompt_templates
            DELETE WHERE project_id = %(project_id)s AND name = %(name)s
        """, parameters=params)

    return {"deleted": True, "name": prompt_name, "version": version}
