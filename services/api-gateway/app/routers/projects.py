"""Project management API endpoints."""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from shared import get_clickhouse_client

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class ProjectCreate(BaseModel):
    """Request model for creating a project."""

    name: str = Field(..., min_length=1, max_length=100, description="Project name")
    description: str = Field(
        default="", max_length=500, description="Project description"
    )
    project_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=50,
        description="Custom project ID (auto-generated from name if not provided)",
    )


class ProjectUpdate(BaseModel):
    """Request model for updating a project."""

    name: str | None = Field(
        default=None, min_length=1, max_length=100, description="Project name"
    )
    description: str | None = Field(
        default=None, max_length=500, description="Project description"
    )


class Project(BaseModel):
    """Project response model."""

    project_id: str
    name: str
    description: str
    webhook_url: str
    created_at: datetime
    updated_at: datetime


class ProjectSummary(BaseModel):
    """Project summary with stats."""

    project_id: str
    name: str
    description: str
    webhook_url: str
    created_at: datetime
    updated_at: datetime
    trace_count_24h: int
    workflow_count: int


# Helper functions
def generate_project_id(name: str) -> str:
    """Generate project ID from name.

    Args:
        name: Project name

    Returns:
        URL-safe project ID
    """
    import re

    # Convert to lowercase, replace spaces and special chars with hyphens
    project_id = re.sub(r"[^a-z0-9]+", "-", name.lower())
    # Remove leading/trailing hyphens
    project_id = project_id.strip("-")
    # Limit length
    return project_id[:50]


def generate_webhook_url(project_id: str, base_url: str = "https://api.prela.dev") -> str:
    """Generate webhook URL for project.

    Args:
        project_id: Project ID
        base_url: Base API URL

    Returns:
        Webhook URL with project parameter
    """
    return f"{base_url}/v1/n8n/webhook?project={project_id}"


# API endpoints
@router.get("/projects", response_model=list[ProjectSummary])
async def list_projects(
    limit: int = 100,
    offset: int = 0,
) -> list[ProjectSummary]:
    """List all projects with statistics.

    Args:
        limit: Maximum number of projects to return
        offset: Number of projects to skip

    Returns:
        List of projects with stats
    """
    try:
        client = get_clickhouse_client()

        # Query projects with stats
        query = """
            SELECT
                p.project_id,
                p.name,
                p.description,
                p.webhook_url,
                p.created_at,
                p.updated_at,
                countIf(t.started_at >= now64(6) - INTERVAL 24 HOUR) AS trace_count_24h,
                uniqExact(JSONExtractString(t.attributes, 'n8n.workflow.id')) AS workflow_count
            FROM projects p
            LEFT JOIN traces t ON p.project_id = t.project_id
            GROUP BY p.project_id, p.name, p.description, p.webhook_url, p.created_at, p.updated_at
            ORDER BY p.created_at DESC
            LIMIT %(limit)s OFFSET %(offset)s
        """

        result = client.query(query, parameters={"limit": limit, "offset": offset})

        projects = []
        for row in result.result_rows:
            projects.append(
                ProjectSummary(
                    project_id=row[0],
                    name=row[1],
                    description=row[2],
                    webhook_url=row[3],
                    created_at=row[4],
                    updated_at=row[5],
                    trace_count_24h=row[6],
                    workflow_count=row[7],
                )
            )

        return projects

    except Exception as e:
        logger.error(f"Error listing projects: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects", response_model=Project, status_code=201)
async def create_project(project: ProjectCreate) -> Project:
    """Create a new project.

    Args:
        project: Project creation data

    Returns:
        Created project

    Raises:
        HTTPException: If project ID already exists
    """
    try:
        client = get_clickhouse_client()

        # Generate project_id if not provided
        project_id = project.project_id or generate_project_id(project.name)

        # Check if project_id already exists
        check_query = """
            SELECT count()
            FROM projects
            WHERE project_id = %(project_id)s
        """
        result = client.query(check_query, parameters={"project_id": project_id})
        if result.result_rows[0][0] > 0:
            raise HTTPException(
                status_code=409,
                detail=f"Project with ID '{project_id}' already exists",
            )

        # Generate webhook URL
        webhook_url = generate_webhook_url(project_id)

        # Insert project
        now = datetime.utcnow()
        insert_query = """
            INSERT INTO projects (project_id, name, description, webhook_url, created_at, updated_at)
            VALUES
        """
        client.insert(
            "projects",
            [
                [
                    project_id,
                    project.name,
                    project.description,
                    webhook_url,
                    now,
                    now,
                ]
            ],
            column_names=[
                "project_id",
                "name",
                "description",
                "webhook_url",
                "created_at",
                "updated_at",
            ],
        )

        return Project(
            project_id=project_id,
            name=project.name,
            description=project.description,
            webhook_url=webhook_url,
            created_at=now,
            updated_at=now,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}", response_model=Project)
async def get_project(project_id: str) -> Project:
    """Get project details.

    Args:
        project_id: Project ID

    Returns:
        Project details

    Raises:
        HTTPException: If project not found
    """
    try:
        client = get_clickhouse_client()

        query = """
            SELECT project_id, name, description, webhook_url, created_at, updated_at
            FROM projects
            WHERE project_id = %(project_id)s
            LIMIT 1
        """

        result = client.query(query, parameters={"project_id": project_id})

        if not result.result_rows:
            raise HTTPException(status_code=404, detail="Project not found")

        row = result.result_rows[0]
        return Project(
            project_id=row[0],
            name=row[1],
            description=row[2],
            webhook_url=row[3],
            created_at=row[4],
            updated_at=row[5],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting project: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/projects/{project_id}", response_model=Project)
async def update_project(project_id: str, update: ProjectUpdate) -> Project:
    """Update project details.

    Args:
        project_id: Project ID
        update: Fields to update

    Returns:
        Updated project

    Raises:
        HTTPException: If project not found
    """
    try:
        client = get_clickhouse_client()

        # Check if project exists
        check_query = """
            SELECT count()
            FROM projects
            WHERE project_id = %(project_id)s
        """
        result = client.query(check_query, parameters={"project_id": project_id})
        if result.result_rows[0][0] == 0:
            raise HTTPException(status_code=404, detail="Project not found")

        # Build update fields
        update_fields = []
        params: dict[str, Any] = {"project_id": project_id}

        if update.name is not None:
            update_fields.append("name = %(name)s")
            params["name"] = update.name

        if update.description is not None:
            update_fields.append("description = %(description)s")
            params["description"] = update.description

        if not update_fields:
            # No fields to update, just return current project
            return await get_project(project_id)

        # Add updated_at
        update_fields.append("updated_at = %(updated_at)s")
        params["updated_at"] = datetime.utcnow()

        # Update project
        update_query = f"""
            ALTER TABLE projects
            UPDATE {', '.join(update_fields)}
            WHERE project_id = %(project_id)s
        """
        client.command(update_query, parameters=params)

        # Return updated project
        return await get_project(project_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating project: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/projects/{project_id}", status_code=204)
async def delete_project(project_id: str) -> None:
    """Delete a project.

    Args:
        project_id: Project ID

    Raises:
        HTTPException: If project not found
    """
    try:
        client = get_clickhouse_client()

        # Check if project exists
        check_query = """
            SELECT count()
            FROM projects
            WHERE project_id = %(project_id)s
        """
        result = client.query(check_query, parameters={"project_id": project_id})
        if result.result_rows[0][0] == 0:
            raise HTTPException(status_code=404, detail="Project not found")

        # Delete project
        delete_query = """
            ALTER TABLE projects
            DELETE WHERE project_id = %(project_id)s
        """
        client.command(delete_query, parameters={"project_id": project_id})

        # Note: Traces and spans remain (orphaned but with project_id for historical purposes)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting project: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/webhook-status")
async def get_webhook_status(project_id: str) -> dict[str, Any]:
    """Get webhook receiver status for project.

    Args:
        project_id: Project ID

    Returns:
        Webhook status including last event timestamp
    """
    try:
        client = get_clickhouse_client()

        # Get last trace received
        query = """
            SELECT
                max(started_at) AS last_event,
                count() AS event_count_1h
            FROM traces
            WHERE project_id = %(project_id)s
                AND started_at >= now64(6) - INTERVAL 1 HOUR
        """

        result = client.query(query, parameters={"project_id": project_id})

        if not result.result_rows:
            return {
                "project_id": project_id,
                "status": "inactive",
                "last_event": None,
                "event_count_1h": 0,
            }

        row = result.result_rows[0]
        last_event = row[0]
        event_count = row[1]

        # Determine status based on last event
        status = "active" if event_count > 0 else "inactive"

        return {
            "project_id": project_id,
            "status": status,
            "last_event": last_event.isoformat() if last_event else None,
            "event_count_1h": event_count,
        }

    except Exception as e:
        logger.error(f"Error getting webhook status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
