"""Data source management endpoints for external integrations."""

import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.auth import get_current_user
from shared.database import (
    create_data_source,
    get_data_sources_by_user_id,
    get_data_source_by_id,
    delete_data_source,
)
from shared.encryption import encrypt_secret, decrypt_secret
from shared.langfuse_normalizer import test_langfuse_connection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/data-sources", tags=["data-sources"])


class LangfuseConfig(BaseModel):
    """Langfuse connection configuration."""

    host: str = Field(default="https://cloud.langfuse.com", description="Langfuse instance URL")
    public_key: str = Field(..., min_length=1, description="Langfuse public key")
    secret_key: str = Field(..., min_length=1, description="Langfuse secret key")


class CreateDataSourceRequest(BaseModel):
    """Request to create a new data source."""

    type: Literal["langfuse"] = "langfuse"
    name: str = Field(..., min_length=1, max_length=100, description="Display name")
    project_id: str = Field(..., min_length=1, description="Prela project ID")
    config: LangfuseConfig


class TestConnectionRequest(BaseModel):
    """Request to test a data source connection without saving."""

    type: Literal["langfuse"] = "langfuse"
    config: LangfuseConfig


def _serialize_data_source(row: dict[str, Any]) -> dict[str, Any]:
    """Serialize a data source row for API response.

    Masks the secret key and converts datetimes to ISO strings.
    """
    config = row.get("config", {})
    if isinstance(config, str):
        import json
        config = json.loads(config)

    # Build safe config (no secret key)
    safe_config = {
        "host": config.get("host", ""),
        "public_key": config.get("public_key", ""),
    }

    return {
        "id": str(row["id"]),
        "user_id": str(row["user_id"]),
        "project_id": row["project_id"],
        "type": row["type"],
        "name": row["name"],
        "config": safe_config,
        "status": row["status"],
        "error_message": row.get("error_message"),
        "last_sync_at": row["last_sync_at"].isoformat() if row.get("last_sync_at") else None,
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
    }


@router.post("/test-connection")
async def test_connection(
    request: TestConnectionRequest,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Test a data source connection without saving.

    Returns:
        {"success": true} or raises 400 with error detail.
    """
    try:
        await test_langfuse_connection(
            host=request.config.host,
            public_key=request.config.public_key,
            secret_key=request.config.secret_key,
        )
        return {"success": True, "message": "Connection successful"}
    except Exception as e:
        logger.warning(f"Connection test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Connection failed: {e}",
        )


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_new_data_source(
    request: CreateDataSourceRequest,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Create a new data source connection.

    Tests the connection first, then encrypts the secret key and stores it.
    """
    user_id = str(user["user_id"])

    # Validate connection before saving
    try:
        await test_langfuse_connection(
            host=request.config.host,
            public_key=request.config.public_key,
            secret_key=request.config.secret_key,
        )
    except Exception as e:
        logger.warning(f"Connection validation failed for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not connect to Langfuse: {e}",
        )

    # Encrypt secret key for storage
    encrypted_secret = encrypt_secret(request.config.secret_key)

    config = {
        "host": request.config.host,
        "public_key": request.config.public_key,
        "encrypted_secret_key": encrypted_secret,
        "last_synced_timestamp": None,
    }

    try:
        result = await create_data_source(
            user_id=user_id,
            project_id=request.project_id,
            source_type=request.type,
            name=request.name,
            config=config,
        )
        logger.info(f"Created data source '{request.name}' for user {user_id}")
        return _serialize_data_source(result)
    except Exception as e:
        logger.error(f"Failed to create data source for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create data source",
        )


@router.get("")
async def list_data_sources(
    user: dict = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """List all data sources for the current user."""
    user_id = str(user["user_id"])
    sources = await get_data_sources_by_user_id(user_id)
    return [_serialize_data_source(s) for s in sources]


@router.delete("/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_data_source(
    source_id: str,
    user: dict = Depends(get_current_user),
):
    """Delete a data source connection."""
    user_id = str(user["user_id"])

    source = await get_data_source_by_id(source_id, user_id)
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data source not found",
        )

    await delete_data_source(source_id, user_id)
    logger.info(f"Deleted data source {source_id} for user {user_id}")


@router.post("/{source_id}/sync")
async def trigger_sync(
    source_id: str,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Trigger a manual sync for a data source.

    Returns sync results (traces/spans imported).
    """
    user_id = str(user["user_id"])

    source = await get_data_source_by_id(source_id, user_id)
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data source not found",
        )

    if source["status"] == "paused":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Data source is paused",
        )

    # Import here to avoid circular imports
    from app.services.data_source_sync import sync_data_source

    try:
        result = await sync_data_source(source_id)
        return result
    except Exception as e:
        logger.error(f"Manual sync failed for {source_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sync failed: {e}",
        )
