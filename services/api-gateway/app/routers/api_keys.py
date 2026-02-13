"""API key management endpoints."""

import hashlib
import logging
import secrets
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from shared.database import (
    create_api_key,
    get_api_keys_by_user_id,
    delete_api_key,
)
from app.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/api-keys", tags=["api-keys"])


class CreateApiKeyRequest(BaseModel):
    """Request to create a new API key."""

    name: str = Field(..., min_length=1, max_length=100, description="Name for the API key")


class ApiKeyResponse(BaseModel):
    """API key response with full key (only returned once)."""

    id: str
    key: str
    key_prefix: str
    name: str
    created_at: str


class ApiKeyListItem(BaseModel):
    """API key list item (without full key)."""

    id: str
    key_prefix: str
    name: str
    last_used_at: str | None
    created_at: str


@router.get("")
async def list_api_keys(
    user: dict = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """List all API keys for the current user.

    Returns list of API keys with:
    - id: API key UUID
    - key_prefix: First 16 characters (e.g., "prela_sk_abc123...")
    - name: Key name/description
    - last_used_at: Timestamp of last use (null if never used)
    - created_at: Timestamp of creation
    """
    user_id = user["user_id"]
    api_keys = await get_api_keys_by_user_id(user_id)

    logger.info(f"Listing {len(api_keys)} API keys for user {user_id}")

    return api_keys


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_new_api_key(
    request: CreateApiKeyRequest,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Create a new API key for the current user.

    **IMPORTANT**: The full API key is only returned once. Store it securely.

    Returns:
    {
        "id": "uuid",
        "key": "prela_sk_...",  // Full key, only returned once
        "key_prefix": "prela_sk_abc123...",
        "name": "My API Key",
        "created_at": "2026-02-01T12:00:00Z"
    }
    """
    user_id = user["user_id"]

    # Generate random API key
    api_key = f"prela_sk_{secrets.token_urlsafe(32)}"

    # Hash the key for storage
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    # Get key prefix for display
    key_prefix = api_key[:16]

    # Store in database
    try:
        result = await create_api_key(
            user_id=user_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=request.name,
        )

        logger.info(f"Created API key for user {user_id}: {key_prefix}...")

        return {
            "id": result["id"],
            "key": api_key,  # Full key, only returned once
            "key_prefix": key_prefix,
            "name": result["name"],
            "created_at": result["created_at"].isoformat() if result["created_at"] else None,
        }
    except Exception as e:
        logger.error(f"Failed to create API key for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key",
        )


@router.delete("/{api_key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    api_key_id: str,
    user: dict = Depends(get_current_user),
):
    """Revoke (delete) an API key.

    Args:
        api_key_id: UUID of the API key to delete.

    Returns:
        204 No Content on success.

    Raises:
        404 if the API key doesn't exist or doesn't belong to the user.
    """
    user_id = user["user_id"]

    try:
        await delete_api_key(api_key_id, user_id)
        logger.info(f"Revoked API key {api_key_id} for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to revoke API key {api_key_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found or access denied",
        )
