"""API key management endpoints."""

import hashlib
import logging
import secrets
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from shared.database import (
    create_api_key,
    get_api_keys_by_user_id,
    delete_api_key,
    delete_api_key_by_team,
)
from app.auth import get_current_user, get_user_team_role

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/api-keys", tags=["api-keys"])


class CreateApiKeyRequest(BaseModel):
    """Request to create a new API key."""

    name: str = Field(..., min_length=1, max_length=100, description="Name for the API key")
    team_id: Optional[str] = Field(default=None, description="If set, creates a team-scoped key visible to all team members")


class ApiKeyResponse(BaseModel):
    """API key response with full key (only returned once)."""

    id: str
    key: str
    key_prefix: str
    name: str
    team_id: str | None
    scope: str
    created_at: str


class ApiKeyListItem(BaseModel):
    """API key list item (without full key)."""

    id: str
    key_prefix: str
    name: str
    last_used_at: str | None
    created_at: str
    team_id: str | None
    scope: str


@router.get("")
async def list_api_keys(
    user: dict = Depends(get_current_user),
    team_id: Optional[str] = Query(default=None, description="Return team keys for this team"),
) -> list[dict[str, Any]]:
    """List API keys for the current user or a team.

    Without team_id: returns personal keys (team_id IS NULL) for the current user.
    With team_id: returns all keys belonging to that team. Caller must be a team member.
    """
    user_id = user["user_id"]

    if team_id is not None:
        role = await get_user_team_role(str(user_id), team_id)
        if role is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not a member of this team",
            )

    api_keys = await get_api_keys_by_user_id(str(user_id), team_id=team_id)
    logger.info(f"Listing {len(api_keys)} API keys for user {user_id} (team_id={team_id})")

    return [
        {
            "id": str(key["id"]),
            "key_prefix": key["key_prefix"],
            "name": key["name"],
            "last_used_at": key["last_used_at"].isoformat() if key.get("last_used_at") else None,
            "created_at": key["created_at"].isoformat() if key.get("created_at") else None,
            "team_id": str(key["team_id"]) if key.get("team_id") else None,
            "scope": "team" if key.get("team_id") else "personal",
        }
        for key in api_keys
    ]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_new_api_key(
    request: CreateApiKeyRequest,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Create a new API key.

    Without team_id: creates a personal key for the current user.
    With team_id: creates a team-scoped key. Requires admin or owner role in the team.

    **IMPORTANT**: The full API key is only returned once. Store it securely.
    """
    user_id = user["user_id"]

    if request.team_id is not None:
        role = await get_user_team_role(str(user_id), request.team_id)
        if role is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not a member of this team",
            )
        if role not in ("admin", "owner"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only team admins and owners can create team API keys",
            )

    # Generate random API key
    api_key = f"prela_sk_{secrets.token_urlsafe(32)}"

    # Hash the key for storage
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    # Get key prefix for display
    key_prefix = api_key[:16]

    try:
        result = await create_api_key(
            user_id=str(user_id),
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=request.name,
            team_id=request.team_id,
        )

        logger.info(f"Created API key for user {user_id} (team_id={request.team_id}): {key_prefix}...")

        return {
            "id": str(result["id"]),
            "key": api_key,  # Full key, only returned once
            "key_prefix": key_prefix,
            "name": result["name"],
            "team_id": str(result["team_id"]) if result.get("team_id") else None,
            "scope": "team" if result.get("team_id") else "personal",
            "created_at": result["created_at"].isoformat() if result.get("created_at") else None,
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
    team_id: Optional[str] = Query(default=None, description="Set to revoke a team key"),
):
    """Revoke (delete) an API key.

    Without team_id: revokes a personal key owned by the current user.
    With team_id: revokes a team key. Requires admin or owner role in the team.
    """
    user_id = user["user_id"]

    if team_id is not None:
        role = await get_user_team_role(str(user_id), team_id)
        if role is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not a member of this team",
            )
        if role not in ("admin", "owner"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only team admins and owners can revoke team API keys",
            )
        deleted = await delete_api_key_by_team(api_key_id, team_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found in this team",
            )
        logger.info(f"Revoked team API key {api_key_id} by user {user_id} (team {team_id})")
    else:
        try:
            await delete_api_key(api_key_id, str(user_id))
            logger.info(f"Revoked personal API key {api_key_id} for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to revoke API key {api_key_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found or access denied",
            )
