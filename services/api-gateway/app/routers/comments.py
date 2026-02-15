"""Trace comment endpoints for team collaboration."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.auth import get_current_user, ROLE_HIERARCHY
from shared.database import (
    create_trace_comment,
    get_trace_comments,
    update_trace_comment,
    delete_trace_comment,
    check_project_access,
    execute,
    fetch_one,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["comments"])


# --- Request models ---


class CreateCommentRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=5000, description="Comment text")
    parent_comment_id: str | None = Field(
        default=None, description="Parent comment ID for threaded replies"
    )


class UpdateCommentRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=5000, description="Updated comment text")


# --- Serialization ---


def _serialize_comment(comment: dict[str, Any]) -> dict[str, Any]:
    """Serialize comment record for API response."""
    return {
        "id": str(comment["id"]),
        "trace_id": comment["trace_id"],
        "project_id": comment["project_id"],
        "user_id": str(comment["user_id"]),
        "content": comment["content"],
        "parent_comment_id": str(comment["parent_comment_id"]) if comment.get("parent_comment_id") else None,
        "email": comment.get("email"),
        "full_name": comment.get("full_name"),
        "profile_image_url": comment.get("profile_image_url"),
        "created_at": comment["created_at"].isoformat() if comment.get("created_at") else None,
        "updated_at": comment["updated_at"].isoformat() if comment.get("updated_at") else None,
    }


# --- Endpoints ---


@router.post(
    "/traces/{trace_id}/comments",
    status_code=status.HTTP_201_CREATED,
)
async def create_comment(
    trace_id: str,
    request: CreateCommentRequest,
    project_id: str = Query(..., description="Project ID for the trace"),
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Create a comment on a trace. Requires project access via team membership."""
    user_id = str(user["user_id"])

    # Verify project access
    access = await check_project_access(user_id, project_id)
    if access is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this project",
        )

    # Members and above can comment
    if ROLE_HIERARCHY.index(access["role"]) < ROLE_HIERARCHY.index("member"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Commenting requires member role or higher",
        )

    try:
        comment = await create_trace_comment(
            trace_id=trace_id,
            project_id=project_id,
            user_id=user_id,
            content=request.content,
            parent_comment_id=request.parent_comment_id,
        )
        logger.info(f"Comment created on trace {trace_id} by user {user_id}")
        return _serialize_comment(comment)
    except Exception as e:
        logger.error(f"Failed to create comment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create comment",
        )


@router.get("/traces/{trace_id}/comments")
async def list_comments(
    trace_id: str,
    project_id: str = Query(..., description="Project ID for the trace"),
    user: dict = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """List all comments on a trace. Requires project access via team membership."""
    user_id = str(user["user_id"])

    # Verify project access
    access = await check_project_access(user_id, project_id)
    if access is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this project",
        )

    comments = await get_trace_comments(trace_id, project_id)
    return [_serialize_comment(c) for c in comments]


@router.put("/comments/{comment_id}")
async def edit_comment(
    comment_id: str,
    request: UpdateCommentRequest,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Edit a comment. Only the author can edit."""
    user_id = str(user["user_id"])

    result = await update_trace_comment(comment_id, user_id, request.content)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not found or you are not the author",
        )

    logger.info(f"Comment {comment_id} updated by user {user_id}")
    return _serialize_comment(result)


@router.delete("/comments/{comment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_comment(
    comment_id: str,
    user: dict = Depends(get_current_user),
):
    """Delete a comment. Author or team admin can delete."""
    user_id = str(user["user_id"])

    # First try to delete as author
    comment = await fetch_one(
        "SELECT * FROM trace_comments WHERE id = $1",
        comment_id,
    )

    if not comment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not found",
        )

    # Check if user is the author
    if str(comment["user_id"]) == user_id:
        await delete_trace_comment(comment_id, user_id)
        logger.info(f"Comment {comment_id} deleted by author {user_id}")
        return

    # Check if user is an admin of the project's team
    access = await check_project_access(user_id, comment["project_id"])
    if access and ROLE_HIERARCHY.index(access["role"]) >= ROLE_HIERARCHY.index("admin"):
        # Admin can delete any comment — bypass ownership check
        await execute("DELETE FROM trace_comments WHERE id = $1", comment_id)
        logger.info(f"Comment {comment_id} deleted by admin {user_id}")
        return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Only the author or a team admin can delete this comment",
    )
