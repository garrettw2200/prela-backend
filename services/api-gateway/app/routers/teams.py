"""Team management endpoints."""

import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.auth import get_current_user, require_team_role
from shared.database import (
    create_team,
    get_teams_for_user,
    get_team_by_id,
    update_team,
    delete_team,
    get_team_members,
    add_team_member,
    update_team_member_role,
    remove_team_member,
    get_team_member_count,
    create_team_invitation,
    get_invitation_by_token,
    accept_invitation,
    get_pending_invitations,
    revoke_invitation,
    assign_project_to_team,
    remove_project_from_team,
    get_projects_for_team,
)
from shared.notifications import send_email_notification

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/teams", tags=["teams"])


# --- Request/Response models ---


class CreateTeamRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Team name")


class UpdateTeamRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="New team name")


class InviteMemberRequest(BaseModel):
    email: str = Field(..., description="Email address to invite")
    role: str = Field(
        default="member",
        description="Role to assign (admin, member, viewer)",
    )


class UpdateMemberRoleRequest(BaseModel):
    role: str = Field(..., description="New role (admin, member, viewer)")


class AssignProjectRequest(BaseModel):
    project_id: str = Field(..., description="ClickHouse project ID to assign")


# --- Serialization helpers ---


def _serialize_team(team: dict[str, Any], role: str | None = None) -> dict[str, Any]:
    """Serialize team record for API response."""
    result = {
        "id": str(team["id"]),
        "name": team["name"],
        "slug": team["slug"],
        "owner_id": str(team["owner_id"]),
        "created_at": team["created_at"].isoformat() if team.get("created_at") else None,
        "updated_at": team["updated_at"].isoformat() if team.get("updated_at") else None,
    }
    if role is not None:
        result["role"] = role
    elif "user_role" in team:
        result["role"] = team["user_role"]
    return result


def _serialize_member(member: dict[str, Any]) -> dict[str, Any]:
    """Serialize team member record for API response."""
    return {
        "id": str(member["id"]),
        "user_id": str(member["user_id"]),
        "role": member["role"],
        "email": member.get("email"),
        "full_name": member.get("full_name"),
        "profile_image_url": member.get("profile_image_url"),
        "joined_at": member["joined_at"].isoformat() if member.get("joined_at") else None,
    }


def _serialize_invitation(inv: dict[str, Any]) -> dict[str, Any]:
    """Serialize invitation record for API response."""
    return {
        "id": str(inv["id"]),
        "email": inv["email"],
        "role": inv["role"],
        "status": inv["status"],
        "invited_by_email": inv.get("invited_by_email"),
        "invited_by_name": inv.get("invited_by_name"),
        "created_at": inv["created_at"].isoformat() if inv.get("created_at") else None,
        "expires_at": inv["expires_at"].isoformat() if inv.get("expires_at") else None,
    }


def _generate_slug(name: str, user_id: str) -> str:
    """Generate a URL-safe slug from team name + user ID suffix."""
    import re

    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    suffix = str(user_id)[:8]
    return f"{base}-{suffix}"


# --- Team CRUD ---


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_new_team(
    request: CreateTeamRequest,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Create a new team. The authenticated user becomes the owner."""
    user_id = str(user["user_id"])
    slug = _generate_slug(request.name, user_id)

    try:
        team = await create_team(
            name=request.name,
            slug=slug,
            owner_id=user_id,
        )
        logger.info(f"Created team {team['id']} for user {user_id}")
        return _serialize_team(team, role="owner")
    except Exception as e:
        logger.error(f"Failed to create team: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create team",
        )


@router.get("")
async def list_teams(
    user: dict = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """List all teams the authenticated user belongs to."""
    user_id = str(user["user_id"])
    teams = await get_teams_for_user(user_id)
    return [_serialize_team(t) for t in teams]


@router.get("/{team_id}")
async def get_team(
    team_id: str,
    user: dict = Depends(require_team_role("viewer")),
) -> dict[str, Any]:
    """Get team details. Requires team membership."""
    team = await get_team_by_id(team_id)
    if not team:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Team not found",
        )
    return _serialize_team(team, role=user["team_role"])


@router.put("/{team_id}")
async def update_team_details(
    team_id: str,
    request: UpdateTeamRequest,
    user: dict = Depends(require_team_role("admin")),
) -> dict[str, Any]:
    """Update team name. Requires admin role or higher."""
    try:
        team = await update_team(team_id, request.name)
        logger.info(f"Updated team {team_id} by user {user['user_id']}")
        return _serialize_team(team, role=user["team_role"])
    except Exception as e:
        logger.error(f"Failed to update team {team_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update team",
        )


@router.delete("/{team_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_team_endpoint(
    team_id: str,
    user: dict = Depends(require_team_role("owner")),
):
    """Delete a team. Only the owner can delete."""
    try:
        await delete_team(team_id)
        logger.info(f"Deleted team {team_id} by user {user['user_id']}")
    except Exception as e:
        logger.error(f"Failed to delete team {team_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete team",
        )


# --- Team members ---


@router.get("/{team_id}/members")
async def list_team_members(
    team_id: str,
    user: dict = Depends(require_team_role("viewer")),
) -> list[dict[str, Any]]:
    """List all members of a team. Requires team membership."""
    members = await get_team_members(team_id)
    return [_serialize_member(m) for m in members]


@router.put("/{team_id}/members/{member_user_id}")
async def update_member_role(
    team_id: str,
    member_user_id: str,
    request: UpdateMemberRoleRequest,
    user: dict = Depends(require_team_role("admin")),
) -> dict[str, Any]:
    """Change a member's role. Requires admin role or higher.

    Cannot change the owner's role.
    """
    if request.role not in ("admin", "member", "viewer"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role. Must be admin, member, or viewer.",
        )

    # Check if target is the owner
    team = await get_team_by_id(team_id)
    if team and str(team["owner_id"]) == member_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot change the owner's role",
        )

    try:
        result = await update_team_member_role(team_id, member_user_id, request.role)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Member not found",
            )
        logger.info(
            f"Updated role for user {member_user_id} to {request.role} "
            f"in team {team_id} by {user['user_id']}"
        )
        return _serialize_member(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update member role: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update member role",
        )


@router.delete(
    "/{team_id}/members/{member_user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_member(
    team_id: str,
    member_user_id: str,
    user: dict = Depends(require_team_role("admin")),
):
    """Remove a member from a team. Requires admin role or higher.

    Cannot remove the owner.
    """
    # Check if target is the owner
    team = await get_team_by_id(team_id)
    if team and str(team["owner_id"]) == member_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot remove the team owner",
        )

    await remove_team_member(team_id, member_user_id)
    logger.info(
        f"Removed user {member_user_id} from team {team_id} by {user['user_id']}"
    )


# --- Invitations ---


@router.post("/{team_id}/invitations", status_code=status.HTTP_201_CREATED)
async def create_invitation(
    team_id: str,
    request: InviteMemberRequest,
    user: dict = Depends(require_team_role("admin")),
) -> dict[str, Any]:
    """Invite a user to the team by email. Requires admin role or higher.

    Generates a unique token and sends an invitation email.
    Pro tier: 5 members included, overages at $12/user/mo.
    """
    if request.role not in ("admin", "member", "viewer"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role. Must be admin, member, or viewer.",
        )

    # Check member count for Pro tier limits
    member_count = await get_team_member_count(team_id)
    pending = await get_pending_invitations(team_id)
    total = member_count + len(pending)

    if total >= 25:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Team has reached the maximum member limit (25). "
            "Contact support for Enterprise tier.",
        )

    # Generate invitation token and expiry
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)

    try:
        invitation = await create_team_invitation(
            team_id=team_id,
            email=request.email,
            role=request.role,
            invited_by=str(user["user_id"]),
            token=token,
            expires_at=expires_at,
        )

        # Send invitation email
        team = await get_team_by_id(team_id)
        team_name = team["name"] if team else "a team"
        invite_url = f"https://app.prela.dev/invite/{token}"

        await send_email_notification(
            to_addresses=[request.email],
            subject=f"You've been invited to join {team_name} on Prela",
            body_html=f"""
            <html>
                <body style="font-family: sans-serif; line-height: 1.6; color: #374151;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                        <h2 style="color: #111827;">Join {team_name} on Prela</h2>
                        <p>You've been invited to join <strong>{team_name}</strong> as a <strong>{request.role}</strong>.</p>
                        <p>
                            <a href="{invite_url}"
                               style="display: inline-block; padding: 12px 24px;
                                      background-color: #4f46e5; color: white;
                                      text-decoration: none; border-radius: 6px;">
                                Accept Invitation
                            </a>
                        </p>
                        <p style="font-size: 12px; color: #6b7280; margin-top: 20px;">
                            This invitation expires in 7 days.
                        </p>
                    </div>
                </body>
            </html>
            """,
            body_text=f"You've been invited to join {team_name} on Prela. "
            f"Accept here: {invite_url}",
        )

        logger.info(
            f"Invitation sent to {request.email} for team {team_id} "
            f"by {user['user_id']}"
        )
        return _serialize_invitation(invitation)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create invitation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create invitation",
        )


@router.get("/{team_id}/invitations")
async def list_invitations(
    team_id: str,
    user: dict = Depends(require_team_role("admin")),
) -> list[dict[str, Any]]:
    """List pending invitations for a team. Requires admin role or higher."""
    invitations = await get_pending_invitations(team_id)
    return [_serialize_invitation(inv) for inv in invitations]


@router.delete(
    "/{team_id}/invitations/{invitation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def revoke_team_invitation(
    team_id: str,
    invitation_id: str,
    user: dict = Depends(require_team_role("admin")),
):
    """Revoke a pending invitation. Requires admin role or higher."""
    await revoke_invitation(invitation_id)
    logger.info(
        f"Revoked invitation {invitation_id} in team {team_id} by {user['user_id']}"
    )


# --- Invitation acceptance (not team-scoped) ---


@router.post("/invitations/{token}/accept")
async def accept_team_invitation(
    token: str,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Accept a team invitation using the token from the email link."""
    invitation = await get_invitation_by_token(token)

    if not invitation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invitation not found or invalid token",
        )

    if invitation["status"] != "pending":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invitation has already been {invitation['status']}",
        )

    # Check expiry
    now = datetime.now(timezone.utc)
    expires_at = invitation["expires_at"]
    if hasattr(expires_at, "tzinfo") and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if now > expires_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invitation has expired",
        )

    user_id = str(user["user_id"])
    team_id = str(invitation["team_id"])

    try:
        # Mark invitation as accepted
        await accept_invitation(str(invitation["id"]))

        # Add user to team
        await add_team_member(
            team_id=team_id,
            user_id=user_id,
            role=invitation["role"],
            invited_by=str(invitation["invited_by"]),
        )

        logger.info(
            f"User {user_id} accepted invitation to team {team_id} "
            f"as {invitation['role']}"
        )

        return {
            "team_id": team_id,
            "team_name": invitation.get("team_name", ""),
            "role": invitation["role"],
        }

    except Exception as e:
        logger.error(f"Failed to accept invitation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to accept invitation",
        )


# --- Project assignments ---


@router.post(
    "/{team_id}/projects",
    status_code=status.HTTP_201_CREATED,
)
async def assign_project(
    team_id: str,
    request: AssignProjectRequest,
    user: dict = Depends(require_team_role("admin")),
) -> dict[str, Any]:
    """Assign a project to a team. Requires admin role or higher."""
    try:
        result = await assign_project_to_team(request.project_id, team_id)
        logger.info(
            f"Assigned project {request.project_id} to team {team_id} "
            f"by {user['user_id']}"
        )
        return {
            "id": str(result["id"]),
            "project_id": result["project_id"],
            "team_id": str(result["team_id"]),
            "created_at": result["created_at"].isoformat() if result.get("created_at") else None,
        }
    except Exception as e:
        if "unique" in str(e).lower() or "duplicate" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Project is already assigned to this team",
            )
        logger.error(f"Failed to assign project: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to assign project",
        )


@router.get("/{team_id}/projects")
async def list_team_projects(
    team_id: str,
    user: dict = Depends(require_team_role("viewer")),
) -> list[dict[str, Any]]:
    """List all projects assigned to a team. Requires team membership."""
    projects = await get_projects_for_team(team_id)
    return [
        {
            "id": str(p["id"]),
            "project_id": p["project_id"],
            "team_id": str(p["team_id"]),
            "created_at": p["created_at"].isoformat() if p.get("created_at") else None,
        }
        for p in projects
    ]


@router.delete(
    "/{team_id}/projects/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_project(
    team_id: str,
    project_id: str,
    user: dict = Depends(require_team_role("admin")),
):
    """Remove a project from a team. Requires admin role or higher."""
    await remove_project_from_team(project_id, team_id)
    logger.info(
        f"Removed project {project_id} from team {team_id} by {user['user_id']}"
    )
