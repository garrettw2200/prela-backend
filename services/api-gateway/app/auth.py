"""Clerk authentication middleware for API Gateway."""

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Tuple

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
import httpx

from shared import settings
from shared.database import (
    get_user_by_clerk_id,
    create_user,
    get_subscription_by_user_id,
    create_free_subscription,
    verify_api_key,
    update_api_key_last_used,
)

logger = logging.getLogger(__name__)

# HTTP Bearer security scheme
http_bearer = HTTPBearer(auto_error=False)

# Cache for Clerk JWKS with TTL (public keys for verifying JWTs)
_jwks_cache: Tuple[dict[str, Any], datetime] | None = None
JWKS_CACHE_TTL = timedelta(hours=1)  # Refresh JWKS every hour


async def get_clerk_jwks() -> dict[str, Any]:
    """Fetch Clerk JWKS (public keys) for JWT verification with TTL-based caching.

    JWKS cache refreshes every hour to support key rotation.

    Returns:
        JWKS dictionary.

    Raises:
        HTTPException: If JWKS fetch fails.
    """
    global _jwks_cache

    now = datetime.now(timezone.utc)

    # Check if cache exists and is still valid
    if _jwks_cache is not None:
        jwks, cached_at = _jwks_cache
        cache_age = now - cached_at

        if cache_age < JWKS_CACHE_TTL:
            logger.debug(f"Using cached JWKS (age: {cache_age})")
            return jwks
        else:
            logger.info(f"JWKS cache expired (age: {cache_age}), refreshing")

    # Fetch fresh JWKS
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(settings.clerk_jwks_url)
            response.raise_for_status()
            jwks = response.json()

            # Cache with timestamp
            _jwks_cache = (jwks, now)
            logger.info(f"Clerk JWKS fetched and cached (TTL: {JWKS_CACHE_TTL})")
            return jwks

    except httpx.TimeoutException:
        logger.error("Timeout fetching Clerk JWKS")
        # If we have stale cache, use it as fallback
        if _jwks_cache is not None:
            logger.warning("Using stale JWKS cache due to timeout")
            return _jwks_cache[0]
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service timeout",
        )
    except Exception as e:
        logger.error(f"Failed to fetch Clerk JWKS: {e}")
        # Fallback to stale cache if available
        if _jwks_cache is not None:
            logger.warning("Using stale JWKS cache due to fetch error")
            return _jwks_cache[0]
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service unavailable",
        )


async def verify_clerk_jwt(token: str) -> dict[str, Any]:
    """Verify Clerk JWT token.

    Args:
        token: JWT token from Authorization header.

    Returns:
        Decoded JWT claims.

    Raises:
        HTTPException: If token is invalid.
    """
    try:
        # Get JWKS
        jwks = await get_clerk_jwks()

        # Decode without verification first to get the header
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")

        if not kid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing kid",
            )

        # Find the matching key
        key = None
        for jwk_key in jwks.get("keys", []):
            if jwk_key.get("kid") == kid:
                key = jwk_key
                break

        if not key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: key not found",
            )

        # Verify and decode the token
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            options={"verify_aud": False},  # Clerk doesn't use aud claim
        )

        return claims

    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    except Exception as e:
        logger.error(f"Unexpected error during JWT verification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication error",
        )


async def get_current_user_from_clerk(
    credentials: HTTPAuthorizationCredentials = Security(http_bearer),
) -> dict[str, Any]:
    """Authenticate user via Clerk JWT and return user + subscription data.

    This is used for dashboard/frontend authentication.

    Args:
        credentials: HTTP Bearer credentials from Authorization header.

    Returns:
        Dictionary containing user_id, clerk_id, email, tier, subscription data.

    Raises:
        HTTPException: If authentication fails.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
        )

    # Verify JWT
    claims = await verify_clerk_jwt(credentials.credentials)
    clerk_id = claims.get("sub")

    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing subject",
        )

    # Get or create user
    user = await get_user_by_clerk_id(clerk_id)
    if not user:
        # Create user from Clerk claims
        email = claims.get("email", "")
        full_name = claims.get("name")
        profile_image_url = claims.get("picture")

        user = await create_user(
            clerk_id=clerk_id,
            email=email,
            full_name=full_name,
            profile_image_url=profile_image_url,
        )
        logger.info(f"Created new user from Clerk: {clerk_id}")

    # Get subscription
    subscription = await get_subscription_by_user_id(user["id"])
    if not subscription:
        # Create free tier subscription for new user
        subscription = await create_free_subscription(user["id"])
        logger.info(f"Created free tier subscription for user: {user['id']}")

    # Check subscription status
    if subscription["status"] not in ("active", "trialing"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Subscription {subscription['status']}. Please update your payment method.",
        )

    return {
        "user_id": user["id"],
        "clerk_id": clerk_id,
        "email": user["email"],
        "tier": subscription["tier"],
        "subscription_id": subscription["id"],
        "trace_limit": subscription["trace_limit"],
        "monthly_usage": subscription["monthly_usage"],
        "subscription_status": subscription["status"],
    }


async def get_current_user_from_api_key(
    credentials: HTTPAuthorizationCredentials = Security(http_bearer),
) -> dict[str, Any]:
    """Authenticate user via API key (for SDK usage).

    This is used for SDK authentication (HTTP exporter).

    Args:
        credentials: HTTP Bearer credentials from Authorization header.

    Returns:
        Dictionary containing user_id, email, tier, subscription data.

    Raises:
        HTTPException: If authentication fails.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
        )

    api_key = credentials.credentials

    # Hash the API key
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    # Verify API key and get user + subscription
    result = await verify_api_key(key_hash)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # Check subscription status
    if result["subscription_status"] not in ("active", "trialing"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Subscription {result['subscription_status']}. Please update your payment method.",
        )

    # Update last_used_at timestamp (async, don't await)
    await update_api_key_last_used(result["api_key_id"])

    return {
        "user_id": result["user_id"],
        "clerk_id": result["clerk_id"],
        "email": result["email"],
        "tier": result["tier"],
        "trace_limit": result["trace_limit"],
        "monthly_usage": result["monthly_usage"],
        "subscription_status": result["subscription_status"],
        "api_key_name": result["api_key_name"],
    }


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(http_bearer),
) -> dict[str, Any]:
    """Authenticate user via either Clerk JWT or API key.

    This function tries both authentication methods:
    1. Clerk JWT (for dashboard/frontend)
    2. API key (for SDK)

    Args:
        credentials: HTTP Bearer credentials from Authorization header.

    Returns:
        Dictionary containing user_id, email, tier, subscription data.

    Raises:
        HTTPException: If authentication fails.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
        )

    token = credentials.credentials

    # Try to determine if it's a JWT or API key
    # API keys start with "prela_sk_" or "sk_"
    if token.startswith("prela_sk_") or token.startswith("sk_"):
        return await get_current_user_from_api_key(credentials)
    else:
        # Assume it's a Clerk JWT
        return await get_current_user_from_clerk(credentials)


# Optional: Dependency for checking tier requirements
def require_tier(minimum_tier: str):
    """Dependency factory for requiring a minimum subscription tier.

    Usage:
        @app.get("/api/v1/replay")
        async def replay_trace(user: dict = Depends(require_tier("lunch-money"))):
            # Only lunch-money, pro, or enterprise users can access

    Args:
        minimum_tier: Minimum tier required (free, lunch-money, pro, enterprise).

    Returns:
        FastAPI dependency function.
    """
    tier_hierarchy = ["free", "lunch-money", "pro", "enterprise"]

    async def tier_dependency(user: dict = Depends(get_current_user)) -> dict:
        user_tier = user["tier"]

        if tier_hierarchy.index(user_tier) < tier_hierarchy.index(minimum_tier):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This feature requires {minimum_tier} tier or higher. "
                f"Current tier: {user_tier}. "
                f"Upgrade at https://prela.app/pricing",
            )

        return user

    return tier_dependency
