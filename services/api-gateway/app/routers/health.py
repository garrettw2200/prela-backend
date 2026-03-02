"""Health check endpoints."""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from shared.database import get_database_pool
from shared.clickhouse import get_clickhouse_client
from shared.redis import get_redis_client

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health_check():
    """Health check endpoint for Railway."""
    return {"status": "healthy"}


@router.get("/ready")
async def readiness_check():
    """Readiness check — verifies connectivity to all critical dependencies."""
    checks: dict[str, str] = {}
    all_ok = True

    # PostgreSQL
    try:
        pool = await get_database_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        checks["postgres"] = "ok"
    except Exception as e:
        logger.warning(f"Readiness: postgres check failed: {e}")
        checks["postgres"] = "error"
        all_ok = False

    # ClickHouse
    try:
        client = get_clickhouse_client()
        client.command("SELECT 1")
        checks["clickhouse"] = "ok"
    except Exception as e:
        logger.warning(f"Readiness: clickhouse check failed: {e}")
        checks["clickhouse"] = "error"
        all_ok = False

    # Redis
    try:
        redis = await get_redis_client()
        await redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        logger.warning(f"Readiness: redis check failed: {e}")
        checks["redis"] = "error"
        all_ok = False

    status_code = 200 if all_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if all_ok else "degraded", **checks},
    )
