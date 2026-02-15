"""E2E test configuration and fixtures.

These tests run against real PostgreSQL, ClickHouse, and Redis instances
started via docker-compose.test.yml. Auth is handled via real API keys
(no Clerk mocking needed for API tests).

Usage:
    # Start test infra
    cd backend/tests/e2e && docker compose -f docker-compose.test.yml up -d

    # Run tests
    cd backend && pytest tests/e2e/ -v --timeout=60

    # Tear down
    cd backend/tests/e2e && docker compose -f docker-compose.test.yml down -v
"""

import hashlib
import importlib
import os
import secrets
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

# Resolve paths
_backend_dir = Path(__file__).resolve().parent.parent.parent
_services_dir = _backend_dir / "services"
_api_gateway_dir = _services_dir / "api-gateway"
_ingest_gateway_dir = _services_dir / "ingest-gateway"

# Add to Python path so imports like `from shared.xxx` and `from app.xxx` work.
# Only add api-gateway (not ingest-gateway) so `from app.xxx` resolves to the
# API gateway's app module. The ingest gateway is loaded via importlib to avoid
# collision (both have app/main.py).
for p in [str(_services_dir), str(_api_gateway_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Set test environment variables BEFORE importing app modules.
# Also clear any .env-sourced vars that aren't in the Settings model
# (e.g. NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY) to avoid pydantic extra_forbidden errors.
_test_env = {
    "DATABASE_URL": "postgresql://prela_test:testpass@localhost:5433/prela_test",
    "CLICKHOUSE_HOST": "localhost",
    "CLICKHOUSE_PORT": "8124",
    "CLICKHOUSE_USER": "default",
    "CLICKHOUSE_PASSWORD": "testpass",
    "CLICKHOUSE_DATABASE": "prela_test",
    "REDIS_URL": "redis://localhost:6380/0",
    "KAFKA_BOOTSTRAP_SERVERS": "localhost:19093",
    "ENVIRONMENT": "test",
    "CLERK_JWKS_URL": "https://test.clerk.accounts.dev/.well-known/jwks.json",
    "STRIPE_SECRET_KEY": "sk_test_fake",
    "STRIPE_WEBHOOK_SECRET": "whsec_test_fake",
    "STRIPE_LUNCH_MONEY_PRICE_ID": "price_test_lunch",
    "STRIPE_PRO_PRICE_ID": "price_test_pro",
    "DATA_SOURCE_ENCRYPTION_KEY": "",
}
os.environ.update(_test_env)

# Remove env vars that exist in .env but aren't defined in the Settings model
# (pydantic-settings with class Config will reject extra fields)
_extra_env_keys = [
    "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY",
    "STRIPE_PUBLISHABLE_KEY",
    "SMTP_USERNAME",
]
for k in _extra_env_keys:
    os.environ.pop(k, None)

# Prevent pydantic-settings from reading the local .env file which contains
# extra keys not defined in the Settings model (causes extra_forbidden errors).
# We patch the BaseSettings.__init_subclass__ won't help here because Settings
# is already defined. Instead, we load the config module source and patch
# the Settings.Config.env_file before Settings() is instantiated.
import importlib.util as _ilu
_config_spec = _ilu.spec_from_file_location(
    "shared.config",
    str(_services_dir / "shared" / "config.py"),
)
_config_mod = _ilu.module_from_spec(_config_spec)
# Patch env_file to /dev/null BEFORE executing the module
# We need to execute the module to get Settings class, but Settings()
# is called at module level. So instead, we read and exec with patched env.
# Simplest approach: just override the .env file path via env var or
# temporarily change directory.
#
# Actually the cleanest fix: pydantic-settings reads env_file relative to cwd.
# We can just set the cwd to the e2e test directory (no .env there) during import.
_original_cwd = os.getcwd()
os.chdir(Path(__file__).parent)  # tests/e2e/ has no .env file

import asyncpg
import pytest
import pytest_asyncio
import redis.asyncio as aioredis
from httpx import ASGITransport, AsyncClient

# Trigger the shared module import (which instantiates Settings) while cwd
# is pointed at tests/e2e/ — no .env file to cause extra_forbidden errors.
import shared.config  # noqa: F401

# Restore cwd
os.chdir(_original_cwd)


# ============================================================
# Pytest-asyncio configuration
# ============================================================

# Use session-scoped event loop for all async fixtures and tests
# so session-scoped fixtures (pg_pool, redis_client, etc.) share
# the same loop as function-scoped fixtures and tests.


# ============================================================
# Database fixtures
# ============================================================

@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def pg_pool() -> AsyncGenerator[asyncpg.Pool, None]:
    """Session-scoped PostgreSQL connection pool."""
    pool = await asyncpg.create_pool(
        os.environ["DATABASE_URL"],
        min_size=2,
        max_size=5,
        command_timeout=30,
    )
    yield pool
    await pool.close()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def redis_client() -> AsyncGenerator[aioredis.Redis, None]:
    """Session-scoped Redis client."""
    client = aioredis.from_url(
        os.environ["REDIS_URL"],
        encoding="utf-8",
        decode_responses=True,
    )
    yield client
    await client.flushdb()
    await client.aclose()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def clickhouse_client():
    """Session-scoped ClickHouse client with test schema initialized."""
    import clickhouse_connect

    client = clickhouse_connect.get_client(
        host="localhost",
        port=8124,
        username="default",
        password="testpass",
        database="default",
    )

    # Create test database
    client.command("CREATE DATABASE IF NOT EXISTS prela_test")

    # Reconnect to test database
    client = clickhouse_connect.get_client(
        host="localhost",
        port=8124,
        username="default",
        password="testpass",
        database="prela_test",
    )

    # Run schema init
    schema_file = Path(__file__).parent / "init_clickhouse.sql"
    schema_sql = schema_file.read_text()
    for statement in schema_sql.split(";"):
        statement = statement.strip()
        if statement:
            client.command(statement)

    yield client

    # Clean up tables after all tests
    for table in ["traces", "spans", "analysis_results", "agent_baselines",
                   "drift_alerts", "alert_rules"]:
        client.command(f"TRUNCATE TABLE IF EXISTS {table}")


# ============================================================
# Per-test cleanup
# ============================================================

@pytest_asyncio.fixture(autouse=True, loop_scope="session")
async def cleanup_redis(redis_client: aioredis.Redis):
    """Flush Redis before each test."""
    await redis_client.flushdb()
    yield


@pytest_asyncio.fixture(autouse=True, loop_scope="session")
async def cleanup_pg(pg_pool: asyncpg.Pool):
    """Clean user-created data after each test (preserve schema)."""
    yield
    async with pg_pool.acquire() as conn:
        await conn.execute("DELETE FROM data_sources")
        await conn.execute("DELETE FROM usage_overages")
        await conn.execute("DELETE FROM usage_records")
        await conn.execute("DELETE FROM api_keys")
        await conn.execute("DELETE FROM subscriptions")
        await conn.execute("DELETE FROM users")


@pytest_asyncio.fixture(autouse=True, loop_scope="session")
async def cleanup_clickhouse(clickhouse_client):
    """Truncate ClickHouse tables after each test."""
    yield
    for table in ["traces", "spans", "analysis_results", "agent_baselines",
                   "drift_alerts", "alert_rules"]:
        clickhouse_client.command(f"TRUNCATE TABLE IF EXISTS {table}")


# ============================================================
# User factory fixtures
# ============================================================

async def _create_test_user(
    pg_pool: asyncpg.Pool,
    tier: str,
    email_prefix: str,
) -> dict[str, Any]:
    """Create a test user with subscription and API key.

    Returns dict with: user_id, clerk_id, email, tier, api_key, api_key_id,
    subscription_id
    """
    clerk_id = f"clerk_test_{email_prefix}_{secrets.token_hex(8)}"
    email = f"{email_prefix}@test.prela.dev"
    api_key = f"prela_sk_test_{email_prefix}_{secrets.token_urlsafe(16)}"
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    key_prefix = api_key[:16]

    tier_limits = {
        "free": 50_000,
        "lunch-money": 100_000,
        "pro": 1_000_000,
        "enterprise": None,
    }
    trace_limit = tier_limits.get(tier, 50_000)

    async with pg_pool.acquire() as conn:
        user = await conn.fetchrow(
            """
            INSERT INTO users (clerk_id, email, full_name)
            VALUES ($1, $2, $3)
            RETURNING *
            """,
            clerk_id,
            email,
            f"Test {email_prefix.title()} User",
        )

        user_id = str(user["id"])

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        sub = await conn.fetchrow(
            """
            INSERT INTO subscriptions (
                user_id, tier, status, trace_limit,
                current_period_start, current_period_end
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING *
            """,
            user["id"],
            tier,
            "active",
            trace_limit or 999_999_999,
            now,
            now + timedelta(days=30),
        )

        ak = await conn.fetchrow(
            """
            INSERT INTO api_keys (user_id, key_hash, key_prefix, name)
            VALUES ($1, $2, $3, $4)
            RETURNING *
            """,
            user["id"],
            key_hash,
            key_prefix,
            f"{tier} test key",
        )

    return {
        "user_id": user_id,
        "clerk_id": clerk_id,
        "email": email,
        "tier": tier,
        "api_key": api_key,
        "api_key_id": str(ak["id"]),
        "subscription_id": str(sub["id"]),
    }


@pytest_asyncio.fixture(loop_scope="session")
async def free_user(pg_pool: asyncpg.Pool) -> dict[str, Any]:
    """Create a free-tier test user."""
    return await _create_test_user(pg_pool, "free", "free")


@pytest_asyncio.fixture(loop_scope="session")
async def lunch_money_user(pg_pool: asyncpg.Pool) -> dict[str, Any]:
    """Create a lunch-money-tier test user."""
    return await _create_test_user(pg_pool, "lunch-money", "lunchmoney")


@pytest_asyncio.fixture(loop_scope="session")
async def pro_user(pg_pool: asyncpg.Pool) -> dict[str, Any]:
    """Create a pro-tier test user."""
    return await _create_test_user(pg_pool, "pro", "pro")


@pytest_asyncio.fixture(loop_scope="session")
async def enterprise_user(pg_pool: asyncpg.Pool) -> dict[str, Any]:
    """Create an enterprise-tier test user."""
    return await _create_test_user(pg_pool, "enterprise", "enterprise")


# ============================================================
# Application fixtures
# ============================================================

@pytest_asyncio.fixture(loop_scope="session")
async def api_gateway_client(
    pg_pool: asyncpg.Pool,
    redis_client: aioredis.Redis,
    clickhouse_client,
) -> AsyncGenerator[AsyncClient, None]:
    """Create an httpx AsyncClient for the API Gateway.

    Patches database, ClickHouse, and Redis connections to use test instances.
    The api-gateway directory is already on sys.path, so `from app.xxx` works.
    """
    # Patch shared.database pool to use test PG
    import shared.database as db_mod
    db_mod._pool = pg_pool

    # Patch shared.clickhouse to use test client.
    # Must patch BOTH shared.clickhouse AND shared (the __init__ re-export)
    # because routers do `from shared import get_clickhouse_client`.
    import shared.clickhouse as ch_mod
    import shared as shared_mod
    original_get_ch = ch_mod.get_clickhouse_client
    test_get_ch = lambda: clickhouse_client
    ch_mod.get_clickhouse_client = test_get_ch
    shared_mod.get_clickhouse_client = test_get_ch

    # Patch AI feature limiter to use test Redis
    from app.middleware.ai_feature_limiter import AIFeatureLimiter
    import app.middleware.ai_feature_limiter as limiter_mod
    limiter_mod._ai_feature_limiter = AIFeatureLimiter(redis_client=redis_client)

    # Import app after patching
    from app.main import app

    # Also patch any already-imported router modules that captured the
    # old get_clickhouse_client reference via `from shared import ...`
    for mod_name, mod in sys.modules.items():
        if mod_name.startswith("app.routers.") and hasattr(mod, "get_clickhouse_client"):
            mod.get_clickhouse_client = test_get_ch

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    # Restore
    ch_mod.get_clickhouse_client = original_get_ch
    shared_mod.get_clickhouse_client = original_get_ch
    limiter_mod._ai_feature_limiter = None
    for mod_name, mod in sys.modules.items():
        if mod_name.startswith("app.routers.") and hasattr(mod, "get_clickhouse_client"):
            mod.get_clickhouse_client = original_get_ch


@pytest_asyncio.fixture(loop_scope="session")
async def ingest_client(
    pg_pool: asyncpg.Pool,
    redis_client: aioredis.Redis,
    clickhouse_client,
) -> AsyncGenerator[AsyncClient, None]:
    """Create an httpx AsyncClient for the Ingest Gateway.

    The ingest-gateway directory is already on sys.path,
    so `from app.main import app` resolves to the ingest gateway app.
    We avoid name collision by importing it under a different name.
    """
    # Patch shared.database pool
    import shared.database as db_mod
    db_mod._pool = pg_pool

    # Patch shared.clickhouse
    import shared.clickhouse as ch_mod
    original_get_ch = ch_mod.get_clickhouse_client
    ch_mod.get_clickhouse_client = lambda: clickhouse_client

    # Patch rate limiter Redis
    from shared.rate_limiter import RateLimiter
    import shared.rate_limiter as rl_mod
    rl_mod._rate_limiter = RateLimiter(redis_client=redis_client)

    # Import the ingest gateway app via importlib to avoid collision
    # with the api-gateway's app.main
    spec = importlib.util.spec_from_file_location(
        "ingest_app",
        str(_ingest_gateway_dir / "app" / "main.py"),
    )
    ingest_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ingest_mod)
    ingest_app = ingest_mod.app

    # The ingest gateway stores clickhouse_client as a module-level global
    # that's normally set in lifespan(). Since we're using TestClient (no
    # lifespan), we set it directly.
    ingest_mod.clickhouse_client = clickhouse_client

    transport = ASGITransport(app=ingest_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    ch_mod.get_clickhouse_client = original_get_ch
    rl_mod._rate_limiter = None


# ============================================================
# Auth helpers
# ============================================================

def auth_headers(api_key: str) -> dict[str, str]:
    """Build Authorization header for a test API key."""
    return {"Authorization": f"Bearer {api_key}"}


# ============================================================
# Sample data helpers
# ============================================================

def make_trace(
    trace_id: str = None,
    service_name: str = "test-service",
    status: str = "completed",
    duration_ms: float = 150.0,
    spans: list[dict] | None = None,
) -> dict[str, Any]:
    """Build a sample trace payload for the ingest gateway."""
    now = datetime.now(timezone.utc)
    return {
        "trace_id": trace_id or secrets.token_hex(16),
        "service_name": service_name,
        "started_at": (now - timedelta(milliseconds=duration_ms)).isoformat(),
        "completed_at": now.isoformat(),
        "duration_ms": duration_ms,
        "status": status,
        "root_span_id": "",
        "span_count": len(spans or []),
        "attributes": {},
        "spans": spans or [],
    }


def make_span(
    span_id: str = None,
    trace_id: str = "dummy",
    name: str = "llm-call",
    span_type: str = "llm",
    status: str = "completed",
    duration_ms: float = 100.0,
    attributes: dict | None = None,
) -> dict[str, Any]:
    """Build a sample span for inclusion in a trace."""
    now = datetime.now(timezone.utc)
    return {
        "span_id": span_id or secrets.token_hex(8),
        "trace_id": trace_id,
        "parent_span_id": "",
        "name": name,
        "span_type": span_type,
        "service_name": "test-service",
        "started_at": (now - timedelta(milliseconds=duration_ms)).isoformat(),
        "ended_at": now.isoformat(),
        "duration_ms": duration_ms,
        "status": status,
        "attributes": attributes or {
            "llm.model": "gpt-4o-mini",
            "llm.vendor": "openai",
            "llm.prompt_tokens": 50,
            "llm.completion_tokens": 100,
            "llm.total_tokens": 150,
            "llm.cost_usd": 0.0015,
        },
        "events": [],
        "replay_snapshot": {},
    }
