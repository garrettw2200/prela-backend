"""Suite 4: AI Feature Usage Limits — E2E Tests

Tests the AIFeatureLimiter against real Redis, exercising tier gating,
usage counting, soft cap enforcement, and the /ai-usage endpoint.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import pytest_asyncio
import redis.asyncio as aioredis
from httpx import AsyncClient

from .conftest import auth_headers, make_trace, make_span

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ============================================================
# Helpers
# ============================================================

def _parse_iso(s: str) -> datetime:
    """Parse ISO 8601 timestamp to datetime (ClickHouse needs datetime objects)."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


async def _seed_trace_with_error_span(clickhouse_client, project_id: str) -> str:
    """Insert a trace with an error span into ClickHouse for debug/security tests."""
    trace_data = json.loads(
        (FIXTURES_DIR / "traces" / "error_trace.json").read_text()
    )
    trace_id = trace_data["trace_id"]
    now = datetime.now(timezone.utc)

    clickhouse_client.insert(
        "traces",
        [[
            trace_id, project_id, trace_data["service_name"],
            _parse_iso(trace_data["started_at"]),
            _parse_iso(trace_data["completed_at"]),
            trace_data["duration_ms"], trace_data["status"],
            trace_data["root_span_id"], trace_data["span_count"],
            json.dumps(trace_data.get("attributes", {})),
            "native", now,
        ]],
        column_names=[
            "trace_id", "project_id", "service_name",
            "started_at", "completed_at", "duration_ms", "status",
            "root_span_id", "span_count", "attributes", "source", "created_at",
        ],
    )

    for span in trace_data["spans"]:
        clickhouse_client.insert(
            "spans",
            [[
                span["span_id"], trace_id, project_id,
                span.get("parent_span_id", ""), span["name"], span["span_type"],
                span.get("service_name", "test"),
                _parse_iso(span["started_at"]),
                _parse_iso(span["ended_at"]),
                span["duration_ms"], span["status"],
                json.dumps(span.get("attributes", {})),
                json.dumps(span.get("events", [])),
                json.dumps(span.get("replay_snapshot", {})),
                "native", now,
            ]],
            column_names=[
                "span_id", "trace_id", "project_id",
                "parent_span_id", "name", "span_type",
                "service_name", "started_at", "ended_at", "duration_ms",
                "status", "attributes", "events", "replay_snapshot",
                "source", "created_at",
            ],
        )

    return trace_id


# ============================================================
# Tests
# ============================================================


class TestFreeUserBlocked:
    """Free-tier users should be blocked from all AI features."""

    @pytest.mark.asyncio
    async def test_free_user_blocked_from_ai_usage_endpoint(
        self, api_gateway_client: AsyncClient, free_user: dict,
    ):
        """Free user can call GET /ai-usage but gets empty usage."""
        resp = await api_gateway_client.get(
            "/api/v1/billing/ai-usage",
            headers=auth_headers(free_user["api_key"]),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "free"
        assert data["usage"] == {}

    @pytest.mark.asyncio
    async def test_free_user_blocked_from_security_scan(
        self, api_gateway_client: AsyncClient, free_user: dict,
        clickhouse_client,
    ):
        """Free user gets 403 when trying to use security scan."""
        trace_id = await _seed_trace_with_error_span(
            clickhouse_client, free_user["user_id"]
        )
        resp = await api_gateway_client.get(
            f"/api/v1/security/traces/{trace_id}/scan",
            params={"project_id": free_user["user_id"]},
            headers=auth_headers(free_user["api_key"]),
        )
        assert resp.status_code == 403
        assert "Pro" in resp.json()["detail"] or "pro" in resp.json()["detail"].lower()


class TestProUserWithinLimits:
    """Pro-tier users should have access to AI features within limits."""

    @pytest.mark.asyncio
    async def test_pro_user_security_scan_allowed(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """Pro user can use security scan and usage is incremented."""
        trace_id = await _seed_trace_with_error_span(
            clickhouse_client, pro_user["user_id"]
        )

        resp = await api_gateway_client.get(
            f"/api/v1/security/traces/{trace_id}/scan",
            params={"project_id": pro_user["user_id"]},
            headers=auth_headers(pro_user["api_key"]),
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_usage_counter_increments(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client, redis_client: aioredis.Redis,
    ):
        """Calling a scanned endpoint multiple times increments usage."""
        trace_id = await _seed_trace_with_error_span(
            clickhouse_client, pro_user["user_id"]
        )

        for _ in range(3):
            resp = await api_gateway_client.get(
                f"/api/v1/security/traces/{trace_id}/scan",
                params={"project_id": pro_user["user_id"]},
                headers=auth_headers(pro_user["api_key"]),
            )
            assert resp.status_code == 200

        # Check usage via API
        usage_resp = await api_gateway_client.get(
            "/api/v1/billing/ai-usage",
            headers=auth_headers(pro_user["api_key"]),
        )
        assert usage_resp.status_code == 200
        usage = usage_resp.json()["usage"]
        assert usage["security"]["used"] == 3


class TestSoftCap:
    """Soft cap at 5x base limit should return 429."""

    @pytest.mark.asyncio
    async def test_soft_cap_returns_429(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client, redis_client: aioredis.Redis,
    ):
        """When usage exceeds 5x limit, endpoint returns 429."""
        trace_id = await _seed_trace_with_error_span(
            clickhouse_client, pro_user["user_id"]
        )

        # Set Redis counter to just over 5x the security limit (10_000 * 5 = 50_000)
        period = datetime.now(timezone.utc).strftime("%Y-%m")
        key = f"ai_security_usage:{pro_user['user_id']}:{period}"
        await redis_client.set(key, "50001")

        resp = await api_gateway_client.get(
            f"/api/v1/security/traces/{trace_id}/scan",
            params={"project_id": pro_user["user_id"]},
            headers=auth_headers(pro_user["api_key"]),
        )
        assert resp.status_code == 429
        assert "enterprise" in resp.json()["detail"].lower() or "contact" in resp.json()["detail"].lower()


class TestEnterpriseUnlimited:
    """Enterprise-tier users should have unlimited AI access."""

    @pytest.mark.asyncio
    async def test_enterprise_unlimited_access(
        self, api_gateway_client: AsyncClient, enterprise_user: dict,
        clickhouse_client,
    ):
        """Enterprise user can use AI features without usage tracking."""
        trace_id = await _seed_trace_with_error_span(
            clickhouse_client, enterprise_user["user_id"]
        )

        resp = await api_gateway_client.get(
            f"/api/v1/security/traces/{trace_id}/scan",
            params={"project_id": enterprise_user["user_id"]},
            headers=auth_headers(enterprise_user["api_key"]),
        )
        assert resp.status_code == 200


class TestUsageResets:
    """Usage counters should reset at the start of each month."""

    @pytest.mark.asyncio
    async def test_usage_resets_monthly(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        redis_client: aioredis.Redis,
    ):
        """Old-period usage keys don't affect current period."""
        # Set usage for last month
        old_period = "2025-12"
        key = f"ai_security_usage:{pro_user['user_id']}:{old_period}"
        await redis_client.set(key, "9999")

        # Current period should be clean
        usage_resp = await api_gateway_client.get(
            "/api/v1/billing/ai-usage",
            headers=auth_headers(pro_user["api_key"]),
        )
        assert usage_resp.status_code == 200
        usage = usage_resp.json()["usage"]
        assert usage["security"]["used"] == 0


class TestGetAiUsage:
    """GET /api/v1/billing/ai-usage should return all 6 features."""

    @pytest.mark.asyncio
    async def test_returns_all_features(
        self, api_gateway_client: AsyncClient, pro_user: dict,
    ):
        """AI usage endpoint returns all 6 feature categories."""
        resp = await api_gateway_client.get(
            "/api/v1/billing/ai-usage",
            headers=auth_headers(pro_user["api_key"]),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "pro"
        usage = data["usage"]

        expected_features = {
            "hallucination", "drift", "nlp", "security", "debug", "eval_generation"
        }
        assert set(usage.keys()) == expected_features

        for feature in expected_features:
            assert "used" in usage[feature]
            assert "limit" in usage[feature]


class TestRedisFailureFailOpen:
    """When Redis fails, AI features should still work (fail open)."""

    @pytest.mark.asyncio
    async def test_redis_failure_allows_request(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client, redis_client: aioredis.Redis,
    ):
        """If Redis raises an error, the request is still allowed."""
        trace_id = await _seed_trace_with_error_span(
            clickhouse_client, pro_user["user_id"]
        )

        # Patch the limiter's Redis client to raise errors
        from app.middleware.ai_feature_limiter import AIFeatureLimiter
        import app.middleware.ai_feature_limiter as limiter_mod
        original_limiter = limiter_mod._ai_feature_limiter

        broken_redis = AsyncMock()
        broken_redis.get = AsyncMock(side_effect=Exception("Redis connection lost"))
        broken_redis.incrby = AsyncMock(side_effect=Exception("Redis connection lost"))
        broken_redis.expire = AsyncMock(side_effect=Exception("Redis connection lost"))

        limiter_mod._ai_feature_limiter = AIFeatureLimiter(redis_client=broken_redis)

        try:
            resp = await api_gateway_client.get(
                f"/api/v1/security/traces/{trace_id}/scan",
                params={"project_id": pro_user["user_id"]},
                headers=auth_headers(pro_user["api_key"]),
            )
            # Should still succeed (fail open)
            assert resp.status_code == 200
        finally:
            limiter_mod._ai_feature_limiter = original_limiter
