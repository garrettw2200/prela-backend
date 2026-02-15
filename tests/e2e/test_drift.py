"""Suite 7: Drift Detection — E2E Tests

Tests baseline calculation, drift checking, alert CRUD, and alert rule
management against real ClickHouse. BaselineCalculator and AnomalyDetector
are purely statistical (no LLM), so no mocking is needed — we seed real
span data and baselines.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from httpx import AsyncClient

from .conftest import auth_headers


# ============================================================
# Helpers
# ============================================================


def _seed_agent_spans(
    clickhouse_client,
    project_id: str,
    agent_name: str = "test-agent",
    service_name: str = "test-service",
    count: int = 20,
    duration_base: float = 100.0,
    status: str = "completed",
):
    """Insert agent spans into ClickHouse for baseline calculation."""
    now = datetime.now(timezone.utc)

    for i in range(count):
        span_id = f"agent_span_{uuid.uuid4().hex[:8]}"
        trace_id = f"drift_trace_{uuid.uuid4().hex[:8]}"
        started = now - timedelta(days=3, hours=i)
        duration = duration_base + (i * 5)  # slight variation

        attrs = json.dumps({
            "agent.name": agent_name,
            "llm.total_tokens": 150 + i,
            "llm.cost_usd": 0.001 * (150 + i),
            "llm.response": "test response",
        })

        # Insert trace
        clickhouse_client.insert(
            "traces",
            [[
                trace_id, project_id, service_name,
                started, started + timedelta(milliseconds=duration),
                duration, status, span_id, 1,
                json.dumps({}), "native", now,
            ]],
            column_names=[
                "trace_id", "project_id", "service_name",
                "started_at", "completed_at", "duration_ms", "status",
                "root_span_id", "span_count", "attributes", "source", "created_at",
            ],
        )

        # Insert agent span
        clickhouse_client.insert(
            "spans",
            [[
                span_id, trace_id, project_id,
                "", f"{agent_name}-task", "agent",
                service_name,
                started, started + timedelta(milliseconds=duration),
                duration, status,
                attrs,
                json.dumps([]),
                json.dumps({}),
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


def _insert_baseline(
    clickhouse_client,
    project_id: str,
    agent_name: str = "test-agent",
    service_name: str = "test-service",
    duration_mean: float = 150.0,
    duration_stddev: float = 25.0,
) -> str:
    """Insert a baseline directly into agent_baselines."""
    baseline_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    clickhouse_client.insert(
        "agent_baselines",
        [[
            baseline_id, project_id, agent_name, service_name,
            now - timedelta(days=7), now,
            50,  # sample_size
            duration_mean, duration_stddev,
            duration_mean * 0.95,  # p50
            duration_mean * 1.5,   # p95
            duration_mean * 2.0,   # p99
            duration_mean * 0.5,   # min
            duration_mean * 3.0,   # max
            150.0, 20.0,   # token_usage_mean, stddev
            50.0, 10.0,    # token_usage_p50, p95
            1.0, 0.5,      # tool_calls_mean, stddev
            200.0, 50.0,   # response_length_mean, stddev
            0.95, 2,        # success_rate, error_count
            0.015, 0.75,   # cost_mean, cost_total
            now, now,      # created_at, updated_at
        ]],
        column_names=[
            "baseline_id", "project_id", "agent_name", "service_name",
            "window_start", "window_end",
            "sample_size",
            "duration_mean", "duration_stddev",
            "duration_p50", "duration_p95", "duration_p99",
            "duration_min", "duration_max",
            "token_usage_mean", "token_usage_stddev",
            "token_usage_p50", "token_usage_p95",
            "tool_calls_mean", "tool_calls_stddev",
            "response_length_mean", "response_length_stddev",
            "success_rate", "error_count",
            "cost_mean", "cost_total",
            "created_at", "updated_at",
        ],
    )

    return baseline_id


def _make_alert_payload(baseline_id: str) -> dict:
    """Create a DriftAlert request payload."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "agent_name": "test-agent",
        "service_name": "test-service",
        "anomalies": [
            {
                "metric_name": "duration_mean",
                "current_value": 500.0,
                "baseline_mean": 150.0,
                "change_percent": 233.3,
                "severity": "high",
                "direction": "increased",
                "unit": "ms",
                "sample_size": 10,
            }
        ],
        "root_causes": [
            {
                "type": "model_change",
                "description": "Model response time increased",
                "confidence": 0.8,
            }
        ],
        "baseline": {
            "baseline_id": baseline_id,
            "agent_name": "test-agent",
            "service_name": "test-service",
            "window_start": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
            "window_end": datetime.now(timezone.utc).isoformat(),
            "sample_size": 50,
            "duration_mean": 150.0,
            "duration_stddev": 25.0,
            "duration_p50": 142.0,
            "duration_p95": 225.0,
            "token_usage_mean": 150.0,
            "token_usage_stddev": 20.0,
            "success_rate": 0.95,
            "error_count": 2,
            "cost_mean": 0.015,
            "cost_total": 0.75,
        },
        "detected_at": now,
    }


# ============================================================
# Tests
# ============================================================


class TestBaselines:
    """Test baseline calculation and listing."""

    @pytest.mark.asyncio
    async def test_list_baselines(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """Listing baselines returns inserted baselines."""
        _insert_baseline(clickhouse_client, pro_user["user_id"])

        resp = await api_gateway_client.get(
            f"/api/v1/drift/projects/{pro_user['user_id']}/baselines",
            headers=auth_headers(pro_user["api_key"]),
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1
        baseline = data["baselines"][0]
        assert baseline["agent_name"] == "test-agent"
        assert baseline["service_name"] == "test-service"
        assert baseline["duration_mean"] > 0
        assert baseline["sample_size"] > 0

    @pytest.mark.asyncio
    async def test_list_baselines_filter_by_agent(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """Listing baselines can filter by agent_name."""
        _insert_baseline(clickhouse_client, pro_user["user_id"], agent_name="agent-a")
        _insert_baseline(clickhouse_client, pro_user["user_id"], agent_name="agent-b")

        resp = await api_gateway_client.get(
            f"/api/v1/drift/projects/{pro_user['user_id']}/baselines",
            params={"agent_name": "agent-a"},
            headers=auth_headers(pro_user["api_key"]),
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["baselines"][0]["agent_name"] == "agent-a"

    @pytest.mark.asyncio
    async def test_calculate_baseline_insufficient_data(
        self, api_gateway_client: AsyncClient, pro_user: dict,
    ):
        """Calculate baselines with no spans returns 0 or message."""
        resp = await api_gateway_client.post(
            f"/api/v1/drift/projects/{pro_user['user_id']}/baselines/calculate",
            headers=auth_headers(pro_user["api_key"]),
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["baselines_calculated"] == 0


class TestAlerts:
    """Test drift alert CRUD."""

    @pytest.mark.asyncio
    async def test_create_alert(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """Create a drift alert."""
        baseline_id = _insert_baseline(clickhouse_client, pro_user["user_id"])
        payload = _make_alert_payload(baseline_id)

        resp = await api_gateway_client.post(
            f"/api/v1/drift/projects/{pro_user['user_id']}/alerts",
            json=payload,
            headers=auth_headers(pro_user["api_key"]),
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "alert_id" in data
        assert data["status"] == "active"
        assert data["severity"] == "high"

    @pytest.mark.asyncio
    async def test_list_alerts(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """List drift alerts returns created alerts."""
        baseline_id = _insert_baseline(clickhouse_client, pro_user["user_id"])
        payload = _make_alert_payload(baseline_id)

        # Create an alert
        create_resp = await api_gateway_client.post(
            f"/api/v1/drift/projects/{pro_user['user_id']}/alerts",
            json=payload,
            headers=auth_headers(pro_user["api_key"]),
        )
        assert create_resp.status_code == 200
        alert_id = create_resp.json()["alert_id"]

        # List alerts
        list_resp = await api_gateway_client.get(
            f"/api/v1/drift/projects/{pro_user['user_id']}/alerts",
            headers=auth_headers(pro_user["api_key"]),
        )

        assert list_resp.status_code == 200
        data = list_resp.json()
        assert data["count"] >= 1
        alert_ids = [a["alert_id"] for a in data["alerts"]]
        assert alert_id in alert_ids

    @pytest.mark.asyncio
    async def test_get_alert_by_id(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """Get a specific alert by ID."""
        baseline_id = _insert_baseline(clickhouse_client, pro_user["user_id"])
        payload = _make_alert_payload(baseline_id)

        create_resp = await api_gateway_client.post(
            f"/api/v1/drift/projects/{pro_user['user_id']}/alerts",
            json=payload,
            headers=auth_headers(pro_user["api_key"]),
        )
        alert_id = create_resp.json()["alert_id"]

        get_resp = await api_gateway_client.get(
            f"/api/v1/drift/projects/{pro_user['user_id']}/alerts/{alert_id}",
            headers=auth_headers(pro_user["api_key"]),
        )

        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["alert_id"] == alert_id
        assert data["agent_name"] == "test-agent"
        assert data["severity"] == "high"
        assert data["status"] == "active"
        assert len(data["anomalies"]) == 1
        assert len(data["root_causes"]) == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent_alert_returns_404(
        self, api_gateway_client: AsyncClient, pro_user: dict,
    ):
        """Getting a non-existent alert returns 404."""
        resp = await api_gateway_client.get(
            f"/api/v1/drift/projects/{pro_user['user_id']}/alerts/nonexistent",
            headers=auth_headers(pro_user["api_key"]),
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_update_alert_status(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """Update an alert's status to acknowledged."""
        baseline_id = _insert_baseline(clickhouse_client, pro_user["user_id"])
        payload = _make_alert_payload(baseline_id)

        create_resp = await api_gateway_client.post(
            f"/api/v1/drift/projects/{pro_user['user_id']}/alerts",
            json=payload,
            headers=auth_headers(pro_user["api_key"]),
        )
        alert_id = create_resp.json()["alert_id"]

        update_resp = await api_gateway_client.patch(
            f"/api/v1/drift/projects/{pro_user['user_id']}/alerts/{alert_id}",
            json={
                "status": "acknowledged",
                "user_id": pro_user["user_id"],
                "notes": "Investigating the issue",
            },
            headers=auth_headers(pro_user["api_key"]),
        )

        assert update_resp.status_code == 200
        data = update_resp.json()
        assert data["alert_id"] == alert_id
        assert data["status"] == "acknowledged"
        assert data["updated"] is True


class TestAlertRules:
    """Test alert rule CRUD."""

    @pytest.mark.asyncio
    async def test_create_alert_rule(
        self, api_gateway_client: AsyncClient, pro_user: dict,
    ):
        """Create an alert rule."""
        resp = await api_gateway_client.post(
            f"/api/v1/drift/projects/{pro_user['user_id']}/alert-rules",
            json={
                "name": "High latency alert",
                "description": "Alert when duration increases by 100%",
                "enabled": True,
                "severity_threshold": "medium",
                "change_percent_min": 100.0,
                "notify_email": False,
                "email_addresses": [],
                "user_id": pro_user["user_id"],
            },
            headers=auth_headers(pro_user["api_key"]),
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "rule_id" in data
        assert data["name"] == "High latency alert"

    @pytest.mark.asyncio
    async def test_list_alert_rules(
        self, api_gateway_client: AsyncClient, pro_user: dict,
    ):
        """List alert rules returns created rules."""
        # Create a rule
        create_resp = await api_gateway_client.post(
            f"/api/v1/drift/projects/{pro_user['user_id']}/alert-rules",
            json={
                "name": "Test rule for listing",
                "enabled": True,
                "severity_threshold": "low",
                "user_id": pro_user["user_id"],
            },
            headers=auth_headers(pro_user["api_key"]),
        )
        assert create_resp.status_code == 200
        rule_id = create_resp.json()["rule_id"]

        # List rules
        list_resp = await api_gateway_client.get(
            f"/api/v1/drift/projects/{pro_user['user_id']}/alert-rules",
            headers=auth_headers(pro_user["api_key"]),
        )

        assert list_resp.status_code == 200
        data = list_resp.json()
        assert data["count"] >= 1
        rule_ids = [r["rule_id"] for r in data["rules"]]
        assert rule_id in rule_ids

    @pytest.mark.asyncio
    async def test_get_alert_rule_by_id(
        self, api_gateway_client: AsyncClient, pro_user: dict,
    ):
        """Get a specific alert rule by ID."""
        create_resp = await api_gateway_client.post(
            f"/api/v1/drift/projects/{pro_user['user_id']}/alert-rules",
            json={
                "name": "Specific rule",
                "description": "For testing get by ID",
                "enabled": True,
                "severity_threshold": "high",
                "user_id": pro_user["user_id"],
            },
            headers=auth_headers(pro_user["api_key"]),
        )
        rule_id = create_resp.json()["rule_id"]

        get_resp = await api_gateway_client.get(
            f"/api/v1/drift/projects/{pro_user['user_id']}/alert-rules/{rule_id}",
            headers=auth_headers(pro_user["api_key"]),
        )

        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["rule_id"] == rule_id
        assert data["name"] == "Specific rule"
        assert data["description"] == "For testing get by ID"

    @pytest.mark.asyncio
    async def test_delete_alert_rule(
        self, api_gateway_client: AsyncClient, pro_user: dict,
    ):
        """Delete an alert rule."""
        create_resp = await api_gateway_client.post(
            f"/api/v1/drift/projects/{pro_user['user_id']}/alert-rules",
            json={
                "name": "Rule to delete",
                "enabled": True,
                "severity_threshold": "low",
                "user_id": pro_user["user_id"],
            },
            headers=auth_headers(pro_user["api_key"]),
        )
        rule_id = create_resp.json()["rule_id"]

        # Delete
        del_resp = await api_gateway_client.delete(
            f"/api/v1/drift/projects/{pro_user['user_id']}/alert-rules/{rule_id}",
            headers=auth_headers(pro_user["api_key"]),
        )

        assert del_resp.status_code == 200
        assert del_resp.json()["deleted"] is True


class TestDriftTierGating:
    """Test tier gating for drift endpoints."""

    @pytest.mark.asyncio
    async def test_free_user_blocked_from_baselines(
        self, api_gateway_client: AsyncClient, free_user: dict,
    ):
        """Free user gets 403 on baseline listing."""
        resp = await api_gateway_client.get(
            f"/api/v1/drift/projects/{free_user['user_id']}/baselines",
            headers=auth_headers(free_user["api_key"]),
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_free_user_blocked_from_alerts(
        self, api_gateway_client: AsyncClient, free_user: dict,
    ):
        """Free user gets 403 on alert listing."""
        resp = await api_gateway_client.get(
            f"/api/v1/drift/projects/{free_user['user_id']}/alerts",
            headers=auth_headers(free_user["api_key"]),
        )
        assert resp.status_code == 403
