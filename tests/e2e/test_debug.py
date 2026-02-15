"""Suite 5: Debug Agent — E2E Tests

Tests the debug trace analysis endpoint against real ClickHouse,
with the DebugAgent LLM calls mocked.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from httpx import AsyncClient

from .conftest import auth_headers

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ============================================================
# Helpers
# ============================================================

def _parse_iso(s: str) -> datetime:
    """Parse ISO 8601 timestamp to datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


async def _seed_trace_with_error_span(clickhouse_client, project_id: str) -> str:
    """Insert a trace with an error span into ClickHouse."""
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


def _make_mock_debug_analysis(trace_id: str):
    """Create a mock DebugAnalysis-like object from the fixture."""
    canned = json.loads(
        (FIXTURES_DIR / "llm_responses" / "debug_analysis.json").read_text()
    )

    @dataclass
    class TimelineEntry:
        span_id: str
        name: str
        span_type: str
        started_at: str
        duration_ms: float
        status: str
        error_message: str | None = None
        parent_span_id: str | None = None

    @dataclass
    class FailureChainEntry:
        span_id: str
        name: str
        span_type: str
        error_message: str
        is_root_cause: bool = False

    @dataclass
    class DebugAnalysis:
        trace_id: str
        root_cause: str
        explanation: str
        fix_suggestions: list
        execution_timeline: list
        failure_chain: list
        confidence_score: float
        analyzed_at: str = field(
            default_factory=lambda: datetime.now(timezone.utc).isoformat()
        )

        def to_dict(self):
            return {
                "trace_id": self.trace_id,
                "root_cause": self.root_cause,
                "explanation": self.explanation,
                "fix_suggestions": self.fix_suggestions,
                "execution_timeline": [
                    {"span_id": e.span_id, "name": e.name, "span_type": e.span_type,
                     "started_at": e.started_at, "duration_ms": e.duration_ms,
                     "status": e.status, "error_message": e.error_message,
                     "parent_span_id": e.parent_span_id}
                    for e in self.execution_timeline
                ],
                "failure_chain": [
                    {"span_id": e.span_id, "name": e.name, "span_type": e.span_type,
                     "error_message": e.error_message, "is_root_cause": e.is_root_cause}
                    for e in self.failure_chain
                ],
                "confidence_score": self.confidence_score,
                "analyzed_at": self.analyzed_at,
            }

    timeline = [
        TimelineEntry(
            span_id=f"span-{i}", name=e.get("event", ""),
            span_type="agent", started_at=e.get("timestamp", ""),
            duration_ms=1.0, status="completed"
        )
        for i, e in enumerate(canned.get("execution_timeline", []))
    ]
    failure_chain = [
        FailureChainEntry(
            span_id=f"fail-{i}", name=f.split(":")[0] if ":" in f else f,
            span_type="llm", error_message=f, is_root_cause=(i == 0)
        )
        for i, f in enumerate(canned.get("failure_chain", []))
    ]

    return DebugAnalysis(
        trace_id=trace_id,
        root_cause=canned["root_cause"],
        explanation=canned["explanation"],
        fix_suggestions=canned["fix_suggestions"],
        execution_timeline=timeline,
        failure_chain=failure_chain,
        confidence_score=0.85,
    )


# ============================================================
# Tests
# ============================================================


class TestDebugAnalysis:
    """Test debug trace analysis endpoint."""

    @pytest.mark.asyncio
    async def test_debug_trace_returns_analysis(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """Pro user can debug a trace and get analysis."""
        trace_id = await _seed_trace_with_error_span(
            clickhouse_client, pro_user["user_id"]
        )
        mock_result = _make_mock_debug_analysis(trace_id)

        with patch("app.routers.debug.DebugAgent") as MockAgent:
            instance = MockAgent.return_value
            instance.analyze_trace.return_value = mock_result

            resp = await api_gateway_client.post(
                f"/api/v1/debug/traces/{trace_id}",
                params={"project_id": pro_user["user_id"]},
                headers=auth_headers(pro_user["api_key"]),
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["trace_id"] == trace_id
        assert data["root_cause"] != ""
        assert data["explanation"] != ""
        assert len(data["fix_suggestions"]) > 0
        assert len(data["execution_timeline"]) > 0
        assert len(data["failure_chain"]) > 0
        assert data["cached"] is False
        assert "confidence_score" in data
        assert "analyzed_at" in data

    @pytest.mark.asyncio
    async def test_debug_result_is_cached(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """Second call to debug same trace returns cached result."""
        trace_id = await _seed_trace_with_error_span(
            clickhouse_client, pro_user["user_id"]
        )
        mock_result = _make_mock_debug_analysis(trace_id)

        with patch("app.routers.debug.DebugAgent") as MockAgent:
            instance = MockAgent.return_value
            instance.analyze_trace.return_value = mock_result

            # First call — fresh analysis
            resp1 = await api_gateway_client.post(
                f"/api/v1/debug/traces/{trace_id}",
                params={"project_id": pro_user["user_id"]},
                headers=auth_headers(pro_user["api_key"]),
            )
            assert resp1.status_code == 200
            assert resp1.json()["cached"] is False

            # Second call — should hit cache
            resp2 = await api_gateway_client.post(
                f"/api/v1/debug/traces/{trace_id}",
                params={"project_id": pro_user["user_id"]},
                headers=auth_headers(pro_user["api_key"]),
            )
            assert resp2.status_code == 200
            assert resp2.json()["cached"] is True

            # LLM should only have been called once
            assert instance.analyze_trace.call_count == 1

    @pytest.mark.asyncio
    async def test_force_refresh_bypasses_cache(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """force=true bypasses the cache and re-runs analysis."""
        trace_id = await _seed_trace_with_error_span(
            clickhouse_client, pro_user["user_id"]
        )
        mock_result = _make_mock_debug_analysis(trace_id)

        with patch("app.routers.debug.DebugAgent") as MockAgent:
            instance = MockAgent.return_value
            instance.analyze_trace.return_value = mock_result

            # First call — populate cache
            resp1 = await api_gateway_client.post(
                f"/api/v1/debug/traces/{trace_id}",
                params={"project_id": pro_user["user_id"]},
                headers=auth_headers(pro_user["api_key"]),
            )
            assert resp1.status_code == 200

            # Second call with force=true — should bypass cache
            resp2 = await api_gateway_client.post(
                f"/api/v1/debug/traces/{trace_id}",
                params={"project_id": pro_user["user_id"], "force": True},
                headers=auth_headers(pro_user["api_key"]),
            )
            assert resp2.status_code == 200
            assert resp2.json()["cached"] is False

            # LLM should have been called twice
            assert instance.analyze_trace.call_count == 2

    @pytest.mark.asyncio
    async def test_trace_not_found_returns_404(
        self, api_gateway_client: AsyncClient, pro_user: dict,
    ):
        """Debug non-existent trace returns 404."""
        with patch("app.routers.debug.DebugAgent") as MockAgent:
            resp = await api_gateway_client.post(
                "/api/v1/debug/traces/nonexistent_trace_id",
                params={"project_id": pro_user["user_id"]},
                headers=auth_headers(pro_user["api_key"]),
            )

        assert resp.status_code == 404


class TestDebugTierGating:
    """Test tier gating for debug endpoint."""

    @pytest.mark.asyncio
    async def test_free_user_blocked(
        self, api_gateway_client: AsyncClient, free_user: dict,
        clickhouse_client,
    ):
        """Free user gets 403 when trying to debug a trace."""
        trace_id = await _seed_trace_with_error_span(
            clickhouse_client, free_user["user_id"]
        )

        resp = await api_gateway_client.post(
            f"/api/v1/debug/traces/{trace_id}",
            params={"project_id": free_user["user_id"]},
            headers=auth_headers(free_user["api_key"]),
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_lunch_money_user_blocked(
        self, api_gateway_client: AsyncClient, lunch_money_user: dict,
        clickhouse_client,
    ):
        """Lunch-money user gets 403 (debug is pro-only)."""
        trace_id = await _seed_trace_with_error_span(
            clickhouse_client, lunch_money_user["user_id"]
        )

        resp = await api_gateway_client.post(
            f"/api/v1/debug/traces/{trace_id}",
            params={"project_id": lunch_money_user["user_id"]},
            headers=auth_headers(lunch_money_user["api_key"]),
        )
        assert resp.status_code == 403
