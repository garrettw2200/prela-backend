"""Suite 6: Security Scanning — E2E Tests

Tests the on-demand security scan endpoint and the security summary
aggregation endpoint against real ClickHouse, with SecurityScanner mocked.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from unittest.mock import patch

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


async def _seed_trace_with_llm_spans(clickhouse_client, project_id: str) -> str:
    """Insert a trace with LLM spans for security scanning."""
    trace_id = f"sec_test_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)

    clickhouse_client.insert(
        "traces",
        [[
            trace_id, project_id, "test-service",
            now - timedelta(seconds=1), now,
            1000.0, "completed", "", 2,
            json.dumps({}), "native", now,
        ]],
        column_names=[
            "trace_id", "project_id", "service_name",
            "started_at", "completed_at", "duration_ms", "status",
            "root_span_id", "span_count", "attributes", "source", "created_at",
        ],
    )

    # Insert LLM spans
    for i in range(2):
        span_id = f"llm_span_{i}_{uuid.uuid4().hex[:8]}"
        attrs = {
            "llm.model": "gpt-4o-mini",
            "llm.vendor": "openai",
            "llm.prompt_tokens": 100,
            "llm.completion_tokens": 200,
        }
        clickhouse_client.insert(
            "spans",
            [[
                span_id, trace_id, project_id,
                "", f"llm-call-{i}", "llm",
                "test-service",
                now - timedelta(milliseconds=500),
                now, 500.0, "completed",
                json.dumps(attrs),
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

    return trace_id


async def _seed_trace_without_llm_spans(clickhouse_client, project_id: str) -> str:
    """Insert a trace with only non-LLM spans (tool spans)."""
    trace_id = f"clean_test_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)

    clickhouse_client.insert(
        "traces",
        [[
            trace_id, project_id, "test-service",
            now - timedelta(seconds=1), now,
            500.0, "completed", "", 1,
            json.dumps({}), "native", now,
        ]],
        column_names=[
            "trace_id", "project_id", "service_name",
            "started_at", "completed_at", "duration_ms", "status",
            "root_span_id", "span_count", "attributes", "source", "created_at",
        ],
    )

    # Insert a tool span (not LLM)
    clickhouse_client.insert(
        "spans",
        [[
            f"tool_span_{uuid.uuid4().hex[:8]}", trace_id, project_id,
            "", "fetch-data", "tool",
            "test-service",
            now - timedelta(milliseconds=500),
            now, 500.0, "completed",
            json.dumps({}),
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

    return trace_id


def _make_mock_security_analysis(span_id: str, has_findings: bool = True):
    """Create a mock SecurityAnalysis-like object."""

    class FindingType(str, Enum):
        PII_EXPOSURE = "pii_exposure"
        PROMPT_INJECTION = "prompt_injection"

    class Severity(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    @dataclass
    class SecurityFinding:
        finding_type: FindingType
        severity: Severity
        confidence: float
        matched_text: str
        pattern_name: str
        location: str
        remediation: str

    @dataclass
    class SecurityAnalysis:
        span_id: str
        findings: list = field(default_factory=list)
        overall_severity: Severity = Severity.LOW
        overall_confidence: float = 0.0
        scanned_at: str = ""

    if has_findings:
        findings = [
            SecurityFinding(
                finding_type=FindingType.PII_EXPOSURE,
                severity=Severity.HIGH,
                confidence=0.95,
                matched_text="user@example.com",
                pattern_name="email_pattern",
                location="prompt",
                remediation="Sanitize PII before sending to LLM",
            ),
        ]
        return SecurityAnalysis(
            span_id=span_id,
            findings=findings,
            overall_severity=Severity.HIGH,
            overall_confidence=0.95,
            scanned_at=datetime.now(timezone.utc).isoformat(),
        )
    else:
        return SecurityAnalysis(
            span_id=span_id,
            findings=[],
            overall_severity=Severity.LOW,
            overall_confidence=0.0,
            scanned_at=datetime.now(timezone.utc).isoformat(),
        )


def _insert_security_analysis_result(
    clickhouse_client, project_id: str, trace_id: str,
    created_at: datetime | None = None,
):
    """Insert a pre-computed security analysis result into analysis_results."""
    result_id = str(uuid.uuid4())
    now = created_at or datetime.now(timezone.utc)
    result_json = json.dumps({
        "findings": [
            {
                "finding_type": "pii_exposure",
                "severity": "high",
                "confidence": 0.95,
                "matched_text": "user@example.com",
                "pattern_name": "email_pattern",
                "location": "prompt",
                "remediation": "Sanitize PII",
            },
        ],
        "span_id": "test_span_1",
        "overall_severity": "HIGH",
        "overall_confidence": 0.95,
    })

    clickhouse_client.insert(
        "analysis_results",
        [[result_id, trace_id, project_id, "security", result_json, 0.95, now]],
        column_names=[
            "result_id", "trace_id", "project_id",
            "analysis_type", "result", "score", "created_at",
        ],
    )


# ============================================================
# Tests
# ============================================================


class TestSecurityScan:
    """Test on-demand security scan endpoint."""

    @pytest.mark.asyncio
    async def test_scan_trace_returns_findings(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """Pro user can scan a trace and get security findings."""
        trace_id = await _seed_trace_with_llm_spans(
            clickhouse_client, pro_user["user_id"]
        )

        def mock_analyze(span_data):
            return _make_mock_security_analysis(
                span_data["span_id"], has_findings=True
            )

        with patch("app.routers.security.SecurityScanner") as MockScanner:
            MockScanner.analyze_span = mock_analyze

            resp = await api_gateway_client.get(
                f"/api/v1/security/traces/{trace_id}/scan",
                params={"project_id": pro_user["user_id"]},
                headers=auth_headers(pro_user["api_key"]),
            )

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0

        scan = data[0]
        assert scan["trace_id"] == trace_id
        assert len(scan["findings"]) > 0
        assert scan["findings"][0]["finding_type"] == "pii_exposure"
        assert scan["findings"][0]["severity"] == "high"
        assert scan["overall_severity"] == "high"

    @pytest.mark.asyncio
    async def test_scan_clean_trace_returns_empty(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """Trace with no LLM spans returns empty findings."""
        trace_id = await _seed_trace_without_llm_spans(
            clickhouse_client, pro_user["user_id"]
        )

        with patch("app.routers.security.SecurityScanner") as MockScanner:
            MockScanner.analyze_span = lambda sd: _make_mock_security_analysis(
                sd["span_id"], has_findings=False
            )

            resp = await api_gateway_client.get(
                f"/api/v1/security/traces/{trace_id}/scan",
                params={"project_id": pro_user["user_id"]},
                headers=auth_headers(pro_user["api_key"]),
            )

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        # No LLM spans → no spans scanned → empty list
        assert len(data) == 0

    @pytest.mark.asyncio
    async def test_scan_not_persisted(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """On-demand scan does NOT persist results to analysis_results."""
        trace_id = await _seed_trace_with_llm_spans(
            clickhouse_client, pro_user["user_id"]
        )

        def mock_analyze(span_data):
            return _make_mock_security_analysis(
                span_data["span_id"], has_findings=True
            )

        with patch("app.routers.security.SecurityScanner") as MockScanner:
            MockScanner.analyze_span = mock_analyze

            resp = await api_gateway_client.get(
                f"/api/v1/security/traces/{trace_id}/scan",
                params={"project_id": pro_user["user_id"]},
                headers=auth_headers(pro_user["api_key"]),
            )
            assert resp.status_code == 200

        # Verify nothing was written to analysis_results
        result = clickhouse_client.query(
            "SELECT count() FROM analysis_results "
            "WHERE project_id = %(pid)s AND analysis_type = 'security'",
            parameters={"pid": pro_user["user_id"]},
        )
        assert result.result_rows[0][0] == 0


class TestSecuritySummary:
    """Test security summary aggregation endpoint."""

    @pytest.mark.asyncio
    async def test_summary_aggregates_pre_computed(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """Summary aggregates pre-computed security analysis results."""
        # Insert 3 security results
        for i in range(3):
            _insert_security_analysis_result(
                clickhouse_client, pro_user["user_id"], f"trace_{i}"
            )

        resp = await api_gateway_client.get(
            "/api/v1/security/summary",
            params={"project_id": pro_user["user_id"], "time_window": "7d"},
            headers=auth_headers(pro_user["api_key"]),
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_findings"] == 3  # 1 finding per result × 3 results
        assert "high" in data["by_severity"]
        assert data["by_severity"]["high"] == 3
        assert "pii_exposure" in data["by_type"]
        assert data["time_window"] == "7d"

    @pytest.mark.asyncio
    async def test_summary_respects_time_window(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """Summary filters results by time window."""
        now = datetime.now(timezone.utc)

        # Insert recent result (within 7d)
        _insert_security_analysis_result(
            clickhouse_client, pro_user["user_id"], "recent_trace",
            created_at=now - timedelta(days=1),
        )

        # Insert old result (outside 7d but within 30d)
        _insert_security_analysis_result(
            clickhouse_client, pro_user["user_id"], "old_trace",
            created_at=now - timedelta(days=15),
        )

        # 7d window should only include the recent one
        resp_7d = await api_gateway_client.get(
            "/api/v1/security/summary",
            params={"project_id": pro_user["user_id"], "time_window": "7d"},
            headers=auth_headers(pro_user["api_key"]),
        )
        assert resp_7d.status_code == 200
        assert resp_7d.json()["total_findings"] == 1

        # 30d window should include both
        resp_30d = await api_gateway_client.get(
            "/api/v1/security/summary",
            params={"project_id": pro_user["user_id"], "time_window": "30d"},
            headers=auth_headers(pro_user["api_key"]),
        )
        assert resp_30d.status_code == 200
        assert resp_30d.json()["total_findings"] == 2


class TestSecurityTierGating:
    """Test tier gating for security endpoints."""

    @pytest.mark.asyncio
    async def test_free_user_blocked_from_scan(
        self, api_gateway_client: AsyncClient, free_user: dict,
        clickhouse_client,
    ):
        """Free user gets 403 when trying to scan a trace."""
        trace_id = await _seed_trace_with_llm_spans(
            clickhouse_client, free_user["user_id"]
        )

        resp = await api_gateway_client.get(
            f"/api/v1/security/traces/{trace_id}/scan",
            params={"project_id": free_user["user_id"]},
            headers=auth_headers(free_user["api_key"]),
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_free_user_blocked_from_summary(
        self, api_gateway_client: AsyncClient, free_user: dict,
    ):
        """Free user gets 403 on security summary."""
        resp = await api_gateway_client.get(
            "/api/v1/security/summary",
            params={"project_id": free_user["user_id"]},
            headers=auth_headers(free_user["api_key"]),
        )
        assert resp.status_code == 403
