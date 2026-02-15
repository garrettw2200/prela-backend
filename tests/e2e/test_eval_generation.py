"""Suite 8: Eval Generation — E2E Tests

Tests the eval generation trigger, status polling, YAML download, and
history listing. The EvalGenerator LLM calls are mocked; status/download/
history tests seed analysis_results directly.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from httpx import AsyncClient

from .conftest import auth_headers

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ============================================================
# Helpers
# ============================================================


def _load_eval_fixture() -> dict:
    """Load the canned eval generation result."""
    return json.loads(
        (FIXTURES_DIR / "llm_responses" / "eval_generation.json").read_text()
    )


def _insert_eval_result(
    clickhouse_client,
    project_id: str,
    generation_id: str | None = None,
    status: str = "completed",
    suite_yaml: str | None = None,
    created_at: datetime | None = None,
) -> str:
    """Insert an eval generation result directly into analysis_results."""
    gen_id = generation_id or str(uuid.uuid4())
    now = created_at or datetime.now(timezone.utc)
    fixture = _load_eval_fixture()

    result_json = json.dumps({
        "suite_name": fixture["suite_name"],
        "suite_yaml": suite_yaml if suite_yaml is not None else fixture["suite_yaml"],
        "cases_generated": fixture["cases_generated"],
        "traces_analyzed": fixture["traces_analyzed"],
        "patterns_found": fixture["patterns_found"],
        "pattern_summary": fixture["pattern_summary"],
        "status": status,
        "error": None if status != "failed" else "Generation failed",
        "started_at": (now - timedelta(seconds=90)).isoformat(),
        "completed_at": now.isoformat() if status == "completed" else None,
    })

    score = fixture["cases_generated"] / max(fixture["traces_analyzed"], 1)

    clickhouse_client.insert(
        "analysis_results",
        [[gen_id, "", project_id, "eval_generation", result_json, score, now]],
        column_names=[
            "result_id", "trace_id", "project_id",
            "analysis_type", "result", "score", "created_at",
        ],
    )

    return gen_id


def _seed_traces_for_generation(clickhouse_client, project_id: str, count: int = 5):
    """Insert sample traces that the eval generator could analyze."""
    now = datetime.now(timezone.utc)

    for i in range(count):
        trace_id = f"eval_trace_{uuid.uuid4().hex[:8]}"
        span_id = f"eval_span_{uuid.uuid4().hex[:8]}"
        started = now - timedelta(hours=i + 1)
        duration = 100.0 + i * 10

        clickhouse_client.insert(
            "traces",
            [[
                trace_id, project_id, "test-service",
                started, started + timedelta(milliseconds=duration),
                duration, "completed", span_id, 1,
                json.dumps({}), "native", now,
            ]],
            column_names=[
                "trace_id", "project_id", "service_name",
                "started_at", "completed_at", "duration_ms", "status",
                "root_span_id", "span_count", "attributes", "source", "created_at",
            ],
        )

        clickhouse_client.insert(
            "spans",
            [[
                span_id, trace_id, project_id,
                "", "llm-call", "llm", "test-service",
                started, started + timedelta(milliseconds=duration),
                duration, "completed",
                json.dumps({
                    "llm.model": "gpt-4o-mini",
                    "llm.prompt_tokens": 50,
                    "llm.completion_tokens": 100,
                }),
                json.dumps([]),
                json.dumps({}),
                "native", now,
            ]],
            column_names=[
                "span_id", "trace_id", "project_id",
                "parent_span_id", "name", "span_type", "service_name",
                "started_at", "ended_at", "duration_ms",
                "status", "attributes", "events", "replay_snapshot",
                "source", "created_at",
            ],
        )


# ============================================================
# Tests
# ============================================================


class TestEvalGeneration:
    """Test eval generation trigger and background task."""

    @pytest.mark.asyncio
    async def test_trigger_generation(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """Triggering eval generation returns generation_id and running status."""
        _seed_traces_for_generation(clickhouse_client, pro_user["user_id"])

        # Mock EvalGenerator so the background task doesn't do real LLM calls
        with patch("app.routers.eval_generation.EvalGenerator"):
            resp = await api_gateway_client.post(
                "/api/v1/eval-generation/generate",
                json={
                    "suite_name": "Test Suite",
                    "time_window_hours": 168,
                    "max_traces": 10,
                    "max_cases": 5,
                },
                params={"project_id": pro_user["user_id"]},
                headers=auth_headers(pro_user["api_key"]),
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "generation_id" in data
        assert data["status"] == "running"
        assert "started_at" in data

    @pytest.mark.asyncio
    async def test_poll_status_running(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """Polling status of a running generation returns running."""
        gen_id = _insert_eval_result(
            clickhouse_client, pro_user["user_id"],
            status="running", suite_yaml=None,
        )

        resp = await api_gateway_client.get(
            f"/api/v1/eval-generation/{gen_id}/status",
            params={"project_id": pro_user["user_id"]},
            headers=auth_headers(pro_user["api_key"]),
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["generation_id"] == gen_id
        assert data["status"] == "running"

    @pytest.mark.asyncio
    async def test_poll_status_completed(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """Polling status of a completed generation returns full details."""
        gen_id = _insert_eval_result(
            clickhouse_client, pro_user["user_id"],
            status="completed",
        )

        resp = await api_gateway_client.get(
            f"/api/v1/eval-generation/{gen_id}/status",
            params={"project_id": pro_user["user_id"]},
            headers=auth_headers(pro_user["api_key"]),
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["generation_id"] == gen_id
        assert data["status"] == "completed"
        assert data["cases_generated"] == 12
        assert data["traces_analyzed"] == 50
        assert data["patterns_found"] == 5
        assert len(data["pattern_summary"]) == 3
        assert data["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_poll_status_not_found(
        self, api_gateway_client: AsyncClient, pro_user: dict,
    ):
        """Polling a non-existent generation returns 404."""
        resp = await api_gateway_client.get(
            "/api/v1/eval-generation/nonexistent_id/status",
            params={"project_id": pro_user["user_id"]},
            headers=auth_headers(pro_user["api_key"]),
        )
        assert resp.status_code == 404


class TestEvalDownload:
    """Test eval suite YAML download."""

    @pytest.mark.asyncio
    async def test_download_completed_yaml(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """Download a completed eval suite as YAML."""
        gen_id = _insert_eval_result(
            clickhouse_client, pro_user["user_id"],
            status="completed",
        )

        resp = await api_gateway_client.get(
            f"/api/v1/eval-generation/{gen_id}/download",
            params={"project_id": pro_user["user_id"]},
            headers=auth_headers(pro_user["api_key"]),
        )

        assert resp.status_code == 200
        assert "application/x-yaml" in resp.headers.get("content-type", "")
        assert "attachment" in resp.headers.get("content-disposition", "")
        # Verify content is YAML-like
        content = resp.text
        assert "name:" in content
        assert "tests:" in content

    @pytest.mark.asyncio
    async def test_download_not_completed_returns_400(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """Downloading a running generation returns 400."""
        gen_id = _insert_eval_result(
            clickhouse_client, pro_user["user_id"],
            status="running", suite_yaml=None,
        )

        resp = await api_gateway_client.get(
            f"/api/v1/eval-generation/{gen_id}/download",
            params={"project_id": pro_user["user_id"]},
            headers=auth_headers(pro_user["api_key"]),
        )

        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_download_not_found_returns_404(
        self, api_gateway_client: AsyncClient, pro_user: dict,
    ):
        """Downloading a non-existent generation returns 404."""
        resp = await api_gateway_client.get(
            "/api/v1/eval-generation/nonexistent_id/download",
            params={"project_id": pro_user["user_id"]},
            headers=auth_headers(pro_user["api_key"]),
        )
        assert resp.status_code == 404


class TestEvalHistory:
    """Test eval generation history listing."""

    @pytest.mark.asyncio
    async def test_history_lists_generations(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """History endpoint lists past generations with pagination."""
        now = datetime.now(timezone.utc)

        # Insert 3 completed generations
        for i in range(3):
            _insert_eval_result(
                clickhouse_client, pro_user["user_id"],
                status="completed",
                created_at=now - timedelta(hours=i),
            )

        resp = await api_gateway_client.get(
            "/api/v1/eval-generation/history",
            params={
                "project_id": pro_user["user_id"],
                "page": 1,
                "page_size": 10,
            },
            headers=auth_headers(pro_user["api_key"]),
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert len(data["generations"]) == 3
        assert data["page"] == 1
        assert data["page_size"] == 10

        # Each generation should have expected fields
        gen = data["generations"][0]
        assert "generation_id" in gen
        assert "suite_name" in gen
        assert gen["status"] == "completed"
        assert gen["cases_generated"] == 12

    @pytest.mark.asyncio
    async def test_history_pagination(
        self, api_gateway_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """History respects page_size limit."""
        now = datetime.now(timezone.utc)

        for i in range(5):
            _insert_eval_result(
                clickhouse_client, pro_user["user_id"],
                status="completed",
                created_at=now - timedelta(hours=i),
            )

        resp = await api_gateway_client.get(
            "/api/v1/eval-generation/history",
            params={
                "project_id": pro_user["user_id"],
                "page": 1,
                "page_size": 2,
            },
            headers=auth_headers(pro_user["api_key"]),
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert len(data["generations"]) == 2


class TestEvalTierGating:
    """Test tier gating for eval generation endpoints."""

    @pytest.mark.asyncio
    async def test_free_user_blocked_from_generation(
        self, api_gateway_client: AsyncClient, free_user: dict,
    ):
        """Free user gets 403 on eval generation."""
        resp = await api_gateway_client.post(
            "/api/v1/eval-generation/generate",
            json={"suite_name": "Test"},
            params={"project_id": free_user["user_id"]},
            headers=auth_headers(free_user["api_key"]),
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_free_user_blocked_from_history(
        self, api_gateway_client: AsyncClient, free_user: dict,
    ):
        """Free user gets 403 on eval history."""
        resp = await api_gateway_client.get(
            "/api/v1/eval-generation/history",
            params={"project_id": free_user["user_id"]},
            headers=auth_headers(free_user["api_key"]),
        )
        assert resp.status_code == 403
