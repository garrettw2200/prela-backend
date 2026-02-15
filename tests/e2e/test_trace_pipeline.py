"""Suite 2: Trace Ingestion Pipeline — E2E Tests

Tests the full ingestion flow: SDK → Ingest Gateway → ClickHouse.
Exercises /v1/traces, /v1/batch, /v1/otlp/v1/traces endpoints
against real ClickHouse and PostgreSQL.
"""

import gzip
import json
import time
from pathlib import Path

import pytest
from httpx import AsyncClient

from .conftest import auth_headers, make_trace, make_span

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestTraceIngestion:
    """POST /v1/traces — ingest a complete trace."""

    @pytest.mark.asyncio
    async def test_ingest_trace_accepted(
        self, ingest_client: AsyncClient, pro_user: dict,
    ):
        """A valid trace is accepted and returns 200."""
        trace = make_trace(trace_id="pipeline_001")
        resp = await ingest_client.post(
            "/v1/traces",
            json=trace,
            headers=auth_headers(pro_user["api_key"]),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["trace_id"] == "pipeline_001"

    @pytest.mark.asyncio
    async def test_trace_queryable_in_clickhouse(
        self, ingest_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """An ingested trace is queryable in ClickHouse."""
        trace = make_trace(trace_id="pipeline_ch_001")
        await ingest_client.post(
            "/v1/traces",
            json=trace,
            headers=auth_headers(pro_user["api_key"]),
        )

        # Query ClickHouse directly
        result = clickhouse_client.query(
            "SELECT trace_id, service_name, status FROM traces "
            "WHERE trace_id = %(tid)s",
            parameters={"tid": "pipeline_ch_001"},
        )
        assert len(result.result_rows) == 1
        row = result.result_rows[0]
        assert row[0] == "pipeline_ch_001"
        assert row[1] == "test-service"
        assert row[2] == "completed"

    @pytest.mark.asyncio
    async def test_trace_with_spans(
        self, ingest_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """A trace with multiple spans stores all spans in ClickHouse."""
        tid = "pipeline_spans_001"
        spans = [
            make_span(span_id=f"s{i}", trace_id=tid, name=f"span-{i}")
            for i in range(3)
        ]
        trace = make_trace(trace_id=tid, spans=spans)

        resp = await ingest_client.post(
            "/v1/traces",
            json=trace,
            headers=auth_headers(pro_user["api_key"]),
        )
        assert resp.status_code == 200

        # Verify spans in ClickHouse
        result = clickhouse_client.query(
            "SELECT span_id FROM spans WHERE trace_id = %(tid)s",
            parameters={"tid": tid},
        )
        span_ids = {row[0] for row in result.result_rows}
        assert span_ids == {"s0", "s1", "s2"}

    @pytest.mark.asyncio
    async def test_missing_trace_id_rejected(
        self, ingest_client: AsyncClient, pro_user: dict,
    ):
        """A trace without trace_id is rejected."""
        resp = await ingest_client.post(
            "/v1/traces",
            json={"service_name": "test"},
            headers=auth_headers(pro_user["api_key"]),
        )
        assert resp.status_code == 400


class TestBatchIngestion:
    """POST /v1/batch — ingest spans in batch."""

    @pytest.mark.asyncio
    async def test_batch_spans_ingested(
        self, ingest_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """Batch of 5 spans all appear in ClickHouse."""
        tid = "batch_001"
        spans = [
            make_span(span_id=f"batch_s{i}", trace_id=tid, name=f"batch-span-{i}")
            for i in range(5)
        ]

        resp = await ingest_client.post(
            "/v1/batch",
            json={"spans": spans},
            headers=auth_headers(pro_user["api_key"]),
        )
        assert resp.status_code == 200
        assert resp.json()["ingested"] == 5

        result = clickhouse_client.query(
            "SELECT count() FROM spans WHERE trace_id = %(tid)s",
            parameters={"tid": tid},
        )
        assert result.result_rows[0][0] == 5

    @pytest.mark.asyncio
    async def test_gzip_batch_accepted(
        self, ingest_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """A gzip-compressed batch is accepted and processed."""
        tid = "gzip_batch_001"
        spans = [
            make_span(span_id=f"gz_s{i}", trace_id=tid) for i in range(3)
        ]
        payload = json.dumps({"spans": spans}).encode()
        compressed = gzip.compress(payload)

        resp = await ingest_client.post(
            "/v1/batch",
            content=compressed,
            headers={
                **auth_headers(pro_user["api_key"]),
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["ingested"] == 3

        result = clickhouse_client.query(
            "SELECT count() FROM spans WHERE trace_id = %(tid)s",
            parameters={"tid": tid},
        )
        assert result.result_rows[0][0] == 3


class TestOtlpIngestion:
    """POST /v1/otlp/v1/traces — ingest OTLP format."""

    @pytest.mark.asyncio
    async def test_otlp_traces_ingested(
        self, ingest_client: AsyncClient, pro_user: dict,
        clickhouse_client,
    ):
        """OTLP traces are normalized and stored in ClickHouse."""
        otlp_data = json.loads(
            (FIXTURES_DIR / "traces" / "otlp_trace.json").read_text()
        )

        resp = await ingest_client.post(
            "/v1/otlp/v1/traces",
            json=otlp_data,
            headers=auth_headers(pro_user["api_key"]),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["spans_ingested"] >= 1


class TestInvalidApiKey:
    """Invalid API keys should be rejected."""

    @pytest.mark.asyncio
    async def test_invalid_key_returns_401(
        self, ingest_client: AsyncClient,
    ):
        """A fake API key is rejected with 401."""
        trace = make_trace(trace_id="invalid_key_001")
        resp = await ingest_client.post(
            "/v1/traces",
            json=trace,
            headers=auth_headers("prela_sk_this_key_does_not_exist"),
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(
        self, ingest_client: AsyncClient,
    ):
        """Missing auth returns 401 or 403."""
        trace = make_trace(trace_id="no_auth_001")
        resp = await ingest_client.post(
            "/v1/traces",
            json=trace,
        )
        assert resp.status_code in (401, 403)


class TestProjectIsolation:
    """Traces are isolated per project (user)."""

    @pytest.mark.asyncio
    async def test_traces_isolated_between_users(
        self, ingest_client: AsyncClient, api_gateway_client: AsyncClient,
        pro_user: dict, free_user: dict, clickhouse_client,
    ):
        """Traces ingested by user A are not visible to user B."""
        # User A ingests a trace
        trace_a = make_trace(trace_id="isolation_a_001", service_name="user-a-service")
        await ingest_client.post(
            "/v1/traces",
            json=trace_a,
            headers=auth_headers(pro_user["api_key"]),
        )

        # User B ingests a trace
        trace_b = make_trace(trace_id="isolation_b_001", service_name="user-b-service")
        await ingest_client.post(
            "/v1/traces",
            json=trace_b,
            headers=auth_headers(free_user["api_key"]),
        )

        # Query ClickHouse for user A's project
        result_a = clickhouse_client.query(
            "SELECT trace_id FROM traces WHERE project_id = %(pid)s",
            parameters={"pid": pro_user["user_id"]},
        )
        trace_ids_a = {row[0] for row in result_a.result_rows}
        assert "isolation_a_001" in trace_ids_a
        assert "isolation_b_001" not in trace_ids_a

        # Query for user B's project
        result_b = clickhouse_client.query(
            "SELECT trace_id FROM traces WHERE project_id = %(pid)s",
            parameters={"pid": free_user["user_id"]},
        )
        trace_ids_b = {row[0] for row in result_b.result_rows}
        assert "isolation_b_001" in trace_ids_b
        assert "isolation_a_001" not in trace_ids_b
