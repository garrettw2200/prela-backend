"""Tests for OTLP normalizer."""

import json
import sys
import os
import pytest
from datetime import datetime, timezone

# Import directly from the module to avoid loading shared.config (which
# requires environment variables). This keeps tests lightweight.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from otlp_normalizer import (
    normalize_otlp_traces,
    _normalize_span,
    _infer_span_type,
    _parse_otlp_attributes,
    _parse_otlp_events,
    _extract_otlp_value,
    _nanos_to_datetime,
)


# --- Test fixtures ---

SAMPLE_OTLP_PAYLOAD = {
    "resourceSpans": [{
        "resource": {
            "attributes": [
                {"key": "service.name", "value": {"stringValue": "langchain-app"}}
            ]
        },
        "scopeSpans": [{
            "scope": {"name": "opentelemetry.instrumentation.langchain"},
            "spans": [
                {
                    "traceId": "0af7651916cd43dd8448eb211c80319c",
                    "spanId": "b7ad6b7169203331",
                    "name": "ChatOpenAI.invoke",
                    "kind": 3,
                    "startTimeUnixNano": "1706000000000000000",
                    "endTimeUnixNano": "1706000001500000000",
                    "attributes": [
                        {"key": "gen_ai.system", "value": {"stringValue": "openai"}},
                        {"key": "gen_ai.request.model", "value": {"stringValue": "gpt-4"}},
                        {"key": "llm.prompt_tokens", "value": {"intValue": "150"}},
                        {"key": "llm.completion_tokens", "value": {"intValue": "42"}},
                    ],
                    "status": {"code": 1},
                    "events": [
                        {
                            "timeUnixNano": "1706000000500000000",
                            "name": "llm.content.completion",
                            "attributes": [
                                {"key": "content", "value": {"stringValue": "Hello world"}}
                            ]
                        }
                    ]
                },
                {
                    "traceId": "0af7651916cd43dd8448eb211c80319c",
                    "spanId": "a1b2c3d4e5f60718",
                    "parentSpanId": "b7ad6b7169203331",
                    "name": "tool.search",
                    "kind": 1,
                    "startTimeUnixNano": "1706000000100000000",
                    "endTimeUnixNano": "1706000000400000000",
                    "attributes": [
                        {"key": "tool.name", "value": {"stringValue": "web_search"}},
                    ],
                    "status": {"code": 1},
                    "events": []
                }
            ]
        }]
    }]
}


def _make_otlp_payload(spans, service_name="test-service"):
    """Helper to build a minimal OTLP payload."""
    return {
        "resourceSpans": [{
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": service_name}}
                ]
            },
            "scopeSpans": [{"scope": {"name": "test"}, "spans": spans}]
        }]
    }


def _make_otlp_span(
    trace_id="aabbccdd11223344aabbccdd11223344",
    span_id="1122334455667788",
    parent_span_id=None,
    name="test-span",
    kind=1,
    start_ns="1706000000000000000",
    end_ns="1706000001000000000",
    attributes=None,
    status_code=1,
    events=None,
):
    """Helper to build a single OTLP span."""
    span = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "kind": kind,
        "startTimeUnixNano": start_ns,
        "endTimeUnixNano": end_ns,
        "attributes": attributes or [],
        "status": {"code": status_code},
        "events": events or [],
    }
    if parent_span_id:
        span["parentSpanId"] = parent_span_id
    return span


# --- Tests for normalize_otlp_traces ---

class TestNormalizeOtlpTraces:
    """Test the top-level normalize_otlp_traces function."""

    def test_empty_resource_spans(self):
        trace_rows, span_rows = normalize_otlp_traces({"resourceSpans": []}, "proj-1")
        assert trace_rows == []
        assert span_rows == []

    def test_missing_resource_spans(self):
        trace_rows, span_rows = normalize_otlp_traces({}, "proj-1")
        assert trace_rows == []
        assert span_rows == []

    def test_single_span_single_trace(self):
        payload = _make_otlp_payload([_make_otlp_span()])
        trace_rows, span_rows = normalize_otlp_traces(payload, "proj-1")
        assert len(trace_rows) == 1
        assert len(span_rows) == 1
        assert trace_rows[0]["trace_id"] == "aabbccdd11223344aabbccdd11223344"
        assert span_rows[0]["trace_id"] == "aabbccdd11223344aabbccdd11223344"

    def test_multiple_spans_same_trace(self):
        spans = [
            _make_otlp_span(span_id="1111111111111111", name="root"),
            _make_otlp_span(span_id="2222222222222222", parent_span_id="1111111111111111", name="child"),
        ]
        payload = _make_otlp_payload(spans)
        trace_rows, span_rows = normalize_otlp_traces(payload, "proj-1")
        assert len(trace_rows) == 1
        assert len(span_rows) == 2
        assert trace_rows[0]["span_count"] == 2

    def test_multiple_traces(self):
        spans = [
            _make_otlp_span(trace_id="aaaa" * 8, span_id="1111111111111111"),
            _make_otlp_span(trace_id="bbbb" * 8, span_id="2222222222222222"),
        ]
        payload = _make_otlp_payload(spans)
        trace_rows, span_rows = normalize_otlp_traces(payload, "proj-1")
        assert len(trace_rows) == 2
        assert len(span_rows) == 2
        trace_ids = {t["trace_id"] for t in trace_rows}
        assert "aaaa" * 8 in trace_ids
        assert "bbbb" * 8 in trace_ids

    def test_source_field_is_otlp(self):
        payload = _make_otlp_payload([_make_otlp_span()])
        trace_rows, span_rows = normalize_otlp_traces(payload, "proj-1")
        assert trace_rows[0]["source"] == "otlp"
        assert span_rows[0]["source"] == "otlp"

    def test_project_id_propagated(self):
        payload = _make_otlp_payload([_make_otlp_span()])
        trace_rows, span_rows = normalize_otlp_traces(payload, "my-project")
        assert trace_rows[0]["project_id"] == "my-project"
        assert span_rows[0]["project_id"] == "my-project"

    def test_service_name_extracted(self):
        payload = _make_otlp_payload([_make_otlp_span()], service_name="my-svc")
        trace_rows, span_rows = normalize_otlp_traces(payload, "proj-1")
        assert span_rows[0]["service_name"] == "my-svc"
        assert trace_rows[0]["service_name"] == "my-svc"

    def test_trace_aggregation_timing(self):
        spans = [
            _make_otlp_span(
                span_id="1111111111111111",
                start_ns="1706000000000000000",
                end_ns="1706000003000000000",
            ),
            _make_otlp_span(
                span_id="2222222222222222",
                parent_span_id="1111111111111111",
                start_ns="1706000001000000000",
                end_ns="1706000002000000000",
            ),
        ]
        payload = _make_otlp_payload(spans)
        trace_rows, _ = normalize_otlp_traces(payload, "proj-1")
        trace = trace_rows[0]
        # started_at should be min (first span's start)
        assert trace["started_at"] == _nanos_to_datetime("1706000000000000000")
        # completed_at should be max (first span's end)
        assert trace["completed_at"] == _nanos_to_datetime("1706000003000000000")
        assert trace["duration_ms"] == pytest.approx(3000.0)

    def test_trace_error_status_propagation(self):
        spans = [
            _make_otlp_span(span_id="1111111111111111", status_code=1),
            _make_otlp_span(span_id="2222222222222222", parent_span_id="1111111111111111", status_code=2),
        ]
        payload = _make_otlp_payload(spans)
        trace_rows, _ = normalize_otlp_traces(payload, "proj-1")
        assert trace_rows[0]["status"] == "error"

    def test_trace_success_status(self):
        spans = [
            _make_otlp_span(span_id="1111111111111111", status_code=1),
            _make_otlp_span(span_id="2222222222222222", parent_span_id="1111111111111111", status_code=1),
        ]
        payload = _make_otlp_payload(spans)
        trace_rows, _ = normalize_otlp_traces(payload, "proj-1")
        assert trace_rows[0]["status"] == "success"

    def test_root_span_id_detection(self):
        spans = [
            _make_otlp_span(span_id="2222222222222222", parent_span_id="1111111111111111"),
            _make_otlp_span(span_id="1111111111111111"),  # root (no parent)
        ]
        payload = _make_otlp_payload(spans)
        trace_rows, _ = normalize_otlp_traces(payload, "proj-1")
        assert trace_rows[0]["root_span_id"] == "1111111111111111"

    def test_sample_langchain_payload(self):
        """End-to-end test with a realistic LangChain-like OTLP payload."""
        trace_rows, span_rows = normalize_otlp_traces(SAMPLE_OTLP_PAYLOAD, "user-123")
        assert len(trace_rows) == 1
        assert len(span_rows) == 2

        # Check LLM span
        llm_span = next(s for s in span_rows if s["name"] == "ChatOpenAI.invoke")
        assert llm_span["span_type"] == "llm"
        attrs = json.loads(llm_span["attributes"])
        assert attrs["gen_ai.system"] == "openai"
        assert attrs["llm.prompt_tokens"] == 150

        # Check tool span
        tool_span = next(s for s in span_rows if s["name"] == "tool.search")
        assert tool_span["span_type"] == "tool"
        assert tool_span["parent_span_id"] == "b7ad6b7169203331"

        # Check events on LLM span
        events = json.loads(llm_span["events"])
        assert len(events) == 1
        assert events[0]["name"] == "llm.content.completion"


# --- Tests for _normalize_span ---

class TestNormalizeSpan:
    """Test per-span normalization."""

    def test_trace_id_hex_preserved(self):
        span = _make_otlp_span(trace_id="0af7651916cd43dd8448eb211c80319c")
        result = _normalize_span(span, "proj-1", "svc")
        assert result["trace_id"] == "0af7651916cd43dd8448eb211c80319c"

    def test_span_id_hex_preserved(self):
        span = _make_otlp_span(span_id="b7ad6b7169203331")
        result = _normalize_span(span, "proj-1", "svc")
        assert result["span_id"] == "b7ad6b7169203331"

    def test_timestamp_conversion(self):
        span = _make_otlp_span(
            start_ns="1706000000000000000",
            end_ns="1706000001500000000",
        )
        result = _normalize_span(span, "proj-1", "svc")
        assert result["started_at"].year == 2024
        assert result["started_at"].tzinfo == timezone.utc

    def test_duration_calculation(self):
        span = _make_otlp_span(
            start_ns="1706000000000000000",
            end_ns="1706000001500000000",
        )
        result = _normalize_span(span, "proj-1", "svc")
        assert result["duration_ms"] == pytest.approx(1500.0)

    def test_status_code_mapping_unset(self):
        span = _make_otlp_span(status_code=0)
        result = _normalize_span(span, "proj-1", "svc")
        assert result["status"] == "pending"

    def test_status_code_mapping_ok(self):
        span = _make_otlp_span(status_code=1)
        result = _normalize_span(span, "proj-1", "svc")
        assert result["status"] == "success"

    def test_status_code_mapping_error(self):
        span = _make_otlp_span(status_code=2)
        result = _normalize_span(span, "proj-1", "svc")
        assert result["status"] == "error"

    def test_parent_span_id_present(self):
        span = _make_otlp_span(parent_span_id="aabb112233445566")
        result = _normalize_span(span, "proj-1", "svc")
        assert result["parent_span_id"] == "aabb112233445566"

    def test_parent_span_id_absent(self):
        span = _make_otlp_span()
        result = _normalize_span(span, "proj-1", "svc")
        assert result["parent_span_id"] == ""

    def test_replay_snapshot_empty(self):
        span = _make_otlp_span()
        result = _normalize_span(span, "proj-1", "svc")
        assert result["replay_snapshot"] == json.dumps({})

    def test_attributes_serialized_as_json(self):
        span = _make_otlp_span(attributes=[
            {"key": "foo", "value": {"stringValue": "bar"}}
        ])
        result = _normalize_span(span, "proj-1", "svc")
        attrs = json.loads(result["attributes"])
        assert attrs["foo"] == "bar"

    def test_events_serialized_as_json(self):
        span = _make_otlp_span(events=[{
            "timeUnixNano": "1706000000000000000",
            "name": "test-event",
            "attributes": []
        }])
        result = _normalize_span(span, "proj-1", "svc")
        events = json.loads(result["events"])
        assert len(events) == 1
        assert events[0]["name"] == "test-event"


# --- Tests for _infer_span_type ---

class TestInferSpanType:
    """Test span type inference heuristics."""

    def test_gen_ai_attributes_infer_llm(self):
        assert _infer_span_type(3, {"gen_ai.system": "openai"}) == "llm"

    def test_gen_ai_request_model_infer_llm(self):
        assert _infer_span_type(1, {"gen_ai.request.model": "gpt-4"}) == "llm"

    def test_llm_attributes_infer_llm(self):
        assert _infer_span_type(3, {"llm.model": "claude-3"}) == "llm"

    def test_db_attributes_infer_retrieval(self):
        assert _infer_span_type(3, {"db.system": "postgresql"}) == "retrieval"

    def test_retrieval_attributes_infer_retrieval(self):
        assert _infer_span_type(1, {"retrieval.source": "pinecone"}) == "retrieval"

    def test_embedding_attributes_infer_embedding(self):
        assert _infer_span_type(3, {"embedding.model": "text-embedding-3"}) == "embedding"

    def test_tool_attributes_infer_tool(self):
        assert _infer_span_type(1, {"tool.name": "web_search"}) == "tool"

    def test_no_semantic_attrs_kind_internal(self):
        assert _infer_span_type(1, {}) == "custom"

    def test_no_semantic_attrs_kind_server(self):
        assert _infer_span_type(2, {}) == "tool"

    def test_no_semantic_attrs_kind_client(self):
        assert _infer_span_type(3, {}) == "custom"

    def test_no_semantic_attrs_kind_unspecified(self):
        assert _infer_span_type(0, {}) == "custom"

    def test_llm_takes_priority_over_tool(self):
        """When both gen_ai and tool attrs present, llm wins."""
        attrs = {"gen_ai.system": "openai", "tool.name": "chat"}
        assert _infer_span_type(1, attrs) == "llm"


# --- Tests for _parse_otlp_attributes ---

class TestParseOtlpAttributes:
    """Test OTLP attribute parsing."""

    def test_string_value(self):
        attrs = [{"key": "foo", "value": {"stringValue": "bar"}}]
        assert _parse_otlp_attributes(attrs) == {"foo": "bar"}

    def test_int_value(self):
        attrs = [{"key": "count", "value": {"intValue": "42"}}]
        result = _parse_otlp_attributes(attrs)
        assert result["count"] == 42
        assert isinstance(result["count"], int)

    def test_double_value(self):
        attrs = [{"key": "score", "value": {"doubleValue": 0.95}}]
        result = _parse_otlp_attributes(attrs)
        assert result["score"] == pytest.approx(0.95)

    def test_bool_value(self):
        attrs = [{"key": "enabled", "value": {"boolValue": True}}]
        assert _parse_otlp_attributes(attrs) == {"enabled": True}

    def test_array_value(self):
        attrs = [{"key": "tags", "value": {"arrayValue": {"values": [
            {"stringValue": "a"},
            {"stringValue": "b"},
        ]}}}]
        assert _parse_otlp_attributes(attrs) == {"tags": ["a", "b"]}

    def test_empty_attributes(self):
        assert _parse_otlp_attributes([]) == {}

    def test_mixed_types(self):
        attrs = [
            {"key": "name", "value": {"stringValue": "test"}},
            {"key": "count", "value": {"intValue": "5"}},
            {"key": "rate", "value": {"doubleValue": 0.8}},
            {"key": "active", "value": {"boolValue": False}},
        ]
        result = _parse_otlp_attributes(attrs)
        assert result == {"name": "test", "count": 5, "rate": pytest.approx(0.8), "active": False}

    def test_empty_key_skipped(self):
        attrs = [{"key": "", "value": {"stringValue": "val"}}]
        assert _parse_otlp_attributes(attrs) == {}

    def test_bytes_value(self):
        attrs = [{"key": "data", "value": {"bytesValue": "aGVsbG8="}}]
        assert _parse_otlp_attributes(attrs) == {"data": "aGVsbG8="}


# --- Tests for _parse_otlp_events ---

class TestParseOtlpEvents:
    """Test OTLP event parsing."""

    def test_event_timestamp_conversion(self):
        events = [{
            "timeUnixNano": "1706000000000000000",
            "name": "test",
            "attributes": [],
        }]
        result = _parse_otlp_events(events)
        assert len(result) == 1
        # Should be ISO 8601 string
        ts = datetime.fromisoformat(result[0]["timestamp"])
        assert ts.year == 2024

    def test_event_name(self):
        events = [{"timeUnixNano": "1706000000000000000", "name": "my.event", "attributes": []}]
        result = _parse_otlp_events(events)
        assert result[0]["name"] == "my.event"

    def test_event_attributes_parsed(self):
        events = [{
            "timeUnixNano": "1706000000000000000",
            "name": "test",
            "attributes": [
                {"key": "msg", "value": {"stringValue": "hello"}}
            ],
        }]
        result = _parse_otlp_events(events)
        assert result[0]["attributes"] == {"msg": "hello"}

    def test_empty_events(self):
        assert _parse_otlp_events([]) == []

    def test_multiple_events(self):
        events = [
            {"timeUnixNano": "1706000000000000000", "name": "start", "attributes": []},
            {"timeUnixNano": "1706000001000000000", "name": "end", "attributes": []},
        ]
        result = _parse_otlp_events(events)
        assert len(result) == 2
        assert result[0]["name"] == "start"
        assert result[1]["name"] == "end"


# --- Tests for _nanos_to_datetime ---

class TestNanosToDatetime:
    """Test nanosecond timestamp conversion."""

    def test_valid_timestamp(self):
        dt = _nanos_to_datetime("1706000000000000000")
        assert dt.tzinfo == timezone.utc
        assert dt.year == 2024

    def test_zero_timestamp(self):
        dt = _nanos_to_datetime("0")
        assert dt.year == 1970

    def test_invalid_timestamp_falls_back(self):
        dt = _nanos_to_datetime("not-a-number")
        assert dt.tzinfo == timezone.utc
        # Should be roughly "now"
        assert dt.year >= 2024

    def test_microsecond_precision(self):
        # 1706000000123456789 ns = 1706000000.123456789 s
        # microsecond precision: 123456 us (last 3 digits lost)
        dt = _nanos_to_datetime("1706000000123456789")
        assert dt.microsecond == 123456 or dt.microsecond == 123457  # allow rounding


# --- Tests for _extract_otlp_value ---

class TestExtractOtlpValue:
    """Test OTLP value extraction."""

    def test_string_value(self):
        assert _extract_otlp_value({"stringValue": "hello"}) == "hello"

    def test_int_value_as_string(self):
        assert _extract_otlp_value({"intValue": "42"}) == 42

    def test_int_value_as_int(self):
        assert _extract_otlp_value({"intValue": 42}) == 42

    def test_double_value(self):
        assert _extract_otlp_value({"doubleValue": 3.14}) == pytest.approx(3.14)

    def test_bool_value(self):
        assert _extract_otlp_value({"boolValue": True}) is True

    def test_array_value(self):
        result = _extract_otlp_value({"arrayValue": {"values": [
            {"stringValue": "a"}, {"intValue": "1"}
        ]}})
        assert result == ["a", 1]

    def test_empty_object(self):
        assert _extract_otlp_value({}) is None
