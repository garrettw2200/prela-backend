"""Tests for Langfuse normalizer."""

import json
import os
import sys
import pytest
from datetime import datetime, timezone

# Import directly from module to avoid loading shared.config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from langfuse_normalizer import (
    normalize_langfuse_traces,
    _normalize_single_trace,
    _normalize_observation,
    _parse_iso_timestamp,
    _OBSERVATION_TYPE_MAP,
    _LEVEL_STATUS_MAP,
)


# --- Test fixtures ---

SAMPLE_LANGFUSE_TRACE = {
    "id": "trace-abc-123",
    "name": "chat-completion",
    "timestamp": "2026-02-13T10:30:00.000Z",
    "userId": "user-42",
    "sessionId": "session-99",
    "tags": ["production", "gpt-4"],
    "release": "v1.2.3",
    "version": "1.0",
    "metadata": {"env": "prod"},
    "input": {"prompt": "Hello"},
    "output": {"response": "Hi there!"},
}

SAMPLE_OBSERVATIONS = [
    {
        "id": "obs-gen-001",
        "traceId": "trace-abc-123",
        "type": "GENERATION",
        "name": "ChatOpenAI",
        "startTime": "2026-02-13T10:30:00.100Z",
        "endTime": "2026-02-13T10:30:01.500Z",
        "model": "gpt-4",
        "modelParameters": {"temperature": 0.7},
        "input": {"messages": [{"role": "user", "content": "Hello"}]},
        "output": {"content": "Hi there!"},
        "usage": {
            "promptTokens": 150,
            "completionTokens": 42,
            "totalTokens": 192,
        },
        "level": "DEFAULT",
        "metadata": {"cost": 0.005},
        "parentObservationId": None,
    },
    {
        "id": "obs-span-002",
        "traceId": "trace-abc-123",
        "type": "SPAN",
        "name": "process-input",
        "startTime": "2026-02-13T10:30:00.000Z",
        "endTime": "2026-02-13T10:30:00.100Z",
        "level": "DEFAULT",
        "parentObservationId": None,
    },
    {
        "id": "obs-tool-003",
        "traceId": "trace-abc-123",
        "type": "TOOL",
        "name": "web-search",
        "startTime": "2026-02-13T10:30:00.050Z",
        "endTime": "2026-02-13T10:30:00.090Z",
        "level": "DEFAULT",
        "parentObservationId": "obs-span-002",
    },
]


# --- Timestamp parsing ---


class TestParseIsoTimestamp:
    def test_utc_z_suffix(self):
        dt = _parse_iso_timestamp("2026-02-13T10:30:00.000Z")
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 2
        assert dt.day == 13
        assert dt.hour == 10
        assert dt.minute == 30
        assert dt.tzinfo is not None

    def test_utc_offset(self):
        dt = _parse_iso_timestamp("2026-02-13T10:30:00+00:00")
        assert dt is not None
        assert dt.hour == 10

    def test_no_timezone(self):
        """Timestamps without timezone should be treated as UTC."""
        dt = _parse_iso_timestamp("2026-02-13T10:30:00")
        assert dt is not None
        assert dt.tzinfo == timezone.utc

    def test_none_value(self):
        assert _parse_iso_timestamp(None) is None

    def test_empty_string(self):
        assert _parse_iso_timestamp("") is None

    def test_non_string(self):
        assert _parse_iso_timestamp(12345) is None

    def test_invalid_format(self):
        assert _parse_iso_timestamp("not-a-date") is None

    def test_with_milliseconds(self):
        dt = _parse_iso_timestamp("2026-02-13T10:30:00.123Z")
        assert dt is not None
        assert dt.microsecond == 123000


# --- Observation type mapping ---


class TestObservationTypeMapping:
    def test_generation_maps_to_llm(self):
        assert _OBSERVATION_TYPE_MAP["GENERATION"] == "llm"

    def test_span_maps_to_custom(self):
        assert _OBSERVATION_TYPE_MAP["SPAN"] == "custom"

    def test_chain_maps_to_custom(self):
        assert _OBSERVATION_TYPE_MAP["CHAIN"] == "custom"

    def test_agent_maps_to_custom(self):
        assert _OBSERVATION_TYPE_MAP["AGENT"] == "custom"

    def test_retriever_maps_to_retrieval(self):
        assert _OBSERVATION_TYPE_MAP["RETRIEVER"] == "retrieval"

    def test_embedding_maps_to_embedding(self):
        assert _OBSERVATION_TYPE_MAP["EMBEDDING"] == "embedding"

    def test_tool_maps_to_tool(self):
        assert _OBSERVATION_TYPE_MAP["TOOL"] == "tool"

    def test_event_maps_to_none(self):
        """EVENTs should be skipped (None)."""
        assert _OBSERVATION_TYPE_MAP["EVENT"] is None


# --- Level to status mapping ---


class TestLevelStatusMapping:
    def test_default_is_success(self):
        assert _LEVEL_STATUS_MAP["DEFAULT"] == "success"

    def test_debug_is_success(self):
        assert _LEVEL_STATUS_MAP["DEBUG"] == "success"

    def test_warning_is_success(self):
        assert _LEVEL_STATUS_MAP["WARNING"] == "success"

    def test_error_is_error(self):
        assert _LEVEL_STATUS_MAP["ERROR"] == "error"


# --- Single observation normalization ---


class TestNormalizeObservation:
    def test_generation_observation(self):
        obs = SAMPLE_OBSERVATIONS[0]
        row = _normalize_observation(obs, "trace-abc-123", "proj-1", "llm")

        assert row["span_id"] == "obs-gen-001"
        assert row["trace_id"] == "trace-abc-123"
        assert row["project_id"] == "proj-1"
        assert row["span_type"] == "llm"
        assert row["name"] == "ChatOpenAI"
        assert row["source"] == "langfuse"
        assert row["status"] == "success"

        # Check duration
        assert row["duration_ms"] == pytest.approx(1400, abs=10)

        # Check attributes contain LLM-specific fields
        attrs = json.loads(row["attributes"])
        assert attrs["gen_ai.request.model"] == "gpt-4"
        assert attrs["gen_ai.usage.prompt_tokens"] == 150
        assert attrs["gen_ai.usage.completion_tokens"] == 42
        assert attrs["gen_ai.usage.total_tokens"] == 192

    def test_span_observation(self):
        obs = SAMPLE_OBSERVATIONS[1]
        row = _normalize_observation(obs, "trace-abc-123", "proj-1", "custom")

        assert row["span_id"] == "obs-span-002"
        assert row["span_type"] == "custom"
        assert row["parent_span_id"] == ""

    def test_tool_observation_with_parent(self):
        obs = SAMPLE_OBSERVATIONS[2]
        row = _normalize_observation(obs, "trace-abc-123", "proj-1", "tool")

        assert row["span_id"] == "obs-tool-003"
        assert row["span_type"] == "tool"
        assert row["parent_span_id"] == "obs-span-002"

    def test_error_level(self):
        obs = {
            "id": "obs-err",
            "traceId": "t1",
            "type": "GENERATION",
            "name": "failing-call",
            "startTime": "2026-02-13T10:00:00Z",
            "endTime": "2026-02-13T10:00:01Z",
            "level": "ERROR",
            "statusMessage": "Rate limit exceeded",
        }
        row = _normalize_observation(obs, "t1", "proj-1", "llm")
        assert row["status"] == "error"

        attrs = json.loads(row["attributes"])
        assert attrs["langfuse.status_message"] == "Rate limit exceeded"

    def test_missing_end_time_defaults_to_start(self):
        obs = {
            "id": "obs-no-end",
            "traceId": "t1",
            "type": "SPAN",
            "name": "no-end",
            "startTime": "2026-02-13T10:00:00Z",
        }
        row = _normalize_observation(obs, "t1", "proj-1", "custom")
        assert row["duration_ms"] == 0

    def test_events_field_is_empty_json_array(self):
        obs = SAMPLE_OBSERVATIONS[0]
        row = _normalize_observation(obs, "t1", "proj-1", "llm")
        assert json.loads(row["events"]) == []

    def test_replay_snapshot_is_empty_json_object(self):
        obs = SAMPLE_OBSERVATIONS[0]
        row = _normalize_observation(obs, "t1", "proj-1", "llm")
        assert json.loads(row["replay_snapshot"]) == {}


# --- Single trace normalization ---


class TestNormalizeSingleTrace:
    def test_basic_trace(self):
        trace_row, span_rows = _normalize_single_trace(
            SAMPLE_LANGFUSE_TRACE,
            SAMPLE_OBSERVATIONS,
            "proj-1",
        )

        # 3 observations, none are EVENTs, so all 3 become spans
        assert len(span_rows) == 3

        # Check trace row
        assert trace_row["trace_id"] == "trace-abc-123"
        assert trace_row["project_id"] == "proj-1"
        assert trace_row["source"] == "langfuse"
        assert trace_row["status"] == "success"
        assert trace_row["span_count"] == 3
        assert trace_row["service_name"] == "chat-completion"

        # Check trace attributes
        attrs = json.loads(trace_row["attributes"])
        assert attrs["langfuse.trace.name"] == "chat-completion"
        assert attrs["langfuse.user_id"] == "user-42"
        assert attrs["langfuse.session_id"] == "session-99"
        assert attrs["langfuse.tags"] == ["production", "gpt-4"]

    def test_event_observations_are_skipped(self):
        """EVENT observations should not produce span rows."""
        observations = [
            {
                "id": "obs-event",
                "traceId": "t1",
                "type": "EVENT",
                "name": "log-message",
                "startTime": "2026-02-13T10:00:00Z",
            },
            {
                "id": "obs-gen",
                "traceId": "t1",
                "type": "GENERATION",
                "name": "llm-call",
                "startTime": "2026-02-13T10:00:00Z",
                "endTime": "2026-02-13T10:00:01Z",
                "level": "DEFAULT",
            },
        ]
        trace = {"id": "t1", "name": "test", "timestamp": "2026-02-13T10:00:00Z"}
        trace_row, span_rows = _normalize_single_trace(trace, observations, "proj-1")

        assert len(span_rows) == 1  # Only GENERATION, not EVENT
        assert span_rows[0]["span_id"] == "obs-gen"

    def test_error_propagation_to_trace(self):
        """If any span has error status, the trace should be error."""
        observations = [
            {
                "id": "obs-1",
                "traceId": "t1",
                "type": "GENERATION",
                "name": "good",
                "startTime": "2026-02-13T10:00:00Z",
                "endTime": "2026-02-13T10:00:01Z",
                "level": "DEFAULT",
            },
            {
                "id": "obs-2",
                "traceId": "t1",
                "type": "GENERATION",
                "name": "bad",
                "startTime": "2026-02-13T10:00:01Z",
                "endTime": "2026-02-13T10:00:02Z",
                "level": "ERROR",
            },
        ]
        trace = {"id": "t1", "name": "test", "timestamp": "2026-02-13T10:00:00Z"}
        trace_row, _ = _normalize_single_trace(trace, observations, "proj-1")
        assert trace_row["status"] == "error"

    def test_root_span_detection(self):
        """Root span should be the one without a parentObservationId."""
        observations = [
            {
                "id": "child",
                "traceId": "t1",
                "type": "SPAN",
                "name": "child",
                "startTime": "2026-02-13T10:00:00Z",
                "endTime": "2026-02-13T10:00:01Z",
                "level": "DEFAULT",
                "parentObservationId": "root",
            },
            {
                "id": "root",
                "traceId": "t1",
                "type": "SPAN",
                "name": "root",
                "startTime": "2026-02-13T10:00:00Z",
                "endTime": "2026-02-13T10:00:02Z",
                "level": "DEFAULT",
                "parentObservationId": None,
            },
        ]
        trace = {"id": "t1", "name": "test", "timestamp": "2026-02-13T10:00:00Z"}
        trace_row, _ = _normalize_single_trace(trace, observations, "proj-1")
        assert trace_row["root_span_id"] == "root"

    def test_empty_observations(self):
        """Trace with no observations should still produce a trace row."""
        trace = {"id": "t1", "name": "empty", "timestamp": "2026-02-13T10:00:00Z"}
        trace_row, span_rows = _normalize_single_trace(trace, [], "proj-1")

        assert trace_row["trace_id"] == "t1"
        assert trace_row["span_count"] == 0
        assert len(span_rows) == 0

    def test_duration_calculated_from_spans(self):
        """Trace duration should be calculated from span timing bounds."""
        observations = [
            {
                "id": "obs-1",
                "traceId": "t1",
                "type": "SPAN",
                "name": "first",
                "startTime": "2026-02-13T10:00:00.000Z",
                "endTime": "2026-02-13T10:00:00.500Z",
                "level": "DEFAULT",
            },
            {
                "id": "obs-2",
                "traceId": "t1",
                "type": "SPAN",
                "name": "second",
                "startTime": "2026-02-13T10:00:01.000Z",
                "endTime": "2026-02-13T10:00:03.000Z",
                "level": "DEFAULT",
            },
        ]
        trace = {"id": "t1", "name": "test", "timestamp": "2026-02-13T10:00:00.000Z"}
        trace_row, _ = _normalize_single_trace(trace, observations, "proj-1")

        # Duration should span from earliest start to latest end: 3000ms
        assert trace_row["duration_ms"] == pytest.approx(3000, abs=10)


# --- Batch normalization ---


class TestNormalizeLangfuseTraces:
    def test_multiple_traces(self):
        traces = [
            {"id": "t1", "name": "trace-1", "timestamp": "2026-02-13T10:00:00Z"},
            {"id": "t2", "name": "trace-2", "timestamp": "2026-02-13T10:01:00Z"},
        ]
        observations_by_trace = {
            "t1": [
                {
                    "id": "obs-1",
                    "traceId": "t1",
                    "type": "GENERATION",
                    "name": "llm",
                    "startTime": "2026-02-13T10:00:00Z",
                    "endTime": "2026-02-13T10:00:01Z",
                    "level": "DEFAULT",
                },
            ],
            "t2": [
                {
                    "id": "obs-2",
                    "traceId": "t2",
                    "type": "SPAN",
                    "name": "custom",
                    "startTime": "2026-02-13T10:01:00Z",
                    "endTime": "2026-02-13T10:01:02Z",
                    "level": "DEFAULT",
                },
            ],
        }

        trace_rows, span_rows = normalize_langfuse_traces(
            traces, observations_by_trace, "proj-1"
        )

        assert len(trace_rows) == 2
        assert len(span_rows) == 2
        assert trace_rows[0]["trace_id"] == "t1"
        assert trace_rows[1]["trace_id"] == "t2"
        assert all(r["source"] == "langfuse" for r in trace_rows)
        assert all(r["source"] == "langfuse" for r in span_rows)

    def test_trace_without_observations(self):
        traces = [{"id": "t1", "name": "empty", "timestamp": "2026-02-13T10:00:00Z"}]
        observations_by_trace = {}

        trace_rows, span_rows = normalize_langfuse_traces(
            traces, observations_by_trace, "proj-1"
        )

        assert len(trace_rows) == 1
        assert len(span_rows) == 0

    def test_skip_trace_without_id(self):
        traces = [{"name": "no-id"}]
        trace_rows, span_rows = normalize_langfuse_traces(traces, {}, "proj-1")
        assert len(trace_rows) == 0

    def test_all_observation_types(self):
        """Test that all Langfuse observation types are handled correctly."""
        trace = {"id": "t1", "name": "all-types", "timestamp": "2026-02-13T10:00:00Z"}
        observations = []
        for i, (lf_type, prela_type) in enumerate(_OBSERVATION_TYPE_MAP.items()):
            observations.append({
                "id": f"obs-{i}",
                "traceId": "t1",
                "type": lf_type,
                "name": f"obs-{lf_type}",
                "startTime": "2026-02-13T10:00:00Z",
                "endTime": "2026-02-13T10:00:01Z",
                "level": "DEFAULT",
            })

        trace_rows, span_rows = normalize_langfuse_traces(
            [trace], {"t1": observations}, "proj-1"
        )

        # EVENT type should be skipped
        expected_count = sum(1 for v in _OBSERVATION_TYPE_MAP.values() if v is not None)
        assert len(span_rows) == expected_count

        # Verify each span type
        span_types = {s["name"]: s["span_type"] for s in span_rows}
        for lf_type, prela_type in _OBSERVATION_TYPE_MAP.items():
            if prela_type is not None:
                assert span_types[f"obs-{lf_type}"] == prela_type


# --- End-to-end normalization ---


class TestEndToEnd:
    def test_full_langfuse_payload(self):
        """End-to-end test with realistic Langfuse data."""
        trace_rows, span_rows = normalize_langfuse_traces(
            [SAMPLE_LANGFUSE_TRACE],
            {"trace-abc-123": SAMPLE_OBSERVATIONS},
            "my-project",
        )

        assert len(trace_rows) == 1
        assert len(span_rows) == 3

        trace = trace_rows[0]
        assert trace["trace_id"] == "trace-abc-123"
        assert trace["project_id"] == "my-project"
        assert trace["source"] == "langfuse"
        assert trace["status"] == "success"
        assert trace["span_count"] == 3

        # Verify span types
        span_types = {s["span_id"]: s["span_type"] for s in span_rows}
        assert span_types["obs-gen-001"] == "llm"
        assert span_types["obs-span-002"] == "custom"
        assert span_types["obs-tool-003"] == "tool"

        # All spans should have the right project and source
        for span in span_rows:
            assert span["project_id"] == "my-project"
            assert span["source"] == "langfuse"

        # Verify JSON fields are valid JSON
        for span in span_rows:
            json.loads(span["attributes"])
            json.loads(span["events"])
            json.loads(span["replay_snapshot"])
        json.loads(trace["attributes"])

    def test_output_matches_clickhouse_schema(self):
        """Verify output rows have all required ClickHouse columns."""
        trace_rows, span_rows = normalize_langfuse_traces(
            [SAMPLE_LANGFUSE_TRACE],
            {"trace-abc-123": SAMPLE_OBSERVATIONS},
            "proj-1",
        )

        expected_trace_columns = {
            "trace_id", "project_id", "service_name", "started_at",
            "completed_at", "duration_ms", "status", "root_span_id",
            "span_count", "attributes", "source",
        }
        expected_span_columns = {
            "span_id", "trace_id", "project_id", "parent_span_id",
            "name", "span_type", "service_name", "started_at", "ended_at",
            "duration_ms", "status", "attributes", "events",
            "replay_snapshot", "source",
        }

        for trace_row in trace_rows:
            assert set(trace_row.keys()) == expected_trace_columns

        for span_row in span_rows:
            assert set(span_row.keys()) == expected_span_columns
