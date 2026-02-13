"""Tests for eval generator module."""

import sys
import os
import json
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval_generator import (
    EvalGenerator,
    EvalGenerationConfig,
    EvalGenerationResult,
    GeneratedEvalCase,
    TracePattern,
    _row_to_dict,
    _truncate,
    _parse_attrs,
    SPAN_COLUMNS,
    TRACE_COLUMNS,
    VALID_ASSERTION_TYPES,
)

import pytest
import yaml


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


def make_span_dict(
    span_id="span-1",
    trace_id="trace-1",
    project_id="proj-1",
    parent_span_id="",
    name="test-span",
    span_type="custom",
    service_name="test-service",
    started_at="2026-02-13 10:00:00.000000",
    ended_at="2026-02-13 10:00:01.000000",
    duration_ms=1000.0,
    status="success",
    attributes="{}",
    events="[]",
    replay_snapshot="",
    source="native",
    created_at="2026-02-13 10:00:01.000000",
):
    return {
        "span_id": span_id,
        "trace_id": trace_id,
        "project_id": project_id,
        "parent_span_id": parent_span_id,
        "name": name,
        "span_type": span_type,
        "service_name": service_name,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_ms": duration_ms,
        "status": status,
        "attributes": attributes,
        "events": events,
        "replay_snapshot": replay_snapshot,
        "source": source,
        "created_at": created_at,
    }


def make_trace_dict(
    trace_id="trace-1",
    project_id="proj-1",
    service_name="test-service",
    started_at="2026-02-13 10:00:00.000000",
    completed_at="2026-02-13 10:00:05.000000",
    duration_ms=5000.0,
    status="success",
    root_span_id="span-1",
    span_count=3,
    attributes="{}",
    source="native",
    created_at="2026-02-13 10:00:05.000000",
):
    return {
        "trace_id": trace_id,
        "project_id": project_id,
        "service_name": service_name,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_ms": duration_ms,
        "status": status,
        "root_span_id": root_span_id,
        "span_count": span_count,
        "attributes": attributes,
        "source": source,
        "created_at": created_at,
    }


MOCK_LLM_EVAL_RESPONSE = """[
  {
    "name": "test_rate_limit_handling",
    "input": {"query": "What is the capital of France?"},
    "expected": {"contains": ["Paris", "capital"]},
    "assertions": [
      {"type": "contains", "text": "Paris"},
      {"type": "length", "min_length": 10}
    ],
    "tags": ["failure_mode", "rate_limit"]
  },
  {
    "name": "test_graceful_error",
    "input": {"query": "Tell me about Python"},
    "expected": {"contains": ["Python"]},
    "assertions": [
      {"type": "contains", "text": "Python"}
    ],
    "tags": ["qa"]
  }
]"""

MOCK_LLM_CATEGORIZATION_RESPONSE = """[
  {"trace_id": "trace-1", "category": "failure_mode", "subcategory": "rate_limit", "description": "Rate limit hit on OpenAI API"}
]"""


# ---------------------------------------------------------------------------
# Utility Function Tests
# ---------------------------------------------------------------------------


class TestRowToDict:
    def test_basic_conversion(self):
        row = ("s1", "t1", "p1", "", "name", "llm", "svc", "ts1", "ts2", 100.0, "success", "{}", "[]", "", "native", "ts3")
        result = _row_to_dict(row, SPAN_COLUMNS)
        assert result["span_id"] == "s1"
        assert result["trace_id"] == "t1"
        assert result["span_type"] == "llm"

    def test_trace_conversion(self):
        row = ("t1", "p1", "svc", "ts1", "ts2", 5000.0, "error", "s1", 3, "{}", "native", "ts3")
        result = _row_to_dict(row, TRACE_COLUMNS)
        assert result["trace_id"] == "t1"
        assert result["status"] == "error"
        assert result["duration_ms"] == 5000.0


class TestTruncate:
    def test_short_string(self):
        assert _truncate("hello", 100) == "hello"

    def test_long_string(self):
        text = "a" * 600
        result = _truncate(text, 500)
        assert len(result) == 503  # 500 + "..."
        assert result.endswith("...")

    def test_exact_length(self):
        text = "a" * 500
        assert _truncate(text, 500) == text


class TestParseAttrs:
    def test_dict_passthrough(self):
        assert _parse_attrs({"key": "val"}) == {"key": "val"}

    def test_json_string(self):
        assert _parse_attrs('{"key": "val"}') == {"key": "val"}

    def test_empty_string(self):
        assert _parse_attrs("") == {}

    def test_invalid_json(self):
        assert _parse_attrs("not json") == {}

    def test_none(self):
        assert _parse_attrs(None) == {}

    def test_non_dict_json(self):
        assert _parse_attrs("[1, 2, 3]") == {}


# ---------------------------------------------------------------------------
# Extract Trace I/O Tests
# ---------------------------------------------------------------------------


class TestExtractTraceIO:
    @patch("eval_generator.OpenAI")
    def setup_method(self, method, mock_openai):
        self.generator = EvalGenerator(openai_api_key="test-key")

    def test_extract_simple_query(self):
        trace = make_trace_dict()
        spans = [
            make_span_dict(
                span_type="llm",
                attributes=json.dumps({"llm.prompt": "What is 2+2?", "llm.response": "4"}),
            )
        ]
        result = self.generator._extract_trace_io(trace, spans)
        assert result["input"]["query"] == "What is 2+2?"
        assert result["output"]["response"] == "4"

    def test_extract_chat_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        trace = make_trace_dict()
        spans = [
            make_span_dict(
                span_type="llm",
                attributes=json.dumps({
                    "llm.prompt": json.dumps(messages),
                    "llm.response": "Hi there!",
                }),
            )
        ]
        result = self.generator._extract_trace_io(trace, spans)
        assert result["input"]["messages"] == messages
        assert result["output"]["response"] == "Hi there!"

    def test_extract_gen_ai_attributes(self):
        trace = make_trace_dict()
        spans = [
            make_span_dict(
                span_type="llm",
                attributes=json.dumps({
                    "gen_ai.prompt": "Explain Python",
                    "gen_ai.completion": "Python is a programming language",
                }),
            )
        ]
        result = self.generator._extract_trace_io(trace, spans)
        assert result["input"]["query"] == "Explain Python"
        assert "Python" in result["output"]["response"]

    def test_extract_tool_calls(self):
        trace = make_trace_dict()
        spans = [
            make_span_dict(span_type="llm", attributes=json.dumps({"llm.prompt": "search for cats"})),
            make_span_dict(
                span_id="span-tool-1",
                span_type="tool",
                name="web_search",
                attributes=json.dumps({"tool.name": "web_search"}),
            ),
        ]
        result = self.generator._extract_trace_io(trace, spans)
        assert len(result["output"]["tool_calls"]) == 1
        assert result["output"]["tool_calls"][0]["name"] == "web_search"

    def test_extract_no_llm_spans(self):
        trace = make_trace_dict(service_name="my-agent")
        spans = [make_span_dict(span_type="custom")]
        result = self.generator._extract_trace_io(trace, spans)
        assert "query" in result["input"]  # fallback to service name

    def test_extract_error_message_from_events(self):
        trace = make_trace_dict(status="error")
        events = [{"name": "exception", "attributes": {"exception.message": "Rate limit exceeded"}}]
        spans = [
            make_span_dict(
                status="error",
                events=json.dumps(events),
            )
        ]
        result = self.generator._extract_trace_io(trace, spans)
        assert result["error_message"] == "Rate limit exceeded"

    def test_extract_error_message_from_attributes(self):
        trace = make_trace_dict(status="error")
        spans = [
            make_span_dict(
                status="error",
                attributes=json.dumps({"error.message": "Connection refused"}),
            )
        ]
        result = self.generator._extract_trace_io(trace, spans)
        assert result["error_message"] == "Connection refused"

    def test_extract_empty_spans(self):
        trace = make_trace_dict(service_name="svc")
        result = self.generator._extract_trace_io(trace, [])
        assert result["input"]  # should have fallback
        assert result["output"] == {}

    def test_truncates_long_prompt(self):
        long_prompt = "x" * 1000
        trace = make_trace_dict()
        spans = [
            make_span_dict(
                span_type="llm",
                attributes=json.dumps({"llm.prompt": long_prompt}),
            )
        ]
        result = self.generator._extract_trace_io(trace, spans)
        assert len(result["input"]["query"]) <= 503  # MAX_TEXT_LENGTH + "..."


# ---------------------------------------------------------------------------
# Categorize Traces Tests
# ---------------------------------------------------------------------------


class TestCategorizeTraces:
    @patch("eval_generator.OpenAI")
    def setup_method(self, method, mock_openai):
        self.generator = EvalGenerator(openai_api_key="test-key")
        self.config = EvalGenerationConfig(project_id="proj-1")

    def _make_item(self, trace_id="t1", status="success", duration_ms=100.0, error_message=None, tool_spans=None):
        trace = make_trace_dict(trace_id=trace_id, status=status, duration_ms=duration_ms)
        io_data = {
            "input": {"query": "test"},
            "output": {"response": "test response"},
            "llm_spans": [],
            "tool_spans": tool_spans or [],
            "error_message": error_message,
        }
        return {"trace": trace, "spans": tool_spans or [], "io": io_data}

    def test_error_trace_categorized_as_failure(self):
        items = [self._make_item(status="error", error_message="Something went wrong")]
        patterns = self.generator._categorize_traces(items, self.config)
        assert len(patterns) >= 1
        failure_patterns = [p for p in patterns if p.category == "failure_mode"]
        assert len(failure_patterns) == 1

    def test_rate_limit_detection(self):
        items = [self._make_item(status="error", error_message="Rate limit exceeded (429)")]
        patterns = self.generator._categorize_traces(items, self.config)
        rl = [p for p in patterns if p.subcategory == "rate_limit"]
        assert len(rl) == 1

    def test_timeout_detection(self):
        items = [self._make_item(status="error", error_message="Request timed out")]
        patterns = self.generator._categorize_traces(items, self.config)
        to = [p for p in patterns if p.subcategory == "timeout"]
        assert len(to) == 1

    def test_auth_error_detection(self):
        items = [self._make_item(status="error", error_message="401 Unauthorized")]
        patterns = self.generator._categorize_traces(items, self.config)
        auth = [p for p in patterns if p.subcategory == "auth_error"]
        assert len(auth) == 1

    def test_tool_error_detection(self):
        tool_span = make_span_dict(span_type="tool", status="error")
        items = [self._make_item(status="error", error_message="tool failed", tool_spans=[tool_span])]
        patterns = self.generator._categorize_traces(items, self.config)
        tool_err = [p for p in patterns if p.subcategory == "tool_error"]
        assert len(tool_err) == 1

    def test_slow_trace_as_edge_case(self):
        # Create multiple traces so p95 is meaningful
        items = [
            self._make_item(trace_id=f"t{i}", duration_ms=100.0)
            for i in range(20)
        ]
        # Add one very slow trace
        items.append(self._make_item(trace_id="t-slow", duration_ms=50000.0))
        patterns = self.generator._categorize_traces(items, self.config)
        slow = [p for p in patterns if p.subcategory == "slow_response"]
        assert len(slow) == 1

    def test_success_trace_as_positive_example(self):
        items = [self._make_item(trace_id=f"t{i}", status="success", duration_ms=100.0) for i in range(5)]
        patterns = self.generator._categorize_traces(items, self.config)
        positive = [p for p in patterns if p.category == "positive_example"]
        assert len(positive) >= 1

    def test_disable_failure_modes(self):
        config = EvalGenerationConfig(project_id="proj-1", include_failure_modes=False)
        items = [self._make_item(status="error", error_message="error")]
        patterns = self.generator._categorize_traces(items, config)
        failure = [p for p in patterns if p.category == "failure_mode"]
        assert len(failure) == 0

    def test_disable_edge_cases(self):
        config = EvalGenerationConfig(project_id="proj-1", include_edge_cases=False)
        items = [self._make_item(trace_id=f"t{i}", duration_ms=100.0) for i in range(20)]
        items.append(self._make_item(trace_id="t-slow", duration_ms=50000.0))
        patterns = self.generator._categorize_traces(items, config)
        edge = [p for p in patterns if p.category == "edge_case"]
        assert len(edge) == 0

    def test_disable_positive_examples(self):
        config = EvalGenerationConfig(project_id="proj-1", include_positive_examples=False)
        items = [self._make_item(status="success")]
        patterns = self.generator._categorize_traces(items, config)
        positive = [p for p in patterns if p.category == "positive_example"]
        assert len(positive) == 0

    def test_empty_traces(self):
        patterns = self.generator._categorize_traces([], self.config)
        assert patterns == []

    def test_mixed_traces(self):
        items = [
            self._make_item(trace_id="t1", status="error", error_message="429 rate limit"),
            self._make_item(trace_id="t2", status="success", duration_ms=50.0),
            self._make_item(trace_id="t3", status="success", duration_ms=100.0),
        ]
        patterns = self.generator._categorize_traces(items, self.config)
        categories = {p.category for p in patterns}
        assert "failure_mode" in categories
        assert "positive_example" in categories


# ---------------------------------------------------------------------------
# Parse Generated Cases Tests
# ---------------------------------------------------------------------------


class TestParseGeneratedCases:
    @patch("eval_generator.OpenAI")
    def setup_method(self, method, mock_openai):
        self.generator = EvalGenerator(openai_api_key="test-key")
        self.pattern = TracePattern(
            pattern_id="p1",
            category="failure_mode",
            subcategory="rate_limit",
            description="Rate limit errors",
            trace_ids=["t1"],
            representative_trace_id="t1",
        )
        self.traces = [{"trace": make_trace_dict(), "spans": [], "io": {"input": {"query": "test"}, "output": {}, "error_message": None}}]

    def test_parse_valid_response(self):
        cases = self.generator._parse_generated_cases(MOCK_LLM_EVAL_RESPONSE, self.pattern, self.traces)
        assert len(cases) == 2
        assert cases[0].case_name == "test_rate_limit_handling"
        assert cases[0].input_query == "What is the capital of France?"
        assert cases[0].expected_contains == ["Paris", "capital"]

    def test_parse_response_with_markdown_fencing(self):
        response = f"```json\n{MOCK_LLM_EVAL_RESPONSE}\n```"
        cases = self.generator._parse_generated_cases(response, self.pattern, self.traces)
        assert len(cases) == 2

    def test_parse_invalid_json(self):
        cases = self.generator._parse_generated_cases("not json at all", self.pattern, self.traces)
        # Should fall back to heuristic cases
        assert all("fallback" in c.case_id for c in cases)

    def test_parse_empty_array(self):
        cases = self.generator._parse_generated_cases("[]", self.pattern, self.traces)
        assert cases == []

    def test_filters_invalid_assertion_types(self):
        response = json.dumps([{
            "name": "test",
            "input": {"query": "hi"},
            "expected": {"contains": ["ok"]},
            "assertions": [
                {"type": "contains", "text": "ok"},
                {"type": "invalid_type", "value": "bad"},
            ],
            "tags": [],
        }])
        cases = self.generator._parse_generated_cases(response, self.pattern, self.traces)
        assert len(cases) == 1
        # Only the valid assertion should remain
        assertion_types = {a["type"] for a in cases[0].assertions}
        assert assertion_types.issubset(VALID_ASSERTION_TYPES)

    def test_adds_fallback_assertion_when_no_expected(self):
        response = json.dumps([{
            "name": "test",
            "input": {"query": "hi"},
            "expected": {},
            "assertions": [],
            "tags": [],
        }])
        cases = self.generator._parse_generated_cases(response, self.pattern, self.traces)
        assert len(cases) == 1
        assert any(a["type"] == "length" for a in cases[0].assertions)

    def test_tags_include_pattern_info(self):
        cases = self.generator._parse_generated_cases(MOCK_LLM_EVAL_RESPONSE, self.pattern, self.traces)
        assert "failure_mode" in cases[0].tags
        assert "rate_limit" in cases[0].tags


# ---------------------------------------------------------------------------
# Generate Fallback Cases Tests
# ---------------------------------------------------------------------------


class TestGenerateFallbackCases:
    @patch("eval_generator.OpenAI")
    def setup_method(self, method, mock_openai):
        self.generator = EvalGenerator(openai_api_key="test-key")
        self.pattern = TracePattern(
            pattern_id="p1",
            category="positive_example",
            subcategory="fast_success",
            description="Fast successes",
            trace_ids=["t1"],
            representative_trace_id="t1",
        )

    def test_generates_cases_from_traces(self):
        traces = [{
            "trace": make_trace_dict(),
            "spans": [],
            "io": {
                "input": {"query": "What is Python?"},
                "output": {"response": "Python is a high-level programming language."},
                "error_message": None,
            },
        }]
        cases = self.generator._generate_fallback_cases(self.pattern, traces)
        assert len(cases) == 1
        assert "fallback" in cases[0].case_id
        assert cases[0].input_query == "What is Python?"

    def test_skips_traces_without_input(self):
        traces = [{
            "trace": make_trace_dict(),
            "spans": [],
            "io": {"input": {}, "output": {}, "error_message": None},
        }]
        cases = self.generator._generate_fallback_cases(self.pattern, traces)
        assert len(cases) == 0

    def test_caps_at_3_cases(self):
        traces = [{
            "trace": make_trace_dict(trace_id=f"t{i}"),
            "spans": [],
            "io": {"input": {"query": f"query {i}"}, "output": {"response": "resp"}, "error_message": None},
        } for i in range(10)]
        cases = self.generator._generate_fallback_cases(self.pattern, traces)
        assert len(cases) <= 3


# ---------------------------------------------------------------------------
# Assemble Suite YAML Tests
# ---------------------------------------------------------------------------


class TestAssembleSuite:
    @patch("eval_generator.OpenAI")
    def setup_method(self, method, mock_openai):
        self.generator = EvalGenerator(openai_api_key="test-key")
        self.config = EvalGenerationConfig(project_id="proj-1")

    def test_yaml_is_parseable(self):
        cases = [GeneratedEvalCase(
            case_id="test_1",
            case_name="Test case 1",
            input_query="What is 2+2?",
            expected_contains=["4"],
            assertions=[{"type": "contains", "text": "4"}],
            tags=["math"],
            source_trace_id="t1",
            source_pattern="positive_example/fast_success",
        )]
        yaml_str = self.generator._assemble_suite(cases, "Test Suite", self.config)
        parsed = yaml.safe_load(yaml_str)
        assert parsed["name"] == "Test Suite"
        assert len(parsed["cases"]) == 1

    def test_suite_structure_matches_sdk(self):
        """Verify the output dict structure matches EvalSuite.from_dict() expectations."""
        cases = [GeneratedEvalCase(
            case_id="test_1",
            case_name="Test case 1",
            input_query="Hello",
            expected_contains=["Hi"],
            assertions=[{"type": "contains", "text": "Hi"}],
        )]
        yaml_str = self.generator._assemble_suite(cases, "Suite", self.config)
        parsed = yaml.safe_load(yaml_str)

        # EvalSuite requires "name"
        assert "name" in parsed

        # Each case must have "id", "name", "input"
        case = parsed["cases"][0]
        assert "id" in case
        assert "name" in case
        assert "input" in case
        assert "query" in case["input"] or "messages" in case["input"]

        # Expected or assertions must exist
        assert "expected" in case or "assertions" in case

    def test_empty_cases(self):
        yaml_str = self.generator._assemble_suite([], "Empty Suite", self.config)
        parsed = yaml.safe_load(yaml_str)
        assert parsed["name"] == "Empty Suite"
        assert parsed["cases"] == []

    def test_includes_metadata(self):
        yaml_str = self.generator._assemble_suite([], "Suite", self.config)
        parsed = yaml.safe_load(yaml_str)
        assert "metadata" in parsed
        assert "generated_at" in parsed["metadata"]
        assert parsed["metadata"]["project_id"] == "proj-1"

    def test_includes_default_assertions(self):
        yaml_str = self.generator._assemble_suite([], "Suite", self.config)
        parsed = yaml.safe_load(yaml_str)
        assert "default_assertions" in parsed
        assert parsed["default_assertions"][0]["type"] == "length"

    def test_input_fallback_when_empty(self):
        cases = [GeneratedEvalCase(
            case_id="test_1",
            case_name="Test",
            assertions=[{"type": "length", "min_length": 1}],
        )]
        yaml_str = self.generator._assemble_suite(cases, "Suite", self.config)
        parsed = yaml.safe_load(yaml_str)
        # Should have fallback input
        assert "query" in parsed["cases"][0]["input"]

    def test_all_fields_serialized(self):
        cases = [GeneratedEvalCase(
            case_id="test_1",
            case_name="Full case",
            input_query="What?",
            input_context={"key": "val"},
            expected_output="Answer",
            expected_contains=["ans"],
            expected_not_contains=["bad"],
            expected_tool_calls=[{"name": "search"}],
            assertions=[{"type": "contains", "text": "ans"}],
            tags=["tag1"],
            source_trace_id="t1",
            source_pattern="positive_example/fast_success",
        )]
        yaml_str = self.generator._assemble_suite(cases, "Suite", self.config)
        parsed = yaml.safe_load(yaml_str)
        case = parsed["cases"][0]
        assert case["input"]["query"] == "What?"
        assert case["input"]["context"] == {"key": "val"}
        assert case["expected"]["output"] == "Answer"
        assert case["expected"]["contains"] == ["ans"]
        assert case["expected"]["not_contains"] == ["bad"]
        assert case["expected"]["tool_calls"] == [{"name": "search"}]
        assert case["assertions"][0]["type"] == "contains"
        assert case["tags"] == ["tag1"]
        assert case["metadata"]["source_trace_id"] == "t1"


# ---------------------------------------------------------------------------
# Generate Cases for Pattern (LLM Integration) Tests
# ---------------------------------------------------------------------------


class TestGenerateCasesForPattern:
    @patch("eval_generator.OpenAI")
    def setup_method(self, method, mock_openai):
        self.mock_client = MagicMock()
        mock_openai.return_value = self.mock_client
        self.generator = EvalGenerator(openai_api_key="test-key")
        self.config = EvalGenerationConfig(project_id="proj-1")
        self.pattern = TracePattern(
            pattern_id="p1",
            category="failure_mode",
            subcategory="rate_limit",
            description="Rate limit errors",
            trace_ids=["t1"],
            representative_trace_id="t1",
        )
        self.traces = [{
            "trace": make_trace_dict(status="error"),
            "spans": [],
            "io": {
                "input": {"query": "What is the capital?"},
                "output": {},
                "error_message": "Rate limit 429",
            },
        }]

    def test_calls_openai(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=MOCK_LLM_EVAL_RESPONSE))]
        self.mock_client.chat.completions.create.return_value = mock_response

        cases = self.generator._generate_cases_for_pattern(self.pattern, self.traces, self.config)
        assert len(cases) == 2
        self.mock_client.chat.completions.create.assert_called_once()

    def test_openai_failure_falls_back(self):
        self.mock_client.chat.completions.create.side_effect = Exception("API error")
        cases = self.generator._generate_cases_for_pattern(self.pattern, self.traces, self.config)
        # Should get fallback cases
        assert len(cases) >= 1
        assert all("fallback" in c.case_id for c in cases)

    def test_empty_traces_returns_empty(self):
        cases = self.generator._generate_cases_for_pattern(self.pattern, [], self.config)
        assert cases == []


# ---------------------------------------------------------------------------
# End-to-End Pipeline Tests
# ---------------------------------------------------------------------------


class TestGenerateEndToEnd:
    @patch("eval_generator.OpenAI")
    def setup_method(self, method, mock_openai):
        self.mock_client = MagicMock()
        mock_openai.return_value = self.mock_client
        self.generator = EvalGenerator(openai_api_key="test-key")

    def _mock_clickhouse(self, traces=None, spans=None):
        """Create a mock ClickHouse client with canned query results."""
        client = MagicMock()

        trace_rows = []
        if traces:
            for t in traces:
                row = tuple(t.get(col, "") for col in TRACE_COLUMNS)
                trace_rows.append(row)

        span_rows = []
        if spans:
            for s in spans:
                row = tuple(s.get(col, "") for col in SPAN_COLUMNS)
                span_rows.append(row)

        def query_side_effect(query_str, parameters=None):
            result = MagicMock()
            if "FROM traces" in query_str:
                result.result_rows = trace_rows
            elif "FROM spans" in query_str:
                result.result_rows = span_rows
            else:
                result.result_rows = []
            return result

        client.query.side_effect = query_side_effect
        return client

    def test_full_pipeline(self):
        """End-to-end: mock ClickHouse + mock OpenAI → valid YAML."""
        traces = [
            make_trace_dict(trace_id="t1", status="error", duration_ms=5000.0),
            make_trace_dict(trace_id="t2", status="success", duration_ms=100.0),
        ]
        spans = [
            make_span_dict(
                trace_id="t1", span_type="llm",
                attributes=json.dumps({"llm.prompt": "What is AI?", "llm.response": "AI is..."}),
            ),
            make_span_dict(
                trace_id="t2", span_type="llm",
                attributes=json.dumps({"llm.prompt": "Hello", "llm.response": "Hi there"}),
            ),
        ]
        client = self._mock_clickhouse(traces, spans)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=MOCK_LLM_EVAL_RESPONSE))]
        self.mock_client.chat.completions.create.return_value = mock_response

        config = EvalGenerationConfig(project_id="proj-1", max_cases=10)
        result = self.generator.generate("proj-1", config, clickhouse_client=client)

        assert result.status == "completed"
        assert result.traces_analyzed == 2
        assert result.cases_generated > 0
        assert result.suite_yaml is not None

        # Verify YAML is valid
        parsed = yaml.safe_load(result.suite_yaml)
        assert "cases" in parsed
        assert parsed["name"] is not None

    def test_no_traces_returns_empty_suite(self):
        client = self._mock_clickhouse(traces=[], spans=[])
        config = EvalGenerationConfig(project_id="proj-1")
        result = self.generator.generate("proj-1", config, clickhouse_client=client)

        assert result.status == "completed"
        assert result.traces_analyzed == 0
        assert result.cases_generated == 0
        assert result.suite_yaml is not None

        parsed = yaml.safe_load(result.suite_yaml)
        assert parsed["cases"] == []

    def test_clickhouse_failure(self):
        client = MagicMock()
        client.query.side_effect = Exception("ClickHouse connection failed")

        config = EvalGenerationConfig(project_id="proj-1")
        result = self.generator.generate("proj-1", config, clickhouse_client=client)

        assert result.status == "failed"
        assert "ClickHouse" in result.error

    def test_respects_max_cases(self):
        traces = [
            make_trace_dict(trace_id=f"t{i}", status="error", duration_ms=100.0)
            for i in range(20)
        ]
        spans = [
            make_span_dict(
                trace_id=f"t{i}", span_type="llm",
                attributes=json.dumps({"llm.prompt": f"query {i}", "llm.response": f"answer {i}"}),
            )
            for i in range(20)
        ]
        client = self._mock_clickhouse(traces, spans)

        # Return many cases from LLM
        big_response = json.dumps([
            {"name": f"case_{i}", "input": {"query": f"q{i}"}, "expected": {"contains": [f"a{i}"]}, "assertions": [], "tags": []}
            for i in range(30)
        ])
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=big_response))]
        self.mock_client.chat.completions.create.return_value = mock_response

        config = EvalGenerationConfig(project_id="proj-1", max_cases=5)
        result = self.generator.generate("proj-1", config, clickhouse_client=client)

        assert result.status == "completed"
        assert result.cases_generated <= 5

    def test_service_name_filter(self):
        traces = [
            make_trace_dict(trace_id="t1", service_name="agent-a"),
            make_trace_dict(trace_id="t2", service_name="agent-b"),
        ]
        spans = []
        client = self._mock_clickhouse(traces, spans)

        config = EvalGenerationConfig(project_id="proj-1", agent_name_filter="agent-a")
        result = self.generator.generate("proj-1", config, clickhouse_client=client)

        # Should only analyze agent-a traces
        assert result.traces_analyzed <= 1


# ---------------------------------------------------------------------------
# Data Structure Tests
# ---------------------------------------------------------------------------


class TestDataStructures:
    def test_eval_generation_result_to_dict(self):
        result = EvalGenerationResult(
            generation_id="g1",
            project_id="p1",
            status="completed",
            suite_name="Test",
            suite_yaml="name: Test",
            cases_generated=5,
        )
        d = result.to_dict()
        assert d["generation_id"] == "g1"
        assert d["status"] == "completed"
        assert d["cases_generated"] == 5

    def test_trace_pattern_label(self):
        p = TracePattern(
            pattern_id="p1",
            category="failure_mode",
            subcategory="rate_limit",
            description="test",
            trace_ids=["t1"],
            representative_trace_id="t1",
        )
        assert p.label() == "failure_mode/rate_limit"

    def test_eval_generation_config_defaults(self):
        config = EvalGenerationConfig(project_id="p1")
        assert config.time_window_hours == 168
        assert config.max_traces == 500
        assert config.max_cases == 50
        assert config.include_failure_modes is True
        assert config.include_edge_cases is True
        assert config.include_positive_examples is True

    def test_generated_eval_case_defaults(self):
        case = GeneratedEvalCase(case_id="c1", case_name="test")
        assert case.assertions == []
        assert case.tags == []
        assert case.input_query is None


# ---------------------------------------------------------------------------
# Init Tests
# ---------------------------------------------------------------------------


class TestEvalGeneratorInit:
    @patch("eval_generator.OpenAI")
    def test_init_with_key(self, mock_openai):
        gen = EvalGenerator(openai_api_key="sk-test")
        mock_openai.assert_called_once_with(api_key="sk-test")

    @patch("eval_generator.OpenAI")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env"})
    def test_init_from_env(self, mock_openai):
        gen = EvalGenerator()
        mock_openai.assert_called_once_with(api_key="sk-env")

    @patch("eval_generator.OpenAI")
    @patch.dict(os.environ, {}, clear=True)
    def test_init_missing_key(self, mock_openai):
        # Remove the key if set
        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(ValueError, match="OpenAI API key required"):
            EvalGenerator()
