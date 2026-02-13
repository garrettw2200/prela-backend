"""Tests for debug agent module."""

import sys
import os
import json
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from debug_agent import (
    DebugAgent,
    DebugAnalysis,
    FailureChainEntry,
    TimelineEntry,
    _row_to_dict,
    SPAN_COLUMNS,
    TRACE_COLUMNS,
)

import pytest


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
    status="error",
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


MOCK_LLM_RESPONSE = """ROOT_CAUSE: The LLM API call failed with a rate limit error (429) after processing the first prompt.

EXPLANATION: The agent started by sending a prompt to GPT-4, but the OpenAI API returned a 429 rate limit error. This caused the downstream tool call to be skipped entirely, resulting in an incomplete response to the user. The root cause is exceeding the API's requests-per-minute quota.

FIX_SUGGESTIONS:
1. Add exponential backoff retry logic for 429 errors
2. Switch to a model with higher rate limits (e.g., GPT-4o-mini)
3. Implement request queuing to stay within rate limits

CONFIDENCE: 0.85"""


MOCK_SUCCESS_RESPONSE = """ROOT_CAUSE: The trace completed successfully with no errors.

EXPLANATION: All spans executed correctly. The LLM call used GPT-4o-mini and returned a response in 1.2 seconds. The tool call completed successfully. Overall execution was efficient.

FIX_SUGGESTIONS:
1. Consider caching repeated prompts to reduce latency
2. Monitor cost — GPT-4o-mini is already cost-efficient

CONFIDENCE: 0.9"""


# ---------------------------------------------------------------------------
# _row_to_dict Tests
# ---------------------------------------------------------------------------


class TestRowToDict:
    def test_basic_conversion(self):
        row = ("span-1", "trace-1", "proj-1")
        columns = ["span_id", "trace_id", "project_id"]
        result = _row_to_dict(row, columns)
        assert result == {"span_id": "span-1", "trace_id": "trace-1", "project_id": "proj-1"}

    def test_full_span_row(self):
        row = tuple(make_span_dict().values())
        result = _row_to_dict(row, SPAN_COLUMNS)
        assert result["span_id"] == "span-1"
        assert result["status"] == "success"

    def test_full_trace_row(self):
        row = tuple(make_trace_dict().values())
        result = _row_to_dict(row, TRACE_COLUMNS)
        assert result["trace_id"] == "trace-1"
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Timeline Tests
# ---------------------------------------------------------------------------


class TestBuildTimeline:
    @patch("debug_agent.OpenAI")
    def test_basic_timeline(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        spans = [
            make_span_dict(span_id="s1", name="llm-call", span_type="llm", duration_ms=500),
            make_span_dict(span_id="s2", name="tool-call", span_type="tool", duration_ms=200),
        ]
        timeline = agent._build_timeline(spans)
        assert len(timeline) == 2
        assert timeline[0].span_id == "s1"
        assert timeline[0].name == "llm-call"
        assert timeline[0].span_type == "llm"
        assert timeline[0].duration_ms == 500.0

    @patch("debug_agent.OpenAI")
    def test_timeline_with_error(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        spans = [
            make_span_dict(
                span_id="s1",
                name="failed-call",
                status="error",
                events=json.dumps([{
                    "name": "exception",
                    "attributes": {"exception.message": "Rate limit exceeded"},
                }]),
            ),
        ]
        timeline = agent._build_timeline(spans)
        assert len(timeline) == 1
        assert timeline[0].status == "error"
        assert timeline[0].error_message == "Rate limit exceeded"

    @patch("debug_agent.OpenAI")
    def test_timeline_empty_spans(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        timeline = agent._build_timeline([])
        assert timeline == []


# ---------------------------------------------------------------------------
# Failure Chain Tests
# ---------------------------------------------------------------------------


class TestIdentifyFailureChain:
    @patch("debug_agent.OpenAI")
    def test_no_errors(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        spans = [
            make_span_dict(status="success"),
        ]
        chain = agent._identify_failure_chain(spans)
        assert chain == []

    @patch("debug_agent.OpenAI")
    def test_single_error(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        spans = [
            make_span_dict(span_id="s1", status="success"),
            make_span_dict(
                span_id="s2",
                name="bad-call",
                status="error",
                attributes=json.dumps({"error.message": "Connection refused"}),
            ),
        ]
        chain = agent._identify_failure_chain(spans)
        assert len(chain) == 1
        assert chain[0].is_root_cause is True
        assert chain[0].name == "bad-call"
        assert "Connection refused" in chain[0].error_message

    @patch("debug_agent.OpenAI")
    def test_cascading_errors(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        spans = [
            make_span_dict(span_id="s1", name="root-error", status="error",
                           attributes=json.dumps({"error.message": "API down"})),
            make_span_dict(span_id="s2", name="cascade", status="error",
                           attributes=json.dumps({"error.message": "Upstream failed"})),
        ]
        chain = agent._identify_failure_chain(spans)
        assert len(chain) == 2
        assert chain[0].is_root_cause is True
        assert chain[1].is_root_cause is False


# ---------------------------------------------------------------------------
# Error Message Extraction Tests
# ---------------------------------------------------------------------------


class TestExtractErrorMessage:
    @patch("debug_agent.OpenAI")
    def test_from_exception_event(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        span = make_span_dict(
            events=json.dumps([{
                "name": "exception",
                "attributes": {"exception.message": "API timeout"},
            }]),
        )
        msg = agent._extract_error_message(span)
        assert msg == "API timeout"

    @patch("debug_agent.OpenAI")
    def test_from_attributes_error_message(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        span = make_span_dict(
            attributes=json.dumps({"error.message": "Not found"}),
        )
        msg = agent._extract_error_message(span)
        assert msg == "Not found"

    @patch("debug_agent.OpenAI")
    def test_from_attributes_exception_message(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        span = make_span_dict(
            attributes=json.dumps({"exception.message": "Validation error"}),
        )
        msg = agent._extract_error_message(span)
        assert msg == "Validation error"

    @patch("debug_agent.OpenAI")
    def test_no_error_message(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        span = make_span_dict()
        msg = agent._extract_error_message(span)
        assert msg is None

    @patch("debug_agent.OpenAI")
    def test_invalid_events_json(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        span = make_span_dict(events="not json")
        msg = agent._extract_error_message(span)
        assert msg is None

    @patch("debug_agent.OpenAI")
    def test_events_already_parsed(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        span = make_span_dict()
        span["events"] = [{"name": "exception", "attributes": {"exception.message": "Parsed"}}]
        msg = agent._extract_error_message(span)
        assert msg == "Parsed"


# ---------------------------------------------------------------------------
# Context Extraction Tests
# ---------------------------------------------------------------------------


class TestExtractContextDetails:
    @patch("debug_agent.OpenAI")
    def test_llm_span_context(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        spans = [
            make_span_dict(
                name="chat-completion",
                span_type="llm",
                attributes=json.dumps({
                    "llm.model": "gpt-4o-mini",
                    "llm.prompt": "Hello, how are you?",
                    "llm.response": "I'm doing great!",
                    "llm.total_tokens": 42,
                }),
            ),
        ]
        details = agent._extract_context_details(spans)
        assert "gpt-4o-mini" in details
        assert "Hello" in details
        assert "42" in details

    @patch("debug_agent.OpenAI")
    def test_tool_span_context(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        spans = [
            make_span_dict(
                name="search",
                span_type="tool",
                attributes=json.dumps({"tool.name": "web_search", "tool.input": "query text"}),
            ),
        ]
        details = agent._extract_context_details(spans)
        assert "web_search" in details
        assert "query text" in details

    @patch("debug_agent.OpenAI")
    def test_retrieval_span_context(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        spans = [
            make_span_dict(
                name="vector-search",
                span_type="retrieval",
                attributes=json.dumps({"retrieval.query": "find documents about AI"}),
            ),
        ]
        details = agent._extract_context_details(spans)
        assert "find documents about AI" in details

    @patch("debug_agent.OpenAI")
    def test_no_attributes(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        spans = [make_span_dict(span_type="custom", attributes="{}")]
        details = agent._extract_context_details(spans)
        assert "No detailed span attributes available" in details

    @patch("debug_agent.OpenAI")
    def test_truncation(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        long_prompt = "x" * 500
        spans = [
            make_span_dict(
                span_type="llm",
                attributes=json.dumps({"llm.model": "gpt-4", "llm.prompt": long_prompt}),
            ),
        ]
        details = agent._extract_context_details(spans)
        assert "..." in details
        assert len(details) < len(long_prompt) + 200


# ---------------------------------------------------------------------------
# Prompt Building Tests
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    @patch("debug_agent.OpenAI")
    def test_prompt_contains_trace_info(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        trace = make_trace_dict()
        spans = [make_span_dict()]
        timeline = agent._build_timeline(spans)
        failure_chain = agent._identify_failure_chain(spans)
        prompt = agent._build_prompt(trace, spans, timeline, failure_chain)
        assert "trace-1" in prompt
        assert "test-service" in prompt
        assert "error" in prompt.lower()

    @patch("debug_agent.OpenAI")
    def test_prompt_contains_instructions(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        trace = make_trace_dict()
        spans = [make_span_dict()]
        timeline = agent._build_timeline(spans)
        failure_chain = agent._identify_failure_chain(spans)
        prompt = agent._build_prompt(trace, spans, timeline, failure_chain)
        assert "ROOT_CAUSE:" in prompt
        assert "EXPLANATION:" in prompt
        assert "FIX_SUGGESTIONS:" in prompt
        assert "CONFIDENCE:" in prompt


# ---------------------------------------------------------------------------
# Response Parsing Tests
# ---------------------------------------------------------------------------


class TestParseResponse:
    @patch("debug_agent.OpenAI")
    def test_full_response(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        analysis = agent._parse_response(
            MOCK_LLM_RESPONSE,
            trace_id="trace-1",
            timeline=[],
            failure_chain=[],
        )
        assert "rate limit" in analysis.root_cause.lower()
        assert "429" in analysis.explanation
        assert len(analysis.fix_suggestions) == 3
        assert "backoff" in analysis.fix_suggestions[0].lower()
        assert analysis.confidence_score == 0.85

    @patch("debug_agent.OpenAI")
    def test_success_response(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        analysis = agent._parse_response(
            MOCK_SUCCESS_RESPONSE,
            trace_id="trace-1",
            timeline=[],
            failure_chain=[],
        )
        assert "successfully" in analysis.root_cause.lower()
        assert len(analysis.fix_suggestions) == 2
        assert analysis.confidence_score == 0.9

    @patch("debug_agent.OpenAI")
    def test_malformed_response(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        analysis = agent._parse_response(
            "This is not a properly formatted response.",
            trace_id="trace-1",
            timeline=[],
            failure_chain=[],
        )
        # Should fall back to defaults
        assert analysis.trace_id == "trace-1"
        assert analysis.confidence_score == 0.5

    @patch("debug_agent.OpenAI")
    def test_confidence_clamping(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        analysis = agent._parse_response(
            "ROOT_CAUSE: test\nEXPLANATION: test\nFIX_SUGGESTIONS:\n1. fix\nCONFIDENCE: 1.5",
            trace_id="trace-1",
            timeline=[],
            failure_chain=[],
        )
        assert analysis.confidence_score == 1.0

    @patch("debug_agent.OpenAI")
    def test_confidence_negative(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        analysis = agent._parse_response(
            "ROOT_CAUSE: test\nEXPLANATION: test\nFIX_SUGGESTIONS:\n1. fix\nCONFIDENCE: -0.5",
            trace_id="trace-1",
            timeline=[],
            failure_chain=[],
        )
        assert analysis.confidence_score == 0.0

    @patch("debug_agent.OpenAI")
    def test_invalid_confidence(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        analysis = agent._parse_response(
            "ROOT_CAUSE: test\nEXPLANATION: test\nFIX_SUGGESTIONS:\n1. fix\nCONFIDENCE: high",
            trace_id="trace-1",
            timeline=[],
            failure_chain=[],
        )
        assert analysis.confidence_score == 0.5  # default


# ---------------------------------------------------------------------------
# End-to-End Tests (with mocked OpenAI)
# ---------------------------------------------------------------------------


class TestAnalyzeTrace:
    @patch("debug_agent.OpenAI")
    def test_error_trace_analysis(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = MOCK_LLM_RESPONSE
        mock_client.chat.completions.create.return_value = mock_response

        agent = DebugAgent(openai_api_key="test-key")
        trace = make_trace_dict(status="error")
        spans = [
            make_span_dict(span_id="s1", name="llm-call", span_type="llm", status="success",
                           attributes=json.dumps({"llm.model": "gpt-4"})),
            make_span_dict(span_id="s2", name="api-call", status="error",
                           attributes=json.dumps({"error.message": "Rate limit exceeded"})),
        ]
        analysis = agent.analyze_trace(trace, spans)

        assert isinstance(analysis, DebugAnalysis)
        assert analysis.trace_id == "trace-1"
        assert "rate limit" in analysis.root_cause.lower()
        assert len(analysis.execution_timeline) == 2
        assert len(analysis.failure_chain) == 1
        assert analysis.failure_chain[0].is_root_cause is True
        assert analysis.confidence_score == 0.85

    @patch("debug_agent.OpenAI")
    def test_success_trace_analysis(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = MOCK_SUCCESS_RESPONSE
        mock_client.chat.completions.create.return_value = mock_response

        agent = DebugAgent(openai_api_key="test-key")
        trace = make_trace_dict(status="success")
        spans = [
            make_span_dict(span_id="s1", name="llm-call", span_type="llm", status="success"),
        ]
        analysis = agent.analyze_trace(trace, spans)

        assert analysis.trace_id == "trace-1"
        assert "successfully" in analysis.root_cause.lower()
        assert len(analysis.failure_chain) == 0

    @patch("debug_agent.OpenAI")
    def test_tuple_input(self, mock_openai_cls):
        """Test that tuple inputs (raw ClickHouse rows) are handled."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = MOCK_SUCCESS_RESPONSE
        mock_client.chat.completions.create.return_value = mock_response

        agent = DebugAgent(openai_api_key="test-key")
        trace_row = tuple(make_trace_dict().values())
        span_rows = [tuple(make_span_dict().values())]
        analysis = agent.analyze_trace(trace_row, span_rows)

        assert isinstance(analysis, DebugAnalysis)
        assert analysis.trace_id == "trace-1"

    @patch("debug_agent.OpenAI")
    def test_openai_failure_fallback(self, mock_openai_cls):
        """Test graceful fallback when OpenAI API fails."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API unreachable")

        agent = DebugAgent(openai_api_key="test-key")
        trace = make_trace_dict()
        spans = [make_span_dict(status="error",
                                attributes=json.dumps({"error.message": "Test error"}))]
        analysis = agent.analyze_trace(trace, spans)

        # Should still return a valid analysis with fallback content
        assert isinstance(analysis, DebugAnalysis)
        assert analysis.confidence_score == 0.1  # fallback confidence


# ---------------------------------------------------------------------------
# DebugAnalysis Dataclass Tests
# ---------------------------------------------------------------------------


class TestDebugAnalysis:
    def test_to_dict(self):
        analysis = DebugAnalysis(
            trace_id="trace-1",
            root_cause="Test root cause",
            explanation="Test explanation",
            fix_suggestions=["Fix 1", "Fix 2"],
            execution_timeline=[
                TimelineEntry(
                    span_id="s1", name="test", span_type="custom",
                    started_at="2026-01-01", duration_ms=100.0, status="success",
                ),
            ],
            failure_chain=[
                FailureChainEntry(
                    span_id="s2", name="error", span_type="llm",
                    error_message="failed", is_root_cause=True,
                ),
            ],
            confidence_score=0.75,
        )
        d = analysis.to_dict()
        assert d["trace_id"] == "trace-1"
        assert d["root_cause"] == "Test root cause"
        assert len(d["fix_suggestions"]) == 2
        assert len(d["execution_timeline"]) == 1
        assert d["execution_timeline"][0]["span_id"] == "s1"
        assert d["failure_chain"][0]["is_root_cause"] is True

    def test_to_dict_roundtrip(self):
        """Ensure to_dict output can be JSON serialized."""
        analysis = DebugAnalysis(
            trace_id="t1",
            root_cause="cause",
            explanation="explain",
            fix_suggestions=["fix"],
            execution_timeline=[],
            failure_chain=[],
            confidence_score=0.5,
        )
        serialized = json.dumps(analysis.to_dict())
        deserialized = json.loads(serialized)
        assert deserialized["trace_id"] == "t1"


# ---------------------------------------------------------------------------
# Init Tests
# ---------------------------------------------------------------------------


class TestDebugAgentInit:
    @patch("debug_agent.OpenAI")
    def test_with_api_key(self, mock_openai):
        agent = DebugAgent(openai_api_key="test-key")
        mock_openai.assert_called_once_with(api_key="test-key")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"})
    @patch("debug_agent.OpenAI")
    def test_with_env_var(self, mock_openai):
        agent = DebugAgent()
        mock_openai.assert_called_once_with(api_key="env-key")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key(self):
        # Remove the key if it exists
        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(ValueError, match="OpenAI API key required"):
            DebugAgent()
