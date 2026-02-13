"""
Eval Generator Module

Analyzes production traces from ClickHouse to auto-generate eval test cases.
Outputs YAML compatible with the SDK's `prela eval run` command.

Pipeline:
1. Fetch candidate traces (errors, edge cases, positive examples)
2. Extract input/output from span attributes
3. Categorize traces into patterns (heuristic-first, LLM for ambiguous)
4. Generate eval cases per pattern via GPT-4o-mini
5. Assemble into EvalSuite YAML
"""

from __future__ import annotations

import json
import logging
import os
import re
import statistics
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import yaml
from openai import OpenAI

logger = logging.getLogger(__name__)

# Reuse column definitions from debug_agent
SPAN_COLUMNS = [
    "span_id", "trace_id", "project_id", "parent_span_id", "name",
    "span_type", "service_name", "started_at", "ended_at", "duration_ms",
    "status", "attributes", "events", "replay_snapshot", "source", "created_at",
]

TRACE_COLUMNS = [
    "trace_id", "project_id", "service_name", "started_at", "completed_at",
    "duration_ms", "status", "root_span_id", "span_count", "attributes",
    "source", "created_at",
]

# Maximum text length for extracted inputs/outputs (token budget)
MAX_TEXT_LENGTH = 500

# Valid assertion types in the SDK's eval runner
VALID_ASSERTION_TYPES = {
    "contains", "not_contains", "regex", "length",
    "json_valid", "tool_called", "tool_args", "tool_sequence",
}


def _row_to_dict(row: tuple | list, columns: list[str]) -> dict[str, Any]:
    """Convert a ClickHouse result row to a dict using column names."""
    return dict(zip(columns, row))


def _truncate(text: str, max_len: int = MAX_TEXT_LENGTH) -> str:
    """Truncate text to max_len, adding ellipsis if truncated."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _parse_attrs(raw: Any) -> dict[str, Any]:
    """Parse span attributes from raw string or dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class EvalGenerationConfig:
    """Configuration for eval generation."""

    project_id: str
    suite_name: str | None = None
    time_window_hours: int = 168  # 7 days
    max_traces: int = 500
    max_cases: int = 50
    include_failure_modes: bool = True
    include_edge_cases: bool = True
    include_positive_examples: bool = True
    agent_name_filter: str | None = None
    service_name_filter: str | None = None
    model: str = "gpt-4o-mini"
    max_tokens: int = 4000


@dataclass
class TracePattern:
    """A categorized group of traces sharing a common pattern."""

    pattern_id: str
    category: str  # "failure_mode", "edge_case", "positive_example"
    subcategory: str
    description: str
    trace_ids: list[str]
    representative_trace_id: str

    def label(self) -> str:
        return f"{self.category}/{self.subcategory}"


@dataclass
class GeneratedEvalCase:
    """An eval case generated from trace data, before final assembly."""

    case_id: str
    case_name: str
    input_query: str | None = None
    input_messages: list[dict] | None = None
    input_context: dict | None = None
    expected_output: str | None = None
    expected_contains: list[str] | None = None
    expected_not_contains: list[str] | None = None
    expected_tool_calls: list[dict] | None = None
    assertions: list[dict] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source_trace_id: str = ""
    source_pattern: str = ""


@dataclass
class EvalGenerationResult:
    """Complete result of eval generation."""

    generation_id: str
    project_id: str
    status: str  # "running", "completed", "failed"
    suite_name: str
    suite_yaml: str | None = None
    cases_generated: int = 0
    traces_analyzed: int = 0
    patterns_found: int = 0
    pattern_summary: list[dict] = field(default_factory=list)
    error: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# LLM Prompts
# ---------------------------------------------------------------------------

CATEGORIZATION_PROMPT = """You are analyzing production traces from an AI agent to categorize them for eval test generation.

For each trace below, classify it into one of these categories:
- failure_mode: Something went wrong (subcategory: rate_limit, tool_error, hallucination, timeout, auth_error, context_limit, other)
- edge_case: Unusual behavior worth testing (subcategory: slow_response, high_token, empty_response, unexpected_tool, other)
- positive_example: Good behavior to preserve (subcategory: correct_tool_usage, efficient_response, complete_answer, other)

Traces:
{traces_summary}

Respond ONLY with a JSON array, no other text:
[
  {{"trace_id": "...", "category": "...", "subcategory": "...", "description": "Brief description of what makes this trace interesting for testing"}}
]"""

EVAL_GENERATION_PROMPT = """You are generating eval test cases for an AI agent based on production trace data.

Pattern: {pattern_description}
Category: {category}/{subcategory}

Here are representative traces for this pattern:

{traces_detail}

Generate eval test cases that would catch this behavior pattern. Rules:
1. Use the ACTUAL input from the trace (not made-up data)
2. For expected output, extract 2-4 key phrases from the actual response (not the full response)
3. Generate 1-3 assertions per case from these types ONLY: contains, not_contains, length, json_valid, tool_called, tool_sequence
4. Keep test names descriptive and snake_case
5. For failure_mode: test that the agent handles the error condition
6. For edge_case: test that the agent handles the unusual scenario
7. For positive_example: test that good behavior is preserved

Respond ONLY with a JSON array, no other text:
[
  {{
    "name": "descriptive_test_name",
    "input": {{
      "query": "the input query"
    }},
    "expected": {{
      "contains": ["key phrase 1", "key phrase 2"]
    }},
    "assertions": [
      {{"type": "contains", "text": "expected text"}},
      {{"type": "length", "min_length": 10}}
    ],
    "tags": ["pattern_tag"]
  }}
]"""


# ---------------------------------------------------------------------------
# Main Class
# ---------------------------------------------------------------------------


class EvalGenerator:
    """Generates eval test cases from production trace data."""

    def __init__(self, openai_api_key: str | None = None):
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required for eval generation")
        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        project_id: str,
        config: EvalGenerationConfig,
        clickhouse_client: Any = None,
    ) -> EvalGenerationResult:
        """Main entry point. Orchestrates the full eval generation pipeline.

        Args:
            project_id: The project to generate evals for.
            config: Generation configuration.
            clickhouse_client: ClickHouse client (injected for testing).

        Returns:
            EvalGenerationResult with suite YAML and metadata.
        """
        generation_id = str(uuid.uuid4())
        suite_name = config.suite_name or f"Generated Suite - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        result = EvalGenerationResult(
            generation_id=generation_id,
            project_id=project_id,
            status="running",
            suite_name=suite_name,
        )

        try:
            if clickhouse_client is None:
                from shared import get_clickhouse_client
                clickhouse_client = get_clickhouse_client()

            # Step 1: Fetch candidate traces
            traces, spans_by_trace = self._fetch_candidate_traces(
                clickhouse_client, project_id, config
            )
            result.traces_analyzed = len(traces)

            if not traces:
                result.status = "completed"
                result.completed_at = datetime.now(timezone.utc).isoformat()
                result.suite_yaml = self._assemble_suite([], suite_name, config)
                return result

            # Step 2: Extract I/O from each trace
            traces_with_io = []
            for trace in traces:
                trace_id = trace.get("trace_id", "")
                spans = spans_by_trace.get(trace_id, [])
                io_data = self._extract_trace_io(trace, spans)
                traces_with_io.append({
                    "trace": trace,
                    "spans": spans,
                    "io": io_data,
                })

            # Step 3: Categorize into patterns
            patterns = self._categorize_traces(traces_with_io, config)
            result.patterns_found = len(patterns)
            result.pattern_summary = [
                {
                    "category": p.category,
                    "subcategory": p.subcategory,
                    "count": len(p.trace_ids),
                    "description": p.description,
                }
                for p in patterns
            ]

            # Step 4: Generate eval cases for each pattern
            all_cases: list[GeneratedEvalCase] = []
            for pattern in patterns:
                pattern_traces = [
                    t for t in traces_with_io
                    if t["trace"].get("trace_id") in pattern.trace_ids
                ]
                cases = self._generate_cases_for_pattern(
                    pattern, pattern_traces, config
                )
                all_cases.extend(cases)

                # Respect max_cases limit
                if len(all_cases) >= config.max_cases:
                    all_cases = all_cases[:config.max_cases]
                    break

            # Step 5: Assemble YAML
            result.suite_yaml = self._assemble_suite(all_cases, suite_name, config)
            result.cases_generated = len(all_cases)
            result.status = "completed"
            result.completed_at = datetime.now(timezone.utc).isoformat()

        except Exception as e:
            logger.error(f"Eval generation failed: {e}", exc_info=True)
            result.status = "failed"
            result.error = str(e)
            result.completed_at = datetime.now(timezone.utc).isoformat()

        return result

    # -------------------------------------------------------------------
    # Step 1: Fetch Traces
    # -------------------------------------------------------------------

    def _fetch_candidate_traces(
        self,
        client: Any,
        project_id: str,
        config: EvalGenerationConfig,
    ) -> tuple[list[dict], dict[str, list[dict]]]:
        """Fetch candidate traces and their spans from ClickHouse.

        Returns:
            (traces, spans_by_trace_id) — both as lists of dicts.
        """
        # Build WHERE clause
        conditions = [
            "project_id = %(project_id)s",
            "started_at >= now() - INTERVAL %(hours)s HOUR",
        ]
        params: dict[str, Any] = {
            "project_id": project_id,
            "hours": config.time_window_hours,
        }

        if config.service_name_filter:
            conditions.append("service_name = %(service_name)s")
            params["service_name"] = config.service_name_filter

        where = " AND ".join(conditions)

        # Fetch traces
        trace_query = f"""
            SELECT {', '.join(TRACE_COLUMNS)}
            FROM traces
            WHERE {where}
            ORDER BY started_at DESC
            LIMIT %(limit)s
        """
        params["limit"] = config.max_traces

        trace_result = client.query(trace_query, parameters=params)
        traces = [_row_to_dict(row, TRACE_COLUMNS) for row in trace_result.result_rows]

        if not traces:
            return [], {}

        # Apply agent_name filter (uses service_name column on traces)
        if config.agent_name_filter:
            traces = [
                t for t in traces
                if t.get("service_name", "") == config.agent_name_filter
            ]

        # Batch fetch spans for all trace IDs
        trace_ids = [t["trace_id"] for t in traces]
        spans_by_trace: dict[str, list[dict]] = {tid: [] for tid in trace_ids}

        # Fetch in batches of 100 trace IDs
        for i in range(0, len(trace_ids), 100):
            batch = trace_ids[i:i + 100]
            placeholders = ", ".join(f"'{tid}'" for tid in batch)
            span_query = f"""
                SELECT {', '.join(SPAN_COLUMNS)}
                FROM spans
                WHERE trace_id IN ({placeholders})
                  AND project_id = %(project_id)s
                ORDER BY started_at ASC
            """
            span_result = client.query(span_query, parameters={"project_id": project_id})
            for row in span_result.result_rows:
                span = _row_to_dict(row, SPAN_COLUMNS)
                tid = span.get("trace_id", "")
                if tid in spans_by_trace:
                    spans_by_trace[tid].append(span)

        return traces, spans_by_trace

    # -------------------------------------------------------------------
    # Step 2: Extract I/O
    # -------------------------------------------------------------------

    def _extract_trace_io(
        self,
        trace: dict[str, Any],
        spans: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Extract input/output from a trace's spans.

        Returns dict with keys: input, output, llm_spans, tool_spans, error_message
        """
        llm_spans = [s for s in spans if s.get("span_type") == "llm"]
        tool_spans = [s for s in spans if s.get("span_type") == "tool"]

        input_data: dict[str, Any] = {}
        output_data: dict[str, Any] = {}
        error_message = None

        # Extract input from first LLM span
        if llm_spans:
            first_attrs = _parse_attrs(llm_spans[0].get("attributes", ""))
            prompt = first_attrs.get("llm.prompt") or first_attrs.get("gen_ai.prompt", "")

            if prompt:
                prompt_str = str(prompt)
                # Try to detect chat message format
                try:
                    messages = json.loads(prompt_str)
                    if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
                        input_data["messages"] = messages
                    else:
                        input_data["query"] = _truncate(prompt_str)
                except (json.JSONDecodeError, TypeError):
                    input_data["query"] = _truncate(prompt_str)

            # Extract output from last LLM span
            last_attrs = _parse_attrs(llm_spans[-1].get("attributes", ""))
            response = last_attrs.get("llm.response") or last_attrs.get("gen_ai.completion", "")
            if response:
                output_data["response"] = _truncate(str(response))

        # Extract tool calls
        if tool_spans:
            tool_calls = []
            for ts in tool_spans:
                ts_attrs = _parse_attrs(ts.get("attributes", ""))
                tool_calls.append({
                    "name": ts_attrs.get("tool.name", ts.get("name", "unknown")),
                    "status": ts.get("status", "success"),
                })
            output_data["tool_calls"] = tool_calls

        # Extract error message
        if trace.get("status") == "error":
            error_spans = [s for s in spans if s.get("status") == "error"]
            for es in error_spans:
                events_raw = es.get("events", "")
                if events_raw:
                    try:
                        events = json.loads(events_raw) if isinstance(events_raw, str) else events_raw
                        if isinstance(events, list):
                            for event in events:
                                if isinstance(event, dict) and event.get("name") == "exception":
                                    msg = event.get("attributes", {}).get("exception.message", "")
                                    if msg:
                                        error_message = str(msg)
                                        break
                    except (json.JSONDecodeError, TypeError):
                        pass

                if error_message:
                    break

                attrs = _parse_attrs(es.get("attributes", ""))
                for key in ["error.message", "exception.message", "error"]:
                    if key in attrs:
                        error_message = str(attrs[key])
                        break
                if error_message:
                    break

        # Ensure at least a query is set (use trace name as fallback)
        if not input_data:
            service = trace.get("service_name", "")
            if service:
                input_data["query"] = f"[Trace from {service}]"

        return {
            "input": input_data,
            "output": output_data,
            "llm_spans": llm_spans,
            "tool_spans": tool_spans,
            "error_message": error_message,
        }

    # -------------------------------------------------------------------
    # Step 3: Categorize Traces
    # -------------------------------------------------------------------

    def _categorize_traces(
        self,
        traces_with_io: list[dict],
        config: EvalGenerationConfig,
    ) -> list[TracePattern]:
        """Categorize traces into patterns using heuristics and LLM."""
        # Collect duration values for percentile calculation
        durations = [
            float(t["trace"].get("duration_ms", 0) or 0)
            for t in traces_with_io
        ]
        p50 = statistics.median(durations) if durations else 0
        p95 = (
            sorted(durations)[int(len(durations) * 0.95)]
            if len(durations) >= 5
            else max(durations) if durations else 0
        )

        # Buckets for categorized traces
        buckets: dict[str, list[str]] = {}
        trace_lookup: dict[str, dict] = {}

        for item in traces_with_io:
            trace = item["trace"]
            io_data = item["io"]
            trace_id = trace.get("trace_id", "")
            trace_lookup[trace_id] = item
            status = trace.get("status", "")
            duration = float(trace.get("duration_ms", 0) or 0)
            error_msg = (io_data.get("error_message") or "").lower()

            categorized = False

            # Failure modes
            if status == "error" and config.include_failure_modes:
                if "rate limit" in error_msg or "429" in error_msg:
                    key = "failure_mode/rate_limit"
                elif "timeout" in error_msg or "timed out" in error_msg:
                    key = "failure_mode/timeout"
                elif "401" in error_msg or "unauthorized" in error_msg or "auth" in error_msg:
                    key = "failure_mode/auth_error"
                elif "context length" in error_msg or "token limit" in error_msg:
                    key = "failure_mode/context_limit"
                else:
                    # Check if tool spans failed
                    tool_errors = [
                        s for s in item.get("spans", [])
                        if s.get("span_type") == "tool" and s.get("status") == "error"
                    ]
                    if tool_errors:
                        key = "failure_mode/tool_error"
                    else:
                        key = "failure_mode/other"

                buckets.setdefault(key, []).append(trace_id)
                categorized = True

            # Edge cases (non-error traces with unusual characteristics)
            if not categorized and config.include_edge_cases:
                if duration > p95 and p95 > 0:
                    key = "edge_case/slow_response"
                    buckets.setdefault(key, []).append(trace_id)
                    categorized = True
                elif not io_data.get("output", {}).get("response"):
                    key = "edge_case/empty_response"
                    buckets.setdefault(key, []).append(trace_id)
                    categorized = True

            # Positive examples (successful traces)
            if not categorized and status == "success" and config.include_positive_examples:
                tool_calls = io_data.get("output", {}).get("tool_calls", [])
                all_tools_ok = all(tc.get("status") == "success" for tc in tool_calls) if tool_calls else False

                if all_tools_ok and tool_calls:
                    key = "positive_example/correct_tool_usage"
                elif duration <= p50 and p50 > 0:
                    key = "positive_example/fast_success"
                else:
                    key = "positive_example/complete_answer"

                buckets.setdefault(key, []).append(trace_id)

        # Convert buckets to TracePattern objects
        patterns: list[TracePattern] = []
        for key, trace_ids in buckets.items():
            if not trace_ids:
                continue
            category, subcategory = key.split("/", 1)

            # Description based on category
            descriptions = {
                "failure_mode/rate_limit": "Traces that hit API rate limits",
                "failure_mode/timeout": "Traces that timed out",
                "failure_mode/auth_error": "Traces with authentication failures",
                "failure_mode/context_limit": "Traces exceeding context/token limits",
                "failure_mode/tool_error": "Traces with tool execution failures",
                "failure_mode/other": "Traces with other error types",
                "edge_case/slow_response": f"Traces with duration > p95 ({p95:.0f}ms)",
                "edge_case/empty_response": "Traces with empty or missing LLM responses",
                "positive_example/correct_tool_usage": "Successful traces with all tool calls passing",
                "positive_example/fast_success": f"Successful traces faster than median ({p50:.0f}ms)",
                "positive_example/complete_answer": "Successful traces with complete answers",
            }

            patterns.append(TracePattern(
                pattern_id=str(uuid.uuid4()),
                category=category,
                subcategory=subcategory,
                description=descriptions.get(key, f"{category}: {subcategory}"),
                trace_ids=trace_ids[:20],  # Cap per pattern to limit LLM calls
                representative_trace_id=trace_ids[0],
            ))

        return patterns

    # -------------------------------------------------------------------
    # Step 4: Generate Eval Cases
    # -------------------------------------------------------------------

    def _generate_cases_for_pattern(
        self,
        pattern: TracePattern,
        traces_with_io: list[dict],
        config: EvalGenerationConfig,
    ) -> list[GeneratedEvalCase]:
        """Generate eval cases for a pattern using LLM.

        Uses up to 5 representative traces per pattern to keep token usage low.
        """
        # Pick up to 5 representative traces
        representative = traces_with_io[:5]
        if not representative:
            return []

        # Build traces detail for the prompt
        traces_detail_parts = []
        for i, item in enumerate(representative, 1):
            trace = item["trace"]
            io_data = item["io"]
            input_data = io_data.get("input", {})
            output_data = io_data.get("output", {})
            error_msg = io_data.get("error_message", "")

            detail = f"Trace {i} (ID: {trace.get('trace_id', 'N/A')}):\n"
            detail += f"  Service: {trace.get('service_name', 'N/A')}\n"
            detail += f"  Status: {trace.get('status', 'N/A')}\n"
            detail += f"  Duration: {trace.get('duration_ms', 0):.0f}ms\n"

            if input_data.get("query"):
                detail += f"  Input: {_truncate(input_data['query'], 300)}\n"
            elif input_data.get("messages"):
                msgs = input_data["messages"]
                # Show last user message
                user_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
                if user_msgs:
                    detail += f"  Input (last user message): {_truncate(str(user_msgs[-1].get('content', '')), 300)}\n"

            if output_data.get("response"):
                detail += f"  Output: {_truncate(output_data['response'], 300)}\n"

            if output_data.get("tool_calls"):
                tools = [tc.get("name", "?") for tc in output_data["tool_calls"]]
                detail += f"  Tool calls: {', '.join(tools)}\n"

            if error_msg:
                detail += f"  Error: {_truncate(error_msg, 200)}\n"

            traces_detail_parts.append(detail)

        traces_detail = "\n".join(traces_detail_parts)

        prompt = EVAL_GENERATION_PROMPT.format(
            pattern_description=pattern.description,
            category=pattern.category,
            subcategory=pattern.subcategory,
            traces_detail=traces_detail,
        )

        # Call LLM
        try:
            response = self.client.chat.completions.create(
                model=config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an eval test case generator. Output ONLY valid JSON arrays.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=config.max_tokens,
            )
            response_text = response.choices[0].message.content or "[]"
        except Exception as e:
            logger.error(f"LLM call failed for pattern {pattern.label()}: {e}")
            # Fallback: generate a simple heuristic case
            return self._generate_fallback_cases(pattern, traces_with_io)

        # Parse LLM response
        return self._parse_generated_cases(response_text, pattern, traces_with_io)

    def _parse_generated_cases(
        self,
        response_text: str,
        pattern: TracePattern,
        traces_with_io: list[dict],
    ) -> list[GeneratedEvalCase]:
        """Parse LLM-generated eval cases from JSON response."""
        # Extract JSON array from response (LLM may include markdown fencing)
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if not json_match:
            logger.warning(f"No JSON array found in LLM response for pattern {pattern.label()}")
            return self._generate_fallback_cases(pattern, traces_with_io)

        try:
            cases_data = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in LLM response for pattern {pattern.label()}")
            return self._generate_fallback_cases(pattern, traces_with_io)

        if not isinstance(cases_data, list):
            return self._generate_fallback_cases(pattern, traces_with_io)

        cases = []
        for i, case_data in enumerate(cases_data):
            if not isinstance(case_data, dict):
                continue

            case_name = case_data.get("name", f"{pattern.subcategory}_case_{i + 1}")
            input_data = case_data.get("input", {})
            expected = case_data.get("expected", {})
            raw_assertions = case_data.get("assertions", [])
            tags = case_data.get("tags", [])

            # Validate and filter assertions to only valid types
            assertions = []
            for a in raw_assertions:
                if isinstance(a, dict) and a.get("type") in VALID_ASSERTION_TYPES:
                    assertions.append(a)

            # Build expected fields
            expected_contains = expected.get("contains") if isinstance(expected, dict) else None
            expected_not_contains = expected.get("not_contains") if isinstance(expected, dict) else None
            expected_tool_calls = expected.get("tool_calls") if isinstance(expected, dict) else None
            expected_output = expected.get("output") if isinstance(expected, dict) else None

            # Ensure we have at least expected or assertions
            if not expected_contains and not expected_not_contains and not expected_tool_calls and not expected_output and not assertions:
                # Add a basic length assertion as fallback
                assertions = [{"type": "length", "min_length": 1}]

            source_trace_id = traces_with_io[0]["trace"].get("trace_id", "") if traces_with_io else ""

            cases.append(GeneratedEvalCase(
                case_id=f"{pattern.subcategory}_{case_name}_{i + 1}",
                case_name=case_name,
                input_query=input_data.get("query"),
                input_messages=input_data.get("messages"),
                input_context=input_data.get("context"),
                expected_output=expected_output,
                expected_contains=expected_contains,
                expected_not_contains=expected_not_contains,
                expected_tool_calls=expected_tool_calls,
                assertions=assertions,
                tags=[pattern.category, pattern.subcategory] + (tags if isinstance(tags, list) else []),
                source_trace_id=source_trace_id,
                source_pattern=pattern.label(),
            ))

        return cases

    def _generate_fallback_cases(
        self,
        pattern: TracePattern,
        traces_with_io: list[dict],
    ) -> list[GeneratedEvalCase]:
        """Generate simple heuristic-based eval cases when LLM fails."""
        cases = []
        for i, item in enumerate(traces_with_io[:3]):
            io_data = item["io"]
            trace = item["trace"]
            input_data = io_data.get("input", {})
            output_data = io_data.get("output", {})

            query = input_data.get("query")
            messages = input_data.get("messages")

            if not query and not messages:
                continue

            assertions: list[dict] = [{"type": "length", "min_length": 1}]

            # Add pattern-specific assertions
            if pattern.category == "positive_example":
                response = output_data.get("response", "")
                if response:
                    # Extract a key word from the response
                    words = response.split()
                    if len(words) > 3:
                        assertions.append({"type": "length", "min_length": 10})

            expected_contains = None
            if pattern.category == "positive_example" and output_data.get("response"):
                # Use first non-trivial sentence from response
                resp = output_data["response"]
                sentences = resp.split(".")
                key_phrases = [s.strip() for s in sentences if len(s.strip()) > 10]
                if key_phrases:
                    expected_contains = [key_phrases[0][:100]]

            cases.append(GeneratedEvalCase(
                case_id=f"fallback_{pattern.subcategory}_{i + 1}",
                case_name=f"fallback_{pattern.subcategory}_case_{i + 1}",
                input_query=query,
                input_messages=messages,
                expected_contains=expected_contains,
                assertions=assertions,
                tags=[pattern.category, pattern.subcategory, "fallback"],
                source_trace_id=trace.get("trace_id", ""),
                source_pattern=pattern.label(),
            ))

        return cases

    # -------------------------------------------------------------------
    # Step 5: Assemble Suite YAML
    # -------------------------------------------------------------------

    def _assemble_suite(
        self,
        cases: list[GeneratedEvalCase],
        suite_name: str,
        config: EvalGenerationConfig,
    ) -> str:
        """Assemble eval cases into SDK-compatible EvalSuite YAML."""
        suite_dict: dict[str, Any] = {
            "name": suite_name,
            "description": f"Auto-generated from {config.project_id} production traces",
            "default_assertions": [
                {"type": "length", "min_length": 1},
            ],
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "project_id": config.project_id,
                "traces_analyzed": getattr(config, "_traces_analyzed", 0),
                "generator_version": "1.0",
            },
            "cases": [],
        }

        for case in cases:
            case_dict: dict[str, Any] = {
                "id": case.case_id,
                "name": case.case_name,
                "input": {},
                "timeout_seconds": 30.0,
                "metadata": {
                    "source_trace_id": case.source_trace_id,
                    "source_pattern": case.source_pattern,
                },
            }

            # Input
            if case.input_query:
                case_dict["input"]["query"] = case.input_query
            if case.input_messages:
                case_dict["input"]["messages"] = case.input_messages
            if case.input_context:
                case_dict["input"]["context"] = case.input_context

            # Ensure input has at least query or messages
            if not case_dict["input"]:
                case_dict["input"]["query"] = "[No input extracted]"

            # Expected
            expected: dict[str, Any] = {}
            if case.expected_output:
                expected["output"] = case.expected_output
            if case.expected_contains:
                expected["contains"] = case.expected_contains
            if case.expected_not_contains:
                expected["not_contains"] = case.expected_not_contains
            if case.expected_tool_calls:
                expected["tool_calls"] = case.expected_tool_calls
            if expected:
                case_dict["expected"] = expected

            # Assertions
            if case.assertions:
                case_dict["assertions"] = case.assertions

            # Tags
            if case.tags:
                case_dict["tags"] = case.tags

            suite_dict["cases"].append(case_dict)

        return yaml.dump(
            suite_dict,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
