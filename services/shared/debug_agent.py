"""
Debug Agent Module

Analyzes traces to produce plain-English explanations of what went wrong.
Collects spans, builds an execution timeline, identifies the failure chain,
and uses GPT-4o-mini to generate root cause analysis and fix suggestions.
Results are cached in the ClickHouse analysis_results table.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class TimelineEntry:
    """A single entry in the execution timeline."""

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
    """A link in the failure chain showing error propagation."""

    span_id: str
    name: str
    span_type: str
    error_message: str
    is_root_cause: bool = False


@dataclass
class DebugAnalysis:
    """Complete debug analysis of a trace."""

    trace_id: str
    root_cause: str
    explanation: str
    fix_suggestions: list[str]
    execution_timeline: list[TimelineEntry]
    failure_chain: list[FailureChainEntry]
    confidence_score: float
    analyzed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Spans table columns (order matches CREATE TABLE in clickhouse.py)
SPAN_COLUMNS = [
    "span_id", "trace_id", "project_id", "parent_span_id", "name",
    "span_type", "service_name", "started_at", "ended_at", "duration_ms",
    "status", "attributes", "events", "replay_snapshot", "source", "created_at",
]

# Traces table columns
TRACE_COLUMNS = [
    "trace_id", "project_id", "service_name", "started_at", "completed_at",
    "duration_ms", "status", "root_span_id", "span_count", "attributes",
    "source", "created_at",
]


def _row_to_dict(row: tuple | list, columns: list[str]) -> dict[str, Any]:
    """Convert a ClickHouse result row to a dict using column names."""
    return dict(zip(columns, row))


class DebugAgent:
    """Analyzes traces and produces plain-English debug explanations."""

    def __init__(self, openai_api_key: str | None = None):
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required for debug analysis")
        self.client = OpenAI(api_key=api_key)

    def analyze_trace(
        self,
        trace_data: tuple | list | dict,
        spans_data: list[tuple | list | dict],
        model: str = "gpt-4o-mini",
        max_tokens: int = 2000,
    ) -> DebugAnalysis:
        """Analyze a trace and produce a debug explanation.

        Args:
            trace_data: Trace row (tuple from ClickHouse or dict).
            spans_data: List of span rows (tuples from ClickHouse or dicts).
            model: OpenAI model to use.
            max_tokens: Max tokens for LLM response.

        Returns:
            DebugAnalysis with root cause, explanation, and fix suggestions.
        """
        # Normalize to dicts
        trace = trace_data if isinstance(trace_data, dict) else _row_to_dict(trace_data, TRACE_COLUMNS)
        spans = [s if isinstance(s, dict) else _row_to_dict(s, SPAN_COLUMNS) for s in spans_data]

        # Build timeline and failure chain
        timeline = self._build_timeline(spans)
        failure_chain = self._identify_failure_chain(spans)

        # Build prompt
        prompt = self._build_prompt(trace, spans, timeline, failure_chain)

        # Call LLM
        response_text = self._call_openai(prompt, model, max_tokens)

        # Parse response
        return self._parse_response(
            response_text,
            trace_id=trace.get("trace_id", ""),
            timeline=timeline,
            failure_chain=failure_chain,
        )

    def _build_timeline(self, spans: list[dict[str, Any]]) -> list[TimelineEntry]:
        """Build a chronological execution timeline from spans."""
        entries = []
        for span in spans:
            error_msg = None
            if span.get("status") == "error":
                # Try to extract error message from events or attributes
                error_msg = self._extract_error_message(span)

            entries.append(TimelineEntry(
                span_id=str(span.get("span_id", "")),
                name=str(span.get("name", "")),
                span_type=str(span.get("span_type", "custom")),
                started_at=str(span.get("started_at", "")),
                duration_ms=float(span.get("duration_ms", 0) or 0),
                status=str(span.get("status", "success")),
                error_message=error_msg,
                parent_span_id=str(span.get("parent_span_id", "")) or None,
            ))
        return entries

    def _identify_failure_chain(self, spans: list[dict[str, Any]]) -> list[FailureChainEntry]:
        """Identify the failure chain — root error and cascading effects."""
        error_spans = [s for s in spans if s.get("status") == "error"]
        if not error_spans:
            return []

        chain = []
        for i, span in enumerate(error_spans):
            error_msg = self._extract_error_message(span) or "Unknown error"
            chain.append(FailureChainEntry(
                span_id=str(span.get("span_id", "")),
                name=str(span.get("name", "")),
                span_type=str(span.get("span_type", "custom")),
                error_message=error_msg,
                is_root_cause=(i == 0),
            ))
        return chain

    def _extract_error_message(self, span: dict[str, Any]) -> str | None:
        """Extract error message from span events or attributes."""
        # Check events for exception events
        events_raw = span.get("events", "")
        if events_raw:
            try:
                events = json.loads(events_raw) if isinstance(events_raw, str) else events_raw
                if isinstance(events, list):
                    for event in events:
                        if isinstance(event, dict) and event.get("name") == "exception":
                            attrs = event.get("attributes", {})
                            msg = attrs.get("exception.message", "")
                            if msg:
                                return str(msg)
            except (json.JSONDecodeError, TypeError):
                pass

        # Check attributes for error info
        attrs_raw = span.get("attributes", "")
        if attrs_raw:
            try:
                attrs = json.loads(attrs_raw) if isinstance(attrs_raw, str) else attrs_raw
                if isinstance(attrs, dict):
                    for key in ["error.message", "exception.message", "error"]:
                        if key in attrs:
                            return str(attrs[key])
            except (json.JSONDecodeError, TypeError):
                pass

        return None

    def _build_prompt(
        self,
        trace: dict[str, Any],
        spans: list[dict[str, Any]],
        timeline: list[TimelineEntry],
        failure_chain: list[FailureChainEntry],
    ) -> str:
        """Build the LLM prompt for trace debugging."""
        # Trace summary
        trace_summary = (
            f"Trace ID: {trace.get('trace_id', 'N/A')}\n"
            f"Service: {trace.get('service_name', 'N/A')}\n"
            f"Status: {trace.get('status', 'N/A')}\n"
            f"Duration: {trace.get('duration_ms', 0):.1f}ms\n"
            f"Span count: {trace.get('span_count', len(spans))}\n"
        )

        # Timeline summary (keep concise for token budget)
        timeline_lines = []
        for entry in timeline[:30]:  # Cap at 30 spans
            status_marker = "ERROR" if entry.status == "error" else "OK"
            line = f"  [{status_marker}] {entry.name} ({entry.span_type}) — {entry.duration_ms:.0f}ms"
            if entry.error_message:
                # Truncate long error messages
                err = entry.error_message[:200]
                line += f"\n         Error: {err}"
            timeline_lines.append(line)
        timeline_text = "\n".join(timeline_lines)

        # Failure chain
        if failure_chain:
            chain_lines = []
            for entry in failure_chain:
                prefix = ">>> ROOT CAUSE" if entry.is_root_cause else "    CASCADE"
                chain_lines.append(f"  {prefix}: {entry.name} ({entry.span_type})")
                chain_lines.append(f"           Error: {entry.error_message[:200]}")
            failure_text = "\n".join(chain_lines)
        else:
            failure_text = "  No errors detected — trace completed successfully."

        # Extract key span details (LLM prompts/responses, tool calls)
        context_details = self._extract_context_details(spans)

        prompt = f"""You are an AI agent debugging assistant. Analyze this trace execution and provide a clear, developer-friendly explanation.

**Trace Summary:**
{trace_summary}

**Execution Timeline (chronological):**
{timeline_text}

**Failure Chain:**
{failure_text}

**Key Span Details:**
{context_details}

**Your Task:**
Provide a concise debug analysis in this exact format:

ROOT_CAUSE: (1-2 sentences identifying the root cause of any issues, or the main observation if the trace succeeded)

EXPLANATION: (2-4 sentences explaining what happened step by step, in plain English. Focus on the "why" — why did this error occur, or why did performance degrade, etc.)

FIX_SUGGESTIONS:
1. (Most impactful fix or improvement)
2. (Second suggestion)
3. (Third suggestion, if applicable)

CONFIDENCE: (a number 0.0-1.0 indicating your confidence in this analysis)

Be concise, practical, and developer-friendly. Avoid jargon where possible."""

        return prompt

    def _extract_context_details(self, spans: list[dict[str, Any]]) -> str:
        """Extract relevant context from span attributes for the LLM."""
        details = []
        for span in spans[:20]:  # Cap to avoid huge prompts
            attrs_raw = span.get("attributes", "")
            try:
                attrs = json.loads(attrs_raw) if isinstance(attrs_raw, str) else attrs_raw
            except (json.JSONDecodeError, TypeError):
                continue

            if not isinstance(attrs, dict):
                continue

            span_type = span.get("span_type", "")
            name = span.get("name", "")

            if span_type == "llm":
                # Extract model, prompt snippet, response snippet
                model = attrs.get("llm.model", attrs.get("gen_ai.request.model", "unknown"))
                prompt = attrs.get("llm.prompt", attrs.get("gen_ai.prompt", ""))
                response = attrs.get("llm.response", attrs.get("gen_ai.completion", ""))
                tokens = attrs.get("llm.total_tokens", attrs.get("gen_ai.usage.total_tokens", ""))

                if isinstance(prompt, str) and len(prompt) > 300:
                    prompt = prompt[:300] + "..."
                if isinstance(response, str) and len(response) > 300:
                    response = response[:300] + "..."

                details.append(
                    f"  LLM Call [{name}]: model={model}"
                    + (f", tokens={tokens}" if tokens else "")
                    + (f"\n    Prompt: {prompt}" if prompt else "")
                    + (f"\n    Response: {response}" if response else "")
                )

            elif span_type == "tool":
                tool_name = attrs.get("tool.name", name)
                tool_input = attrs.get("tool.input", "")
                if isinstance(tool_input, str) and len(tool_input) > 200:
                    tool_input = tool_input[:200] + "..."
                details.append(f"  Tool Call [{tool_name}]: input={tool_input}")

            elif span_type == "retrieval":
                query = attrs.get("retrieval.query", attrs.get("db.statement", ""))
                if isinstance(query, str) and len(query) > 200:
                    query = query[:200] + "..."
                details.append(f"  Retrieval [{name}]: query={query}")

        if not details:
            return "  (No detailed span attributes available)"

        return "\n".join(details)

    def _call_openai(self, prompt: str, model: str, max_tokens: int) -> str:
        """Call OpenAI API for debug analysis."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a debugging assistant for AI agent traces. Be concise and practical.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return self._get_fallback_response()

    def _get_fallback_response(self) -> str:
        """Fallback response when LLM call fails."""
        return (
            "ROOT_CAUSE: Unable to generate AI-powered analysis at this time.\n\n"
            "EXPLANATION: The debug analysis service encountered an error calling the LLM API. "
            "Review the execution timeline and failure chain above for manual debugging.\n\n"
            "FIX_SUGGESTIONS:\n"
            "1. Check the execution timeline for error spans\n"
            "2. Review error messages in the failure chain\n"
            "3. Check span attributes for detailed context\n\n"
            "CONFIDENCE: 0.1"
        )

    def _parse_response(
        self,
        response: str,
        trace_id: str,
        timeline: list[TimelineEntry],
        failure_chain: list[FailureChainEntry],
    ) -> DebugAnalysis:
        """Parse LLM response into DebugAnalysis."""
        root_cause = ""
        explanation = ""
        fix_suggestions: list[str] = []
        confidence = 0.5

        # Parse ROOT_CAUSE
        if "ROOT_CAUSE:" in response:
            parts = response.split("ROOT_CAUSE:", 1)
            remainder = parts[1]
            # Find next section
            for marker in ["EXPLANATION:", "FIX_SUGGESTIONS:", "CONFIDENCE:"]:
                if marker in remainder:
                    root_cause = remainder.split(marker, 1)[0].strip()
                    break
            else:
                root_cause = remainder.strip()

        # Parse EXPLANATION
        if "EXPLANATION:" in response:
            parts = response.split("EXPLANATION:", 1)
            remainder = parts[1]
            for marker in ["FIX_SUGGESTIONS:", "CONFIDENCE:"]:
                if marker in remainder:
                    explanation = remainder.split(marker, 1)[0].strip()
                    break
            else:
                explanation = remainder.strip()

        # Parse FIX_SUGGESTIONS
        if "FIX_SUGGESTIONS:" in response:
            parts = response.split("FIX_SUGGESTIONS:", 1)
            remainder = parts[1]
            if "CONFIDENCE:" in remainder:
                suggestions_text = remainder.split("CONFIDENCE:", 1)[0].strip()
            else:
                suggestions_text = remainder.strip()

            for line in suggestions_text.split("\n"):
                line = line.strip()
                if line and line[0].isdigit() and "." in line:
                    # Remove the number prefix (e.g., "1. ")
                    suggestion = line.split(".", 1)[1].strip()
                    if suggestion:
                        fix_suggestions.append(suggestion)

        # Parse CONFIDENCE
        if "CONFIDENCE:" in response:
            conf_text = response.split("CONFIDENCE:", 1)[1].strip().split("\n")[0].strip()
            try:
                confidence = float(conf_text)
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = 0.5

        return DebugAnalysis(
            trace_id=trace_id,
            root_cause=root_cause or "Analysis could not determine a specific root cause.",
            explanation=explanation or "Review the execution timeline for details.",
            fix_suggestions=fix_suggestions or ["Review span details for more context"],
            execution_timeline=timeline,
            failure_chain=failure_chain,
            confidence_score=confidence,
        )
