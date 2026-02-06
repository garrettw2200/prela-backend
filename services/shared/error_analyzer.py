"""
Error Analysis Module

Provides intelligent error classification and actionable recommendations
for debugging agent failures.

This module analyzes span errors and generates context-aware suggestions
for fixes, including one-click replay parameters and code snippets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ErrorCategory(str, Enum):
    """Categories of errors that can occur in agent execution."""

    RATE_LIMIT = "rate_limit"  # 429, "rate limit exceeded"
    AUTH_FAILURE = "auth_failure"  # 401, "invalid api key"
    TOKEN_LIMIT = "token_limit"  # "context length exceeded"
    MODEL_NOT_FOUND = "model_not_found"  # "model does not exist"
    NETWORK_ERROR = "network_error"  # timeout, connection refused
    INVALID_REQUEST = "invalid_request"  # 400, malformed request
    SERVICE_ERROR = "service_error"  # 500, 503, provider issues
    TOOL_ERROR = "tool_error"  # Tool not found, execution failed
    RETRIEVAL_ERROR = "retrieval_error"  # Vector DB issues
    UNKNOWN = "unknown"  # Unclassified errors


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""

    CRITICAL = "CRITICAL"  # System-breaking, requires immediate attention
    HIGH = "HIGH"  # Major functionality impaired
    MEDIUM = "MEDIUM"  # Partial functionality affected
    LOW = "LOW"  # Minor issue, degraded experience


@dataclass
class ErrorRecommendation:
    """A single recommendation for fixing an error."""

    title: str  # e.g., "Switch to a cheaper model"
    description: str  # Detailed explanation
    action_type: str  # "replay" | "code_change" | "config_change"
    replay_params: dict[str, Any] | None  # Parameters for one-click replay
    code_snippet: str | None  # Example code to fix the issue
    confidence: float  # 0.0-1.0, how confident we are in this fix
    estimated_cost_impact: str | None  # e.g., "+$0.50/day" or "-50% tokens"


@dataclass
class ErrorAnalysis:
    """Complete analysis of a span error with recommendations."""

    category: ErrorCategory
    severity: ErrorSeverity
    error_type: str | None  # Exception class name
    error_message: str  # Full error message
    error_code: int | None  # HTTP status code if applicable
    recommendations: list[ErrorRecommendation]
    context: dict[str, Any]  # Additional context (model, tokens, etc.)


class ErrorAnalyzer:
    """Analyzes span errors and provides actionable insights."""

    @staticmethod
    def analyze_span_error(span_data: dict[str, Any]) -> ErrorAnalysis:
        """
        Analyzes a failed span and returns structured error information.

        Args:
            span_data: Span dictionary with attributes, events, status.
                Expected keys: attributes (dict), events (list), span_type, name

        Returns:
            ErrorAnalysis with category, severity, and recommendations
        """
        # Extract error information from span attributes
        attributes = span_data.get("attributes", {})
        error_type = attributes.get("error.type")
        error_message = attributes.get("error.message", "")
        error_code = attributes.get("error.status_code")

        # Classify the error
        category = ErrorAnalyzer._classify_error(error_message, error_code, error_type)

        # Calculate severity
        severity = ErrorAnalyzer._calculate_severity(category, span_data)

        # Generate recommendations
        recommendations = ErrorAnalyzer._generate_recommendations(
            category=category, span_data=span_data, error_message=error_message
        )

        # Extract additional context
        context = ErrorAnalyzer._extract_error_context(span_data)

        return ErrorAnalysis(
            category=category,
            severity=severity,
            error_type=error_type,
            error_message=error_message,
            error_code=error_code,
            recommendations=recommendations,
            context=context,
        )

    @staticmethod
    def _classify_error(
        message: str, code: int | None, error_type: str | None
    ) -> ErrorCategory:
        """
        Classifies an error into a category based on message, code, and type.

        Args:
            message: Error message text
            code: HTTP status code (if applicable)
            error_type: Exception class name

        Returns:
            ErrorCategory enum value
        """
        message_lower = message.lower() if message else ""

        # Rate limiting (highest priority - affects all subsequent calls)
        if code == 429 or "rate limit" in message_lower or "429" in message_lower:
            return ErrorCategory.RATE_LIMIT

        # Authentication failures
        if (
            code == 401
            or "api key" in message_lower
            or "unauthorized" in message_lower
            or "authentication" in message_lower
            or "invalid key" in message_lower
        ):
            return ErrorCategory.AUTH_FAILURE

        # Token/context limits
        if (
            "context length" in message_lower
            or "maximum context" in message_lower
            or "token limit" in message_lower
            or "max tokens" in message_lower
        ):
            return ErrorCategory.TOKEN_LIMIT

        # Model not found/available
        if "model" in message_lower and (
            "not found" in message_lower
            or "does not exist" in message_lower
            or "not available" in message_lower
            or "unavailable" in message_lower
        ):
            return ErrorCategory.MODEL_NOT_FOUND

        # Network issues
        if (
            "timeout" in message_lower
            or "connection" in message_lower
            or "network" in message_lower
            or code in [502, 503, 504]
        ):
            return ErrorCategory.NETWORK_ERROR

        # Invalid request (client error)
        if code == 400 or "invalid" in message_lower or "malformed" in message_lower:
            return ErrorCategory.INVALID_REQUEST

        # Service errors (server error)
        if code in [500, 503] or "server error" in message_lower or "service" in message_lower:
            return ErrorCategory.SERVICE_ERROR

        # Tool errors
        if (
            "tool" in message_lower
            and error_type
            and ("ToolError" in error_type or "FunctionNotFoundError" in error_type)
        ):
            return ErrorCategory.TOOL_ERROR

        # Retrieval/vector DB errors
        if (
            "retrieval" in message_lower
            or "vector" in message_lower
            or "embedding" in message_lower
            or "chroma" in message_lower
            or "pinecone" in message_lower
        ):
            return ErrorCategory.RETRIEVAL_ERROR

        # Unknown error
        return ErrorCategory.UNKNOWN

    @staticmethod
    def _calculate_severity(category: ErrorCategory, span_data: dict[str, Any]) -> ErrorSeverity:
        """
        Calculates the severity of an error based on its category and context.

        Args:
            category: The error category
            span_data: Span data for additional context

        Returns:
            ErrorSeverity enum value
        """
        # Authentication and service errors are always critical
        if category in [ErrorCategory.AUTH_FAILURE, ErrorCategory.SERVICE_ERROR]:
            return ErrorSeverity.CRITICAL

        # Rate limits and network errors are high priority
        if category in [ErrorCategory.RATE_LIMIT, ErrorCategory.NETWORK_ERROR]:
            return ErrorSeverity.HIGH

        # Token limits and model not found are medium
        if category in [ErrorCategory.TOKEN_LIMIT, ErrorCategory.MODEL_NOT_FOUND]:
            return ErrorSeverity.MEDIUM

        # Tool and retrieval errors are medium
        if category in [ErrorCategory.TOOL_ERROR, ErrorCategory.RETRIEVAL_ERROR]:
            return ErrorSeverity.MEDIUM

        # Invalid requests are low (user error, easy to fix)
        if category == ErrorCategory.INVALID_REQUEST:
            return ErrorSeverity.LOW

        # Unknown errors default to medium
        return ErrorSeverity.MEDIUM

    @staticmethod
    def _generate_recommendations(
        category: ErrorCategory, span_data: dict[str, Any], error_message: str
    ) -> list[ErrorRecommendation]:
        """
        Generates actionable recommendations based on error category.

        Args:
            category: The error category
            span_data: Span data for context
            error_message: Original error message

        Returns:
            List of ErrorRecommendation objects
        """
        recommendations: list[ErrorRecommendation] = []
        attributes = span_data.get("attributes", {})

        if category == ErrorCategory.RATE_LIMIT:
            # Automatic retry recommendation
            recommendations.append(
                ErrorRecommendation(
                    title="Wait and retry with exponential backoff",
                    description="Rate limits reset after a short period. The replay engine will automatically retry with exponential backoff.",
                    action_type="replay",
                    replay_params={},  # No changes needed, retry logic handles it
                    code_snippet=None,
                    confidence=0.95,
                    estimated_cost_impact=None,
                )
            )

            # Suggest cheaper model if expensive model was used
            model = attributes.get("llm.model", "")
            if model and "gpt-4" in model.lower() and "mini" not in model.lower():
                recommendations.append(
                    ErrorRecommendation(
                        title="Switch to gpt-4o-mini for lower rate limits",
                        description="gpt-4o-mini has higher rate limits and is 83% cheaper. Try this for non-critical tasks.",
                        action_type="replay",
                        replay_params={"model": "gpt-4o-mini"},
                        code_snippet='client.chat.completions.create(model="gpt-4o-mini", ...)',
                        confidence=0.85,
                        estimated_cost_impact="-83% cost, +200% rate limit capacity",
                    )
                )

        elif category == ErrorCategory.TOKEN_LIMIT:
            # Extract current max_tokens (default to 2048 if not found)
            max_tokens = attributes.get("llm.max_tokens", 2048)
            recommended_tokens = int(max_tokens * 1.5)

            recommendations.append(
                ErrorRecommendation(
                    title=f"Increase max_tokens to {recommended_tokens}",
                    description="Your response was cut off due to token limits. Increasing max_tokens allows longer responses.",
                    action_type="replay",
                    replay_params={"max_tokens": recommended_tokens},
                    code_snippet=f'client.chat.completions.create(max_tokens={recommended_tokens}, ...)',
                    confidence=0.90,
                    estimated_cost_impact=f"+50% output capacity, +${(recommended_tokens - max_tokens) * 0.00001:.4f} per call",
                )
            )

            # Also suggest prompt compression
            recommendations.append(
                ErrorRecommendation(
                    title="Compress your system prompt",
                    description="Remove unnecessary examples or verbose instructions to reduce input token usage.",
                    action_type="code_change",
                    replay_params=None,
                    code_snippet="# Remove verbose examples\nsystem_prompt = 'Be concise. Answer directly.'  # Instead of long examples",
                    confidence=0.70,
                    estimated_cost_impact="-20% input tokens",
                )
            )

        elif category == ErrorCategory.MODEL_NOT_FOUND:
            # Suggest alternative models
            model = attributes.get("llm.model", "")
            if model:
                alternatives = ErrorAnalyzer._get_model_alternatives(model)

                for alt_model in alternatives[:2]:  # Top 2 alternatives
                    recommendations.append(
                        ErrorRecommendation(
                            title=f"Try {alt_model} instead",
                            description=f"Model '{model}' is not available. {alt_model} is a similar alternative.",
                            action_type="replay",
                            replay_params={"model": alt_model},
                            code_snippet=f'client.chat.completions.create(model="{alt_model}", ...)',
                            confidence=0.80,
                            estimated_cost_impact="Similar pricing and capabilities",
                        )
                    )

        elif category == ErrorCategory.AUTH_FAILURE:
            recommendations.append(
                ErrorRecommendation(
                    title="Check your API key configuration",
                    description="Your API key is invalid or missing. Verify it's set correctly in your environment.",
                    action_type="config_change",
                    replay_params=None,
                    code_snippet='# Set your API key\nimport os\nos.environ["OPENAI_API_KEY"] = "sk-..."  # Or ANTHROPIC_API_KEY',
                    confidence=0.95,
                    estimated_cost_impact=None,
                )
            )

        elif category == ErrorCategory.NETWORK_ERROR:
            recommendations.append(
                ErrorRecommendation(
                    title="Retry the request",
                    description="Network errors are usually transient. The replay engine will automatically retry.",
                    action_type="replay",
                    replay_params={},
                    code_snippet=None,
                    confidence=0.85,
                    estimated_cost_impact=None,
                )
            )

            recommendations.append(
                ErrorRecommendation(
                    title="Check your network connection",
                    description="If the error persists, verify your internet connection and firewall settings.",
                    action_type="config_change",
                    replay_params=None,
                    code_snippet="# Test connection\nimport requests\nrequests.get('https://api.openai.com/v1/models')",
                    confidence=0.70,
                    estimated_cost_impact=None,
                )
            )

        elif category == ErrorCategory.TOOL_ERROR:
            recommendations.append(
                ErrorRecommendation(
                    title="Verify tool is registered",
                    description="Check that the tool function is properly defined and registered with your agent.",
                    action_type="code_change",
                    replay_params=None,
                    code_snippet="# Register tool\nagent.register_tool(tool_name, tool_function)",
                    confidence=0.80,
                    estimated_cost_impact=None,
                )
            )

        elif category == ErrorCategory.UNKNOWN:
            recommendations.append(
                ErrorRecommendation(
                    title="Review the error message",
                    description="This error doesn't match known patterns. Check the error details for clues.",
                    action_type="code_change",
                    replay_params=None,
                    code_snippet=None,
                    confidence=0.50,
                    estimated_cost_impact=None,
                )
            )

        return recommendations

    @staticmethod
    def _get_model_alternatives(model: str) -> list[str]:
        """
        Suggests alternative models based on the requested model.

        Args:
            model: The model that wasn't found

        Returns:
            List of suggested alternative model names
        """
        model_lower = model.lower()

        # OpenAI models
        if "gpt-4" in model_lower:
            return ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"]
        elif "gpt-3.5" in model_lower:
            return ["gpt-4o-mini", "gpt-3.5-turbo"]

        # Anthropic models
        elif "claude-3-opus" in model_lower or "opus" in model_lower:
            return ["claude-sonnet-4-20250514", "claude-3-sonnet-20240229"]
        elif "claude" in model_lower:
            return ["claude-sonnet-4-20250514", "claude-haiku-4-20250514"]

        # No specific alternatives
        return []

    @staticmethod
    def _extract_error_context(span_data: dict[str, Any]) -> dict[str, Any]:
        """
        Extracts relevant context from the span for error analysis.

        Args:
            span_data: Span data dictionary

        Returns:
            Dictionary with context information
        """
        attributes = span_data.get("attributes", {})
        context: dict[str, Any] = {}

        # LLM-related context
        if "llm.model" in attributes:
            context["model"] = attributes["llm.model"]
        if "llm.max_tokens" in attributes:
            context["max_tokens"] = attributes["llm.max_tokens"]
        if "llm.temperature" in attributes:
            context["temperature"] = attributes["llm.temperature"]
        if "llm.prompt_tokens" in attributes:
            context["prompt_tokens"] = attributes["llm.prompt_tokens"]
        if "llm.completion_tokens" in attributes:
            context["completion_tokens"] = attributes["llm.completion_tokens"]

        # Span metadata
        if "span_type" in span_data:
            context["span_type"] = span_data["span_type"]
        if "name" in span_data:
            context["span_name"] = span_data["name"]
        if "duration_ms" in span_data:
            context["duration_ms"] = span_data["duration_ms"]

        return context
