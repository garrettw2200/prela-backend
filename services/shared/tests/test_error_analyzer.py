"""Tests for error analyzer module."""

import pytest

from shared.error_analyzer import (
    ErrorAnalyzer,
    ErrorAnalysis,
    ErrorCategory,
    ErrorRecommendation,
    ErrorSeverity,
)


class TestErrorClassification:
    """Tests for error classification logic."""

    def test_rate_limit_detection_429(self):
        """Test detection of rate limit errors via HTTP 429."""
        span_data = {
            "attributes": {
                "error.message": "Rate limit exceeded. Please retry after 30 seconds.",
                "error.status_code": 429,
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        assert analysis.category == ErrorCategory.RATE_LIMIT
        assert analysis.error_code == 429
        assert analysis.severity in [ErrorSeverity.HIGH, ErrorSeverity.MEDIUM]
        assert len(analysis.recommendations) > 0
        assert any("retry" in rec.description.lower() for rec in analysis.recommendations)

    def test_rate_limit_detection_message(self):
        """Test detection of rate limit errors via message text."""
        span_data = {
            "attributes": {
                "error.message": "You have exceeded your rate limit. Try again later.",
                "error.status_code": None,
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        assert analysis.category == ErrorCategory.RATE_LIMIT

    def test_auth_failure_detection_401(self):
        """Test detection of authentication failures via HTTP 401."""
        span_data = {
            "attributes": {
                "error.message": "Invalid API key provided",
                "error.status_code": 401,
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        assert analysis.category == ErrorCategory.AUTH_FAILURE
        assert analysis.severity == ErrorSeverity.CRITICAL
        assert len(analysis.recommendations) > 0

    def test_auth_failure_detection_message(self):
        """Test detection of auth failures via message text."""
        span_data = {
            "attributes": {
                "error.message": "Authentication failed: unauthorized access",
                "error.status_code": None,
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        assert analysis.category == ErrorCategory.AUTH_FAILURE

    def test_token_limit_detection(self):
        """Test detection of token/context limit errors."""
        span_data = {
            "attributes": {
                "error.message": "This model's maximum context length is 4096 tokens. You requested 5000 tokens.",
                "llm.max_tokens": 2048,
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        assert analysis.category == ErrorCategory.TOKEN_LIMIT
        assert analysis.severity in [ErrorSeverity.HIGH, ErrorSeverity.MEDIUM]
        # Should suggest increasing max_tokens
        assert any(
            rec.replay_params and "max_tokens" in rec.replay_params
            for rec in analysis.recommendations
        )

    def test_model_not_found_detection(self):
        """Test detection of model not found errors."""
        span_data = {
            "attributes": {
                "error.message": "The model 'gpt-4-ultra' does not exist",
                "llm.model": "gpt-4-ultra",
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        assert analysis.category == ErrorCategory.MODEL_NOT_FOUND
        assert len(analysis.recommendations) > 0
        # Should suggest alternative models
        assert any("model" in rec.replay_params for rec in analysis.recommendations if rec.replay_params)

    def test_network_error_detection_timeout(self):
        """Test detection of network timeout errors."""
        span_data = {
            "attributes": {
                "error.message": "Request timeout after 30 seconds",
                "error.type": "TimeoutError",
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        assert analysis.category == ErrorCategory.NETWORK_ERROR

    def test_network_error_detection_connection(self):
        """Test detection of network connection errors."""
        span_data = {
            "attributes": {
                "error.message": "Connection refused by server",
                "error.type": "ConnectionError",
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        assert analysis.category == ErrorCategory.NETWORK_ERROR

    def test_tool_error_detection(self):
        """Test detection of tool execution errors."""
        span_data = {
            "attributes": {
                "error.message": "Tool 'calculator' not found",
                "error.type": "ToolError",
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        assert analysis.category == ErrorCategory.TOOL_ERROR

    def test_unknown_error_fallback(self):
        """Test fallback to unknown category for unrecognized errors."""
        span_data = {
            "attributes": {
                "error.message": "Something mysterious happened",
                "error.type": "WeirdError",
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        assert analysis.category == ErrorCategory.UNKNOWN


class TestRecommendationGeneration:
    """Tests for recommendation generation logic."""

    def test_rate_limit_recommendations(self):
        """Test recommendations for rate limit errors."""
        span_data = {
            "attributes": {
                "error.message": "Rate limit exceeded",
                "error.status_code": 429,
                "llm.model": "gpt-4",
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        # Should have retry recommendation
        assert any(
            rec.action_type == "replay" and "retry" in rec.description.lower()
            for rec in analysis.recommendations
        )

        # Should suggest cheaper model if expensive model used
        assert any(
            rec.replay_params and "model" in rec.replay_params
            for rec in analysis.recommendations
        )

    def test_token_limit_recommendations_increase(self):
        """Test token limit recommendation suggests increasing tokens."""
        span_data = {
            "attributes": {
                "error.message": "Context length exceeded",
                "llm.max_tokens": 2048,
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        # Should suggest increasing max_tokens by ~50%
        token_recs = [
            rec
            for rec in analysis.recommendations
            if rec.replay_params and "max_tokens" in rec.replay_params
        ]
        assert len(token_recs) > 0
        assert token_recs[0].replay_params["max_tokens"] > 2048

    def test_token_limit_recommendations_compression(self):
        """Test token limit also suggests prompt compression."""
        span_data = {
            "attributes": {
                "error.message": "Context length exceeded",
                "llm.max_tokens": 2048,
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        # Should suggest compressing prompt
        assert any(
            "compress" in rec.description.lower() or "reduce" in rec.description.lower()
            for rec in analysis.recommendations
        )

    def test_model_alternatives_gpt4(self):
        """Test model alternatives for GPT-4."""
        span_data = {
            "attributes": {
                "error.message": "Model 'gpt-4' does not exist",
                "llm.model": "gpt-4",
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        # Should suggest gpt-4o, gpt-4-turbo, or gpt-4o-mini
        model_names = [
            rec.replay_params.get("model")
            for rec in analysis.recommendations
            if rec.replay_params and "model" in rec.replay_params
        ]
        assert any("gpt-4o" in model or "gpt-4-turbo" in model for model in model_names if model)

    def test_model_alternatives_claude(self):
        """Test model alternatives for Claude."""
        span_data = {
            "attributes": {
                "error.message": "Model 'claude-3-opus' does not exist",
                "llm.model": "claude-3-opus",
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        # Should suggest Claude alternatives
        model_names = [
            rec.replay_params.get("model")
            for rec in analysis.recommendations
            if rec.replay_params and "model" in rec.replay_params
        ]
        assert any("claude" in model.lower() for model in model_names if model)

    def test_confidence_scores(self):
        """Test that recommendations include confidence scores."""
        span_data = {
            "attributes": {
                "error.message": "Rate limit exceeded",
                "error.status_code": 429,
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        # All recommendations should have confidence between 0 and 1
        for rec in analysis.recommendations:
            assert 0.0 <= rec.confidence <= 1.0

    def test_cost_impact_estimates(self):
        """Test that cost-related recommendations include impact estimates."""
        span_data = {
            "attributes": {
                "error.message": "Rate limit exceeded",
                "error.status_code": 429,
                "llm.model": "gpt-4",
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        # Model switch recommendations should have cost impact
        model_recs = [
            rec
            for rec in analysis.recommendations
            if rec.replay_params and "model" in rec.replay_params
        ]
        if model_recs:
            assert any(rec.estimated_cost_impact is not None for rec in model_recs)


class TestErrorContext:
    """Tests for error context extraction."""

    def test_context_includes_model(self):
        """Test that context includes model information."""
        span_data = {
            "attributes": {
                "error.message": "Some error",
                "llm.model": "gpt-4",
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        assert "llm.model" in analysis.context
        assert analysis.context["llm.model"] == "gpt-4"

    def test_context_includes_tokens(self):
        """Test that context includes token usage."""
        span_data = {
            "attributes": {
                "error.message": "Some error",
                "llm.prompt_tokens": 100,
                "llm.completion_tokens": 50,
                "llm.total_tokens": 150,
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        assert "llm.prompt_tokens" in analysis.context
        assert analysis.context["llm.prompt_tokens"] == 100

    def test_context_includes_duration(self):
        """Test that context includes duration."""
        span_data = {
            "attributes": {"error.message": "Some error"},
            "duration_ms": 1234.5,
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        assert "duration_ms" in analysis.context
        assert analysis.context["duration_ms"] == 1234.5


class TestSeverityCalculation:
    """Tests for error severity calculation."""

    def test_auth_failure_critical(self):
        """Test that auth failures are marked as critical."""
        span_data = {
            "attributes": {
                "error.message": "Invalid API key",
                "error.status_code": 401,
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        assert analysis.severity == ErrorSeverity.CRITICAL

    def test_rate_limit_high_or_medium(self):
        """Test that rate limits are high or medium severity."""
        span_data = {
            "attributes": {
                "error.message": "Rate limit exceeded",
                "error.status_code": 429,
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        assert analysis.severity in [ErrorSeverity.HIGH, ErrorSeverity.MEDIUM]

    def test_network_error_medium(self):
        """Test that network errors are typically medium severity."""
        span_data = {
            "attributes": {
                "error.message": "Connection timeout",
                "error.type": "TimeoutError",
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        assert analysis.severity in [ErrorSeverity.MEDIUM, ErrorSeverity.LOW]


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_span_data(self):
        """Test handling of empty span data."""
        span_data = {}

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        # Should still return valid analysis
        assert analysis.category == ErrorCategory.UNKNOWN
        assert analysis.error_message == ""
        assert isinstance(analysis.recommendations, list)

    def test_missing_attributes(self):
        """Test handling of missing attributes."""
        span_data = {"attributes": {}}

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        assert analysis.category == ErrorCategory.UNKNOWN
        assert isinstance(analysis.recommendations, list)

    def test_none_error_code(self):
        """Test handling of None error code."""
        span_data = {
            "attributes": {
                "error.message": "Some error",
                "error.status_code": None,
            }
        }

        analysis = ErrorAnalyzer.analyze_span_error(span_data)

        assert analysis.error_code is None
        assert isinstance(analysis.recommendations, list)
