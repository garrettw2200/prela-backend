"""
Error Explanation Module

Uses GPT-4o-mini to generate natural language explanations for agent errors,
providing developers with clear "why" and "what to do" guidance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class ErrorExplanation:
    """Natural language explanation of an error with actionable guidance."""

    why_it_happened: str  # 2-3 sentence explanation
    what_to_do: str  # Step-by-step guidance
    related_patterns: list[str]  # Similar error patterns
    estimated_fix_time: str  # "< 1 minute", "5-10 minutes", etc.


class ErrorExplainer:
    """Generates AI-powered explanations for agent errors."""

    def __init__(self, openai_api_key: str | None = None):
        """
        Initialize the error explainer.

        Args:
            openai_api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        """
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required for error explanations")

        self.client = OpenAI(api_key=api_key)

    def explain_error(
        self, category: str, severity: str, error_message: str, context: dict[str, Any]
    ) -> ErrorExplanation:
        """
        Generate natural language explanation for an error.

        Args:
            category: Error category (rate_limit, auth_failure, etc.)
            severity: Error severity (CRITICAL, HIGH, MEDIUM, LOW)
            error_message: Full error message from span
            context: Additional context (model, tokens, provider, etc.)

        Returns:
            ErrorExplanation with why/what/patterns/time
        """
        # Build prompt for GPT-4o-mini
        prompt = self._build_prompt(category, severity, error_message, context)

        # Call OpenAI API
        response = self._call_openai(prompt)

        # Parse response
        return self._parse_response(response, category)

    def _build_prompt(
        self, category: str, severity: str, error_message: str, context: dict[str, Any]
    ) -> str:
        """Build prompt for GPT-4o-mini."""
        # Extract relevant context
        model = context.get("llm.model", "unknown")
        provider = context.get("llm.vendor", "unknown")
        tokens_used = context.get("llm.total_tokens")
        token_limit = context.get("llm.max_tokens")

        # Category-specific context
        category_context = self._get_category_context(category, context)

        prompt = f"""You are an AI agent debugging assistant helping developers fix errors.

Analyze this error and provide concise, actionable guidance:

**Error Details:**
- Category: {category}
- Severity: {severity}
- Message: {error_message}
- Model: {model}
- Provider: {provider}
{category_context}

**Your Task:**
Provide TWO things in your response:

1. WHY_IT_HAPPENED (2-3 sentences):
   - Explain the root cause in simple terms
   - Avoid jargon, be developer-friendly
   - Focus on the immediate cause

2. WHAT_TO_DO (numbered steps, max 3 steps):
   - Specific, actionable steps
   - Prioritize quickest fixes first
   - Include exact commands or parameters when possible

Be concise, friendly, and practical. Don't apologize or be overly formal.

Format your response as:
WHY: <explanation>
WHAT:
1. <step 1>
2. <step 2>
3. <step 3>
"""
        return prompt

    def _get_category_context(self, category: str, context: dict[str, Any]) -> str:
        """Get category-specific context for prompt."""
        if category == "rate_limit":
            tokens = context.get("llm.total_tokens", "unknown")
            return f"- Tokens: {tokens}"
        elif category == "token_limit":
            max_tokens = context.get("llm.max_tokens", "unknown")
            return f"- Token limit: {max_tokens}"
        elif category == "auth_failure":
            return "- Authentication issue detected"
        elif category == "model_not_found":
            model = context.get("llm.model", "unknown")
            return f"- Requested model: {model}"
        elif category == "network_error":
            return "- Network connectivity issue"
        elif category == "tool_error":
            tool_name = context.get("tool.name", "unknown")
            return f"- Tool: {tool_name}"
        elif category == "retrieval_error":
            return "- Vector database or retrieval issue"
        else:
            return ""

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI GPT-4o-mini API."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful debugging assistant for AI agents. Be concise and practical.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Low temperature for consistent, factual responses
                max_tokens=300,  # Keep explanations concise
            )

            return response.choices[0].message.content or ""
        except Exception as e:
            # Fallback to generic explanation if LLM fails
            return self._get_fallback_explanation()

    def _get_fallback_explanation(self) -> str:
        """Fallback explanation if LLM call fails."""
        return """WHY: The agent encountered an error during execution. This could be due to various factors including API issues, configuration problems, or resource constraints.
WHAT:
1. Check the error message above for specific details
2. Review your agent configuration and API keys
3. Try running the replay with suggested fixes"""

    def _parse_response(self, response: str, category: str) -> ErrorExplanation:
        """Parse GPT-4o-mini response into ErrorExplanation."""
        # Split by WHY: and WHAT:
        why_section = ""
        what_section = ""

        if "WHY:" in response:
            parts = response.split("WHY:", 1)
            if len(parts) > 1:
                why_part = parts[1]
                if "WHAT:" in why_part:
                    why_section, what_section = why_part.split("WHAT:", 1)
                else:
                    why_section = why_part

        # Clean up sections
        why_it_happened = why_section.strip()
        what_to_do = what_section.strip()

        # Generate related patterns based on category
        related_patterns = self._get_related_patterns(category)

        # Estimate fix time based on category
        estimated_fix_time = self._estimate_fix_time(category)

        return ErrorExplanation(
            why_it_happened=why_it_happened or "Error analysis unavailable.",
            what_to_do=what_to_do or "Review error details and try suggested fixes.",
            related_patterns=related_patterns,
            estimated_fix_time=estimated_fix_time,
        )

    def _get_related_patterns(self, category: str) -> list[str]:
        """Get related patterns for error category."""
        patterns = {
            "rate_limit": [
                "Common during peak usage hours",
                "Often resolved by switching to cheaper models",
                "Can be prevented with exponential backoff",
            ],
            "auth_failure": [
                "Check API key is not expired",
                "Verify environment variables are set",
                "Ensure correct provider credentials",
            ],
            "token_limit": [
                "Reduce prompt length or max_tokens",
                "Switch to models with larger context windows",
                "Implement prompt compression techniques",
            ],
            "model_not_found": [
                "Verify model name spelling",
                "Check model availability for your account",
                "Some models require special access",
            ],
            "network_error": [
                "Temporary connectivity issues",
                "Provider API may be experiencing downtime",
                "Retry with exponential backoff usually works",
            ],
            "tool_error": [
                "Tool configuration may be incorrect",
                "Tool function may have bugs",
                "Check tool input/output schemas",
            ],
            "retrieval_error": [
                "Vector database connection issues",
                "Check vector DB credentials",
                "Verify collection/index exists",
            ],
        }
        return patterns.get(category, ["Review error details carefully"])

    def _estimate_fix_time(self, category: str) -> str:
        """Estimate time to fix based on category."""
        quick_fixes = ["rate_limit", "model_not_found", "token_limit"]
        medium_fixes = ["network_error", "auth_failure"]

        if category in quick_fixes:
            return "< 1 minute"
        elif category in medium_fixes:
            return "5-10 minutes"
        else:
            return "10-30 minutes"
