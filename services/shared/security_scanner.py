"""
Security Scanner Module

Detects prompt injection attempts and PII/data leakage in LLM trace data.
Provides regex and heuristic-based scanning for security issues in
prompts (inputs) and completions (outputs).

Pattern: follows error_analyzer.py structure (str Enums, dataclasses, static methods).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class SecurityFindingType(str, Enum):
    """Types of security findings."""

    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK_ATTEMPT = "jailbreak_attempt"
    ROLE_CONFUSION = "role_confusion"
    ENCODED_INJECTION = "encoded_injection"
    DELIMITER_INJECTION = "delimiter_injection"
    PII_EMAIL = "pii_email"
    PII_PHONE = "pii_phone"
    PII_SSN = "pii_ssn"
    PII_CREDIT_CARD = "pii_credit_card"
    API_KEY_LEAK = "api_key_leak"


class SecuritySeverity(str, Enum):
    """Severity levels for security findings."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# Severity ordering for comparison
_SEVERITY_ORDER = {
    SecuritySeverity.LOW: 0,
    SecuritySeverity.MEDIUM: 1,
    SecuritySeverity.HIGH: 2,
    SecuritySeverity.CRITICAL: 3,
}


@dataclass
class SecurityFinding:
    """A single security finding from span analysis."""

    finding_type: SecurityFindingType
    severity: SecuritySeverity
    confidence: float  # 0.0-1.0
    matched_text: str  # Truncated/redacted excerpt
    pattern_name: str  # e.g. "instruction_override", "ssn_pattern"
    location: str  # "prompt" or "response"
    remediation: str


@dataclass
class SecurityAnalysis:
    """Complete security analysis of a span."""

    span_id: str
    findings: list[SecurityFinding] = field(default_factory=list)
    overall_severity: SecuritySeverity = SecuritySeverity.LOW
    overall_confidence: float = 0.0
    scanned_at: str = ""


def _truncate(text: str, max_len: int = 80) -> str:
    """Truncate text for safe display in findings."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# ---------------------------------------------------------------------------
# Prompt Injection Detector
# ---------------------------------------------------------------------------

# Pre-compiled pattern groups: (compiled_regex, pattern_name, finding_type, severity, confidence, remediation)
_INJECTION_PATTERNS: list[tuple[re.Pattern, str, SecurityFindingType, SecuritySeverity, float, str]] = []


def _build_injection_patterns() -> list[tuple[re.Pattern, str, SecurityFindingType, SecuritySeverity, float, str]]:
    """Build and compile injection detection patterns."""
    patterns = []

    # --- Instruction override (CRITICAL) ---
    for name, regex in [
        ("ignore_previous", r"ignore\s+(all\s+)?previous\s+instructions?"),
        ("disregard_previous", r"disregard\s+(all\s+)?previous\s+(commands?|instructions?|prompts?)"),
        ("forget_everything", r"forget\s+(everything|all\s+previous|your\s+instructions?)"),
        ("new_instructions", r"new\s+instructions?\s*:"),
        ("override_system", r"override\s+(system|safety)\s+(prompt|instructions?)"),
        ("do_not_follow", r"do\s+not\s+follow\s+(your|the)\s+(previous|original)\s+instructions?"),
    ]:
        patterns.append((
            re.compile(regex, re.IGNORECASE),
            name,
            SecurityFindingType.PROMPT_INJECTION,
            SecuritySeverity.CRITICAL,
            0.90,
            "Sanitize user input before sending to LLM. Consider input validation and prompt hardening.",
        ))

    # --- Jailbreak attempts (HIGH) ---
    for name, regex in [
        ("dan_mode", r"\bDAN\s+mode\b"),
        ("you_are_dan", r"you\s+are\s+now\s+(a\s+)?DAN"),
        ("developer_mode", r"developer\s+mode\s+(enabled|activated|on)"),
        ("act_without_restrictions", r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions?"),
        ("pretend_no_guidelines", r"pretend\s+(you\s+)?(are|have)\s+(no|without)\s+(guidelines?|restrictions?|rules?)"),
        ("hypothetical_scenario", r"in\s+a\s+hypothetical\s+scenario\s+where\s+you\s+(have|had)\s+no\s+(rules?|restrictions?)"),
    ]:
        patterns.append((
            re.compile(regex, re.IGNORECASE),
            name,
            SecurityFindingType.JAILBREAK_ATTEMPT,
            SecuritySeverity.HIGH,
            0.85,
            "Implement jailbreak detection in your prompt pipeline. Consider using a moderation layer.",
        ))

    # --- Role confusion (HIGH) ---
    for name, regex in [
        ("system_role_inject", r"system\s*:\s*you\s+are"),
        ("system_tag_inject", r"<\|system\|>"),
        ("system_bracket_inject", r"\[SYSTEM\]"),
        ("system_xml_inject", r"<system>"),
        ("assistant_role_inject", r"<\|assistant\|>"),
    ]:
        patterns.append((
            re.compile(regex, re.IGNORECASE),
            name,
            SecurityFindingType.ROLE_CONFUSION,
            SecuritySeverity.HIGH,
            0.80,
            "Escape or strip role-boundary tokens from user input before prompt construction.",
        ))

    # --- Encoded injection (MEDIUM) ---
    for name, regex in [
        ("base64_decode", r"base64[\s._-]*(decode|encoded?)"),
        ("eval_call", r"\beval\s*\("),
        ("exec_call", r"\bexec\s*\("),
        ("dunder_import", r"__import__\s*\("),
        ("hex_encode_trick", r"\\x[0-9a-f]{2}\\x[0-9a-f]{2}"),
    ]:
        patterns.append((
            re.compile(regex, re.IGNORECASE),
            name,
            SecurityFindingType.ENCODED_INJECTION,
            SecuritySeverity.MEDIUM,
            0.70,
            "Filter encoded content in user inputs. Consider a content-type allowlist.",
        ))

    # --- Delimiter injection (MEDIUM) ---
    for name, regex in [
        ("closing_prompt_tag", r"</prompt>"),
        ("end_marker", r"---\s*END\s*---"),
        ("triple_backtick_boundary", r"```\s*(system|instructions?|prompt)\s*```"),
        ("xml_closing_tags", r"</?(instructions?|context|system_prompt)>"),
    ]:
        patterns.append((
            re.compile(regex, re.IGNORECASE),
            name,
            SecurityFindingType.DELIMITER_INJECTION,
            SecuritySeverity.MEDIUM,
            0.65,
            "Use robust prompt delimiters that cannot be mimicked by user input. Consider XML-safe escaping.",
        ))

    return patterns


_INJECTION_PATTERNS = _build_injection_patterns()


class PromptInjectionDetector:
    """Detects prompt injection attempts in LLM inputs."""

    @staticmethod
    def detect(text: str, location: str = "prompt") -> list[SecurityFinding]:
        """Detect injection patterns in text.

        Args:
            text: The text to scan (prompt or response).
            location: "prompt" or "response".

        Returns:
            List of SecurityFinding objects for detected patterns.
        """
        if not text or not text.strip():
            return []

        findings: list[SecurityFinding] = []
        seen_types: set[str] = set()  # Deduplicate by pattern name

        for compiled, pattern_name, finding_type, severity, confidence, remediation in _INJECTION_PATTERNS:
            if pattern_name in seen_types:
                continue

            match = compiled.search(text)
            if match:
                seen_types.add(pattern_name)
                findings.append(SecurityFinding(
                    finding_type=finding_type,
                    severity=severity,
                    confidence=confidence,
                    matched_text=_truncate(match.group(0)),
                    pattern_name=pattern_name,
                    location=location,
                    remediation=remediation,
                ))

        return findings


# ---------------------------------------------------------------------------
# Data Leakage Detector
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CC_RE = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")

_API_KEY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"AKIA[0-9A-Z]{16}"), "aws_access_key"),
    (re.compile(r"sk_live_[0-9a-zA-Z]{24,}"), "stripe_secret_key"),
    (re.compile(r"ghp_[0-9a-zA-Z]{36}"), "github_pat"),
    (re.compile(r"sk-[0-9a-zA-Z]{20,}"), "openai_api_key"),
    (re.compile(r"xoxb-[0-9a-zA-Z-]+"), "slack_bot_token"),
    (re.compile(r"AIza[0-9A-Za-z_-]{35}"), "google_api_key"),
]


def _luhn_check(number: str) -> bool:
    """Validate a credit card number with the Luhn algorithm."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _redact_api_key(key: str) -> str:
    """Redact API key, keeping first 8 chars."""
    if len(key) <= 8:
        return key[:4] + "..."
    return key[:8] + "..."


class DataLeakageDetector:
    """Detects PII and sensitive data leakage in LLM outputs."""

    @staticmethod
    def detect(text: str, location: str = "response") -> list[SecurityFinding]:
        """Detect PII and secrets in text.

        Args:
            text: The text to scan.
            location: "prompt" or "response".

        Returns:
            List of SecurityFinding objects for detected data leakage.
        """
        if not text or not text.strip():
            return []

        findings: list[SecurityFinding] = []

        # --- SSN (CRITICAL) ---
        for match in _SSN_RE.finditer(text):
            findings.append(SecurityFinding(
                finding_type=SecurityFindingType.PII_SSN,
                severity=SecuritySeverity.CRITICAL,
                confidence=0.95,
                matched_text="XXX-XX-XXXX",
                pattern_name="ssn",
                location=location,
                remediation="CRITICAL: SSN detected. Purge from logs and review data handling. Add output filtering.",
            ))

        # --- Credit Card (CRITICAL, with Luhn validation) ---
        for match in _CC_RE.finditer(text):
            raw = match.group(0)
            digits_only = re.sub(r"[\s-]", "", raw)
            if _luhn_check(digits_only):
                last_four = digits_only[-4:]
                findings.append(SecurityFinding(
                    finding_type=SecurityFindingType.PII_CREDIT_CARD,
                    severity=SecuritySeverity.CRITICAL,
                    confidence=0.95,
                    matched_text=f"****-****-****-{last_four}",
                    pattern_name="credit_card",
                    location=location,
                    remediation="CRITICAL: Credit card number detected. Purge from logs and add PCI-compliant output filtering.",
                ))

        # --- Email (MEDIUM) ---
        seen_emails: set[str] = set()
        for match in _EMAIL_RE.finditer(text):
            email = match.group(0).lower()
            if email not in seen_emails:
                seen_emails.add(email)
                findings.append(SecurityFinding(
                    finding_type=SecurityFindingType.PII_EMAIL,
                    severity=SecuritySeverity.MEDIUM,
                    confidence=0.90,
                    matched_text=_truncate(match.group(0)),
                    pattern_name="email_address",
                    location=location,
                    remediation="Redact email addresses from LLM outputs. Consider PII masking in your pipeline.",
                ))

        # --- Phone (MEDIUM) ---
        seen_phones: set[str] = set()
        for match in _PHONE_RE.finditer(text):
            normalized = re.sub(r"[\s().-]", "", match.group(0))
            if normalized not in seen_phones:
                seen_phones.add(normalized)
                findings.append(SecurityFinding(
                    finding_type=SecurityFindingType.PII_PHONE,
                    severity=SecuritySeverity.MEDIUM,
                    confidence=0.80,
                    matched_text=_truncate(match.group(0)),
                    pattern_name="phone_number",
                    location=location,
                    remediation="Redact phone numbers from LLM outputs. Consider PII masking in your pipeline.",
                ))

        # --- API Keys (HIGH) ---
        for pattern, key_type in _API_KEY_PATTERNS:
            for match in pattern.finditer(text):
                findings.append(SecurityFinding(
                    finding_type=SecurityFindingType.API_KEY_LEAK,
                    severity=SecuritySeverity.HIGH,
                    confidence=0.92,
                    matched_text=_redact_api_key(match.group(0)),
                    pattern_name=key_type,
                    location=location,
                    remediation=f"API key ({key_type}) detected. Rotate immediately and add secret scanning to your pipeline.",
                ))

        return findings


# ---------------------------------------------------------------------------
# SecurityScanner Facade
# ---------------------------------------------------------------------------


class SecurityScanner:
    """Main facade for security analysis of LLM spans."""

    @staticmethod
    def analyze_span(span_data: dict[str, Any]) -> SecurityAnalysis:
        """Analyze a span for security issues.

        Args:
            span_data: Span dictionary with attributes.
                Expected attribute keys: llm.prompt / gen_ai.prompt,
                llm.response / gen_ai.completion.

        Returns:
            SecurityAnalysis with all findings.
        """
        attributes = span_data.get("attributes", {})
        if isinstance(attributes, str):
            try:
                attributes = json.loads(attributes)
            except (json.JSONDecodeError, TypeError):
                attributes = {}

        # Extract prompt and response text
        prompt = (
            attributes.get("llm.prompt", "")
            or attributes.get("gen_ai.prompt", "")
            or ""
        )
        response = (
            attributes.get("llm.response", "")
            or attributes.get("gen_ai.completion", "")
            or ""
        )

        findings: list[SecurityFinding] = []

        # Run injection detection on prompt
        if prompt:
            findings.extend(PromptInjectionDetector.detect(prompt, location="prompt"))
            # Also check prompt for PII (user may paste sensitive data)
            findings.extend(DataLeakageDetector.detect(prompt, location="prompt"))

        # Run leakage detection on response
        if response:
            findings.extend(DataLeakageDetector.detect(response, location="response"))
            # Also check response for injection echoing
            findings.extend(PromptInjectionDetector.detect(response, location="response"))

        # Calculate overall severity and confidence
        overall_severity = SecuritySeverity.LOW
        overall_confidence = 0.0

        for f in findings:
            if _SEVERITY_ORDER[f.severity] > _SEVERITY_ORDER[overall_severity]:
                overall_severity = f.severity
            if f.confidence > overall_confidence:
                overall_confidence = f.confidence

        return SecurityAnalysis(
            span_id=span_data.get("span_id", ""),
            findings=findings,
            overall_severity=overall_severity,
            overall_confidence=overall_confidence,
            scanned_at=datetime.now(timezone.utc).isoformat(),
        )
