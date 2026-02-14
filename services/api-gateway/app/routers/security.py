"""
Security Scanning API Router

Provides on-demand trace security scanning and aggregated security
summary for the insights dashboard. Reads pre-computed results from
the analysis_results ClickHouse table (populated by background batch scan).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from shared import get_clickhouse_client
from ..auth import require_tier
from shared.security_scanner import SecurityScanner

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------


class SecurityFindingResponse(BaseModel):
    finding_type: str
    severity: str
    confidence: float
    matched_text: str
    pattern_name: str
    location: str
    remediation: str


class SecurityScanResponse(BaseModel):
    trace_id: str
    span_id: str
    findings: list[SecurityFindingResponse]
    overall_severity: str
    confidence: float
    scanned_at: str


class SecuritySummaryResponse(BaseModel):
    total_findings: int
    by_severity: dict[str, int]
    by_type: dict[str, int]
    recent_findings: list[SecurityScanResponse]
    time_window: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/traces/{trace_id}/scan",
    response_model=list[SecurityScanResponse],
)
async def scan_trace(
    trace_id: str,
    project_id: str = Query(..., description="Project ID"),
    user: dict = Depends(require_tier("pro")),
) -> list[SecurityScanResponse]:
    """Scan a specific trace for security issues on-demand.

    Runs the security scanner against all LLM spans in the trace.
    Does NOT persist results — use for real-time, interactive analysis.
    """
    try:
        client = get_clickhouse_client()

        spans_result = client.query(
            """
            SELECT span_id, name, attributes
            FROM spans
            WHERE trace_id = %(trace_id)s
              AND project_id = %(project_id)s
              AND span_type = 'llm'
            ORDER BY started_at ASC
            """,
            parameters={"trace_id": trace_id, "project_id": project_id},
        )

        scan_results: list[SecurityScanResponse] = []

        for row in spans_result.result_rows:
            span_data = {
                "span_id": row[0],
                "name": row[1],
                "attributes": json.loads(row[2]) if row[2] else {},
            }

            analysis = SecurityScanner.analyze_span(span_data)

            if analysis.findings:
                scan_results.append(SecurityScanResponse(
                    trace_id=trace_id,
                    span_id=analysis.span_id,
                    findings=[
                        SecurityFindingResponse(
                            finding_type=f.finding_type.value,
                            severity=f.severity.value,
                            confidence=f.confidence,
                            matched_text=f.matched_text,
                            pattern_name=f.pattern_name,
                            location=f.location,
                            remediation=f.remediation,
                        )
                        for f in analysis.findings
                    ],
                    overall_severity=analysis.overall_severity.value,
                    confidence=analysis.overall_confidence,
                    scanned_at=analysis.scanned_at,
                ))

        return scan_results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to scan trace {trace_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Security scan failed: {str(e)}")


@router.get("/summary", response_model=SecuritySummaryResponse)
async def get_security_summary(
    project_id: str = Query(..., description="Project ID"),
    time_window: str = Query("7d", description="Time window (7d, 30d, 90d)"),
    user: dict = Depends(require_tier("pro")),
) -> SecuritySummaryResponse:
    """Get aggregated security findings from pre-computed analysis results.

    Used by the insights dashboard SecuritySummary component.
    """
    days_map = {"7d": 7, "30d": 30, "90d": 90}
    days = days_map.get(time_window, 7)
    since_date = datetime.utcnow() - timedelta(days=days)
    since_str = since_date.strftime("%Y-%m-%d %H:%M:%S")

    try:
        client = get_clickhouse_client()

        results = client.query(
            """
            SELECT result_id, trace_id, result, score, created_at
            FROM analysis_results
            WHERE project_id = %(project_id)s
              AND analysis_type = 'security'
              AND created_at >= %(since)s
            ORDER BY created_at DESC
            LIMIT 200
            """,
            parameters={"project_id": project_id, "since": since_str},
        )

        by_severity: Counter[str] = Counter()
        by_type: Counter[str] = Counter()
        recent_findings: list[SecurityScanResponse] = []

        for row in results.result_rows:
            try:
                result_data = json.loads(row[2]) if row[2] else {}
            except (json.JSONDecodeError, TypeError):
                continue

            findings_raw = result_data.get("findings", [])

            for finding in findings_raw:
                sev = finding.get("severity", "MEDIUM")
                ftype = finding.get("finding_type", "unknown")
                by_severity[sev] += 1
                by_type[ftype] += 1

            # Collect recent examples (up to 5)
            if len(recent_findings) < 5 and findings_raw:
                recent_findings.append(SecurityScanResponse(
                    trace_id=row[1],
                    span_id=result_data.get("span_id", ""),
                    findings=[
                        SecurityFindingResponse(**f) for f in findings_raw
                    ],
                    overall_severity=result_data.get("overall_severity", "LOW"),
                    confidence=result_data.get("overall_confidence", 0.0),
                    scanned_at=str(row[4]),
                ))

        return SecuritySummaryResponse(
            total_findings=sum(by_severity.values()),
            by_severity=dict(by_severity),
            by_type=dict(by_type),
            recent_findings=recent_findings,
            time_window=time_window,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get security summary: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get security summary: {str(e)}"
        )
