"""Background security scanning service.

Periodically scans new LLM spans for prompt injection and PII leakage,
storing results in the analysis_results ClickHouse table.

Pattern: follows data_source_sync.py (async loop with sleep, error per iteration).
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

from shared import settings
from shared.clickhouse import get_clickhouse_client
from shared.security_scanner import SecurityScanner

logger = logging.getLogger(__name__)

# Default: scan every 24 hours
SCAN_INTERVAL_HOURS = 24
# Max spans to scan per run (avoid OOM on large datasets)
SCAN_BATCH_LIMIT = 5000


async def scan_recent_traces() -> dict[str, int]:
    """Scan LLM spans from the last 24 hours for security issues.

    Deduplicates against already-scanned traces in analysis_results.

    Returns:
        Dict with spans_scanned and findings_count.
    """
    client = get_clickhouse_client()

    since = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Get trace_ids already scanned (for deduplication)
    already_scanned = client.query(
        """
        SELECT DISTINCT trace_id
        FROM analysis_results
        WHERE analysis_type = 'security'
          AND created_at >= %(since)s
        """,
        parameters={"since": since},
    )
    scanned_trace_ids = {row[0] for row in already_scanned.result_rows}

    # Fetch recent LLM spans
    spans_result = client.query(
        """
        SELECT span_id, trace_id, project_id, attributes
        FROM spans
        WHERE span_type = 'llm'
          AND started_at >= %(since)s
        ORDER BY started_at DESC
        LIMIT %(limit)s
        """,
        parameters={"since": since, "limit": SCAN_BATCH_LIMIT},
    )

    results_to_insert: list[dict] = []
    spans_scanned = 0

    for row in spans_result.result_rows:
        span_id, trace_id, project_id = row[0], row[1], row[2]

        # Skip if this trace was already scanned
        if trace_id in scanned_trace_ids:
            continue

        try:
            attributes = json.loads(row[3]) if row[3] else {}
        except (json.JSONDecodeError, TypeError):
            attributes = {}

        span_data = {
            "span_id": span_id,
            "attributes": attributes,
        }

        analysis = SecurityScanner.analyze_span(span_data)
        spans_scanned += 1

        # Only store spans with findings
        if analysis.findings:
            results_to_insert.append({
                "result_id": str(uuid.uuid4()),
                "trace_id": trace_id,
                "project_id": project_id,
                "analysis_type": "security",
                "result": json.dumps({
                    "span_id": analysis.span_id,
                    "findings": [
                        {
                            "finding_type": f.finding_type.value,
                            "severity": f.severity.value,
                            "confidence": f.confidence,
                            "matched_text": f.matched_text,
                            "pattern_name": f.pattern_name,
                            "location": f.location,
                            "remediation": f.remediation,
                        }
                        for f in analysis.findings
                    ],
                    "overall_severity": analysis.overall_severity.value,
                    "overall_confidence": analysis.overall_confidence,
                }),
                "score": analysis.overall_confidence,
            })

            # Mark trace as scanned to avoid re-scanning other spans of same trace
            scanned_trace_ids.add(trace_id)

    # Batch insert to ClickHouse
    if results_to_insert:
        client.insert(
            "analysis_results",
            [list(r.values()) for r in results_to_insert],
            column_names=list(results_to_insert[0].keys()),
        )

    logger.info(
        f"[SECURITY] Scanned {spans_scanned} spans, "
        f"found {len(results_to_insert)} traces with security issues"
    )

    return {
        "spans_scanned": spans_scanned,
        "findings_count": len(results_to_insert),
    }


async def background_security_scan_loop() -> None:
    """Background loop that runs security scans periodically.

    Runs every SCAN_INTERVAL_HOURS (default 24h).
    """
    logger.info(
        f"[SECURITY] Background scan loop started (interval={SCAN_INTERVAL_HOURS}h)"
    )

    # Initial delay to let the app fully start
    await asyncio.sleep(60)

    while True:
        try:
            result = await scan_recent_traces()
            logger.info(f"[SECURITY] Scan complete: {result}")
        except Exception as e:
            logger.error(f"[SECURITY] Background scan error: {e}", exc_info=True)

        await asyncio.sleep(SCAN_INTERVAL_HOURS * 3600)
