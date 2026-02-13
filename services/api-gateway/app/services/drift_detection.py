"""Background drift detection service.

Periodically calculates baselines and detects anomalies in agent behavior,
creating drift alerts and sending notifications when drift is detected.

Pattern: follows security_scan.py (async loop with sleep, error per iteration).
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from shared import (
    get_clickhouse_client,
    send_email_notification,
    send_slack_notification,
    format_drift_alert_email,
    format_drift_alert_slack,
    settings,
)
from shared.anomaly_detector import AnomalyDetector, AnomalySeverity
from shared.baseline_calculator import BaselineCalculator

logger = logging.getLogger(__name__)

# Intervals
DRIFT_CHECK_INTERVAL_MINUTES = getattr(settings, "drift_check_interval_minutes", 15)
BASELINE_REFRESH_HOURS = getattr(settings, "drift_baseline_refresh_hours", 1)

# Track baseline refresh timing
_last_baseline_refresh: datetime | None = None


async def _get_active_projects() -> list[str]:
    """Get projects with recent span activity (last 24h)."""
    client = get_clickhouse_client()
    result = client.query(
        """
        SELECT DISTINCT project_id
        FROM spans
        WHERE started_at >= now() - INTERVAL 24 HOUR
        """
    )
    return [row[0] for row in result.result_rows if row[0]]


async def _get_recent_alert_ids(
    client: Any, project_id: str, hours: int = 1
) -> set[str]:
    """Get agent names that already have recent alerts (for dedup)."""
    result = client.query(
        """
        SELECT DISTINCT agent_name
        FROM drift_alerts
        WHERE project_id = %(project_id)s
          AND status = 'active'
          AND detected_at >= now() - INTERVAL %(hours)s HOUR
        """,
        parameters={"project_id": project_id, "hours": hours},
    )
    return {row[0] for row in result.result_rows}


async def _send_notifications(
    client: Any,
    project_id: str,
    agent_name: str,
    severity: str,
    anomaly_dicts: list[dict[str, Any]],
) -> None:
    """Match alert rules and send email/Slack notifications."""
    try:
        result = client.query(
            """
            SELECT
                rule_id, name, notify_email, email_addresses,
                notify_slack, slack_webhook_url, slack_channel
            FROM alert_rules
            WHERE project_id = %(project_id)s
              AND enabled = true
              AND (agent_name IS NULL OR agent_name = %(agent_name)s)
              AND severity_threshold <= %(severity)s
            """,
            parameters={
                "project_id": project_id,
                "agent_name": agent_name,
                "severity": severity,
            },
        )

        if not result.result_rows:
            return

        for row in result.result_rows:
            rule_name = row[1]
            notify_email = row[2]
            email_addresses = row[3]
            notify_slack = row[4]
            slack_webhook_url = row[5]
            slack_channel = row[6]

            logger.info(f"[DRIFT] Matched alert rule: {rule_name}")

            if notify_email and email_addresses:
                html_body, text_body = format_drift_alert_email(
                    agent_name=agent_name,
                    severity=severity,
                    anomalies=anomaly_dicts,
                )
                await send_email_notification(
                    to_addresses=email_addresses,
                    subject=f"[Prela] Drift Alert: {agent_name}",
                    body_html=html_body,
                    body_text=text_body,
                )

            if notify_slack and slack_webhook_url:
                slack_payload = format_drift_alert_slack(
                    agent_name=agent_name,
                    severity=severity,
                    anomalies=anomaly_dicts,
                )
                await send_slack_notification(
                    webhook_url=slack_webhook_url,
                    message=slack_payload,
                    channel=slack_channel,
                )

    except Exception as e:
        logger.error(f"[DRIFT] Failed to send notifications: {e}")


async def detect_drift_for_project(project_id: str) -> dict[str, int]:
    """Run drift detection for a single project.

    Returns:
        Dict with agents_checked, anomalies_found, alerts_created.
    """
    client = get_clickhouse_client()
    calculator = BaselineCalculator(client)
    detector = AnomalyDetector(client, sensitivity=2.0)

    # Get agents that already have recent alerts (dedup)
    recently_alerted = await _get_recent_alert_ids(client, project_id)

    # Get agents with recent activity
    agents_result = client.query(
        """
        SELECT DISTINCT
            JSONExtractString(attributes, 'agent.name') AS agent_name,
            service_name
        FROM spans
        WHERE project_id = %(project_id)s
          AND span_type = 'agent'
          AND started_at >= now() - INTERVAL 24 HOUR
          AND JSONHas(attributes, 'agent.name')
        """,
        parameters={"project_id": project_id},
    )

    agents = [(row[0], row[1]) for row in agents_result.result_rows if row[0]]

    agents_checked = 0
    anomalies_found = 0
    alerts_created = 0

    for agent_name, service_name in agents:
        # Skip agents with recent alerts
        if agent_name in recently_alerted:
            continue

        agents_checked += 1

        try:
            # Get latest baseline
            baseline = calculator.get_latest_baseline(
                project_id, agent_name, service_name
            )
            if not baseline:
                logger.debug(
                    f"[DRIFT] No baseline for agent '{agent_name}', skipping"
                )
                continue

            # Detect anomalies
            anomalies = detector.detect_anomalies(
                project_id, agent_name, service_name, baseline, lookback_hours=24
            )

            if not anomalies:
                continue

            anomalies_found += len(anomalies)

            # Analyze root causes
            root_causes = detector.analyze_root_causes(
                project_id, agent_name, service_name, anomalies, lookback_hours=24
            )

            # Determine max severity
            severity_order = {
                AnomalySeverity.LOW: 1,
                AnomalySeverity.MEDIUM: 2,
                AnomalySeverity.HIGH: 3,
                AnomalySeverity.CRITICAL: 4,
            }
            max_severity = max(
                (a["severity"] for a in anomalies),
                key=lambda s: severity_order.get(s, 0),
            )

            # Build anomaly dicts for storage
            anomaly_dicts = [
                {
                    "metric_name": a["metric_name"],
                    "current_value": a["current_value"],
                    "baseline_mean": a.get("baseline_mean", 0),
                    "change_percent": a["change_percent"],
                    "severity": a["severity"].value
                    if hasattr(a["severity"], "value")
                    else str(a["severity"]),
                    "direction": a["direction"],
                    "unit": a["unit"],
                    "sample_size": a.get("sample_size", 0),
                }
                for a in anomalies
            ]

            root_cause_dicts = [
                {
                    "type": rc["type"],
                    "description": rc["description"],
                    "confidence": rc["confidence"],
                }
                for rc in root_causes
            ]

            # Insert drift alert
            alert_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)

            severity_str = (
                max_severity.value
                if hasattr(max_severity, "value")
                else str(max_severity)
            )

            client.insert(
                "drift_alerts",
                [
                    [
                        alert_id,
                        project_id,
                        agent_name,
                        service_name,
                        baseline["baseline_id"],
                        now,
                        severity_str,
                        "active",
                        json.dumps(anomaly_dicts),
                        json.dumps(root_cause_dicts),
                    ]
                ],
                column_names=[
                    "alert_id",
                    "project_id",
                    "agent_name",
                    "service_name",
                    "baseline_id",
                    "detected_at",
                    "severity",
                    "status",
                    "anomalies",
                    "root_causes",
                ],
            )

            # Store in analysis_results for insights integration
            client.insert(
                "analysis_results",
                [
                    [
                        str(uuid.uuid4()),
                        alert_id,  # use alert_id as trace_id for drift results
                        project_id,
                        "drift",
                        json.dumps(
                            {
                                "agent_name": agent_name,
                                "anomalies": anomaly_dicts,
                                "root_causes": root_cause_dicts,
                                "severity": severity_str,
                            }
                        ),
                        severity_order.get(max_severity, 0) / 4.0,
                    ]
                ],
                column_names=[
                    "result_id",
                    "trace_id",
                    "project_id",
                    "analysis_type",
                    "result",
                    "score",
                ],
            )

            alerts_created += 1

            logger.info(
                f"[DRIFT] Alert created for agent '{agent_name}' "
                f"in project '{project_id}': "
                f"{len(anomalies)} anomalies, severity={severity_str}"
            )

            # Send notifications
            await _send_notifications(
                client, project_id, agent_name, severity_str, anomaly_dicts
            )

        except Exception as e:
            logger.error(
                f"[DRIFT] Error checking agent '{agent_name}' "
                f"in project '{project_id}': {e}",
                exc_info=True,
            )

    return {
        "agents_checked": agents_checked,
        "anomalies_found": anomalies_found,
        "alerts_created": alerts_created,
    }


async def detect_drift_all_projects() -> dict[str, Any]:
    """Run drift detection across all active projects.

    Returns:
        Dict with project count and aggregate metrics.
    """
    projects = await _get_active_projects()

    total_agents = 0
    total_anomalies = 0
    total_alerts = 0
    errors = 0

    for project_id in projects:
        try:
            result = await detect_drift_for_project(project_id)
            total_agents += result["agents_checked"]
            total_anomalies += result["anomalies_found"]
            total_alerts += result["alerts_created"]
        except Exception as e:
            errors += 1
            logger.error(
                f"[DRIFT] Error processing project '{project_id}': {e}",
                exc_info=True,
            )

    return {
        "projects_checked": len(projects),
        "agents_checked": total_agents,
        "anomalies_found": total_anomalies,
        "alerts_created": total_alerts,
        "errors": errors,
    }


async def refresh_baselines_all_projects() -> dict[str, Any]:
    """Refresh baselines for all active projects.

    Returns:
        Dict with project count and baselines calculated.
    """
    projects = await _get_active_projects()

    total_baselines = 0
    errors = 0

    for project_id in projects:
        try:
            client = get_clickhouse_client()
            calculator = BaselineCalculator(client)
            count = calculator.calculate_all_baselines(project_id)
            total_baselines += count
        except Exception as e:
            errors += 1
            logger.error(
                f"[DRIFT] Error refreshing baselines for project '{project_id}': {e}",
                exc_info=True,
            )

    return {
        "projects_checked": len(projects),
        "baselines_calculated": total_baselines,
        "errors": errors,
    }


async def background_drift_detection_loop() -> None:
    """Background loop that runs drift detection periodically.

    Baselines are refreshed every BASELINE_REFRESH_HOURS.
    Anomaly detection runs every DRIFT_CHECK_INTERVAL_MINUTES.
    """
    global _last_baseline_refresh

    logger.info(
        f"[DRIFT] Background loop started "
        f"(check interval={DRIFT_CHECK_INTERVAL_MINUTES}min, "
        f"baseline refresh={BASELINE_REFRESH_HOURS}h)"
    )

    # Initial delay to let the app fully start
    await asyncio.sleep(60)

    iteration = 0

    while True:
        try:
            now = datetime.now(timezone.utc)

            # Refresh baselines if needed
            should_refresh = (
                _last_baseline_refresh is None
                or (now - _last_baseline_refresh).total_seconds()
                >= BASELINE_REFRESH_HOURS * 3600
            )

            if should_refresh:
                logger.info("[DRIFT] Refreshing baselines...")
                baseline_result = await refresh_baselines_all_projects()
                _last_baseline_refresh = now
                logger.info(f"[DRIFT] Baseline refresh complete: {baseline_result}")

            # Run anomaly detection
            result = await detect_drift_all_projects()
            logger.info(f"[DRIFT] Detection complete: {result}")

        except Exception as e:
            logger.error(f"[DRIFT] Background loop error: {e}", exc_info=True)

        iteration += 1
        await asyncio.sleep(DRIFT_CHECK_INTERVAL_MINUTES * 60)
