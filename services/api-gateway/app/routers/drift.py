"""Drift detection API endpoints."""

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..auth import require_tier

from shared import (
    get_clickhouse_client,
    send_email_notification,
    send_slack_notification,
    format_drift_alert_email,
    format_drift_alert_slack,
)
from shared.anomaly_detector import AnomalyDetector, AnomalySeverity
from shared.baseline_calculator import BaselineCalculator

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models


class BaselineMetrics(BaseModel):
    """Baseline metrics for an agent."""

    baseline_id: str
    agent_name: str
    service_name: str
    window_start: datetime
    window_end: datetime
    sample_size: int

    # Duration metrics
    duration_mean: float
    duration_stddev: float
    duration_p50: float
    duration_p95: float

    # Token usage
    token_usage_mean: float
    token_usage_stddev: float

    # Success rate
    success_rate: float
    error_count: int

    # Cost
    cost_mean: float
    cost_total: float


class Anomaly(BaseModel):
    """Detected anomaly."""

    metric_name: str
    current_value: float
    baseline_mean: float
    change_percent: float
    severity: AnomalySeverity
    direction: str  # "increased" or "decreased"
    unit: str
    sample_size: int


class RootCause(BaseModel):
    """Potential root cause for anomaly."""

    type: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)


class DriftAlert(BaseModel):
    """Drift detection alert."""

    agent_name: str
    service_name: str
    anomalies: list[Anomaly]
    root_causes: list[RootCause]
    baseline: BaselineMetrics
    detected_at: datetime


class StoredAlert(BaseModel):
    """Stored drift alert with metadata."""

    alert_id: str
    project_id: str
    agent_name: str
    service_name: str
    baseline_id: str
    detected_at: datetime
    severity: AnomalySeverity
    status: str  # "active", "acknowledged", "dismissed", "muted"
    anomalies: list[Anomaly]
    root_causes: list[RootCause]
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    dismissed_by: str | None = None
    dismissed_at: datetime | None = None
    mute_until: datetime | None = None
    notes: str | None = None
    created_at: datetime
    updated_at: datetime


class UpdateAlertRequest(BaseModel):
    """Request to update alert status."""

    status: str = Field(..., pattern="^(acknowledged|dismissed|muted|active)$")
    user_id: str | None = None
    notes: str | None = None
    mute_hours: int | None = Field(None, ge=1, le=168)  # Max 1 week


class AlertRule(BaseModel):
    """Alert rule configuration."""

    rule_id: str
    project_id: str
    name: str
    description: str | None = None
    enabled: bool = True

    # Rule conditions
    agent_name: str | None = None  # Null = all agents
    metric_name: str | None = None  # Null = all metrics
    severity_threshold: AnomalySeverity = AnomalySeverity.MEDIUM
    change_percent_min: float | None = None

    # Notification configuration
    notify_email: bool = False
    email_addresses: list[str] = []
    notify_slack: bool = False
    slack_webhook_url: str | None = None
    slack_channel: str | None = None

    # Metadata
    created_by: str
    created_at: datetime
    updated_at: datetime


class CreateAlertRuleRequest(BaseModel):
    """Request to create alert rule."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    enabled: bool = True

    # Rule conditions
    agent_name: str | None = None
    metric_name: str | None = None
    severity_threshold: AnomalySeverity = AnomalySeverity.MEDIUM
    change_percent_min: float | None = Field(None, ge=0, le=1000)

    # Notification configuration
    notify_email: bool = False
    email_addresses: list[str] = []
    notify_slack: bool = False
    slack_webhook_url: str | None = None
    slack_channel: str | None = None

    # User ID
    user_id: str


class UpdateAlertRuleRequest(BaseModel):
    """Request to update alert rule."""

    name: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = None
    enabled: bool | None = None

    # Rule conditions
    agent_name: str | None = None
    metric_name: str | None = None
    severity_threshold: AnomalySeverity | None = None
    change_percent_min: float | None = Field(None, ge=0, le=1000)

    # Notification configuration
    notify_email: bool | None = None
    email_addresses: list[str] | None = None
    notify_slack: bool | None = None
    slack_webhook_url: str | None = None
    slack_channel: str | None = None


# Helper Functions


async def _check_and_send_notifications(
    client: Any,
    project_id: str,
    agent_name: str,
    severity: AnomalySeverity,
    anomalies: list[Any],
) -> None:
    """Check alert rules and send notifications if matched.

    Args:
        client: ClickHouse client.
        project_id: Project identifier.
        agent_name: Agent name from alert.
        severity: Alert severity.
        anomalies: List of anomalies.
    """
    try:
        # Query matching alert rules
        query = """
            SELECT
                rule_id, name, notify_email, email_addresses,
                notify_slack, slack_webhook_url, slack_channel
            FROM alert_rules
            WHERE project_id = %(project_id)s
              AND enabled = true
              AND (agent_name IS NULL OR agent_name = %(agent_name)s)
              AND severity_threshold <= %(severity)s
        """

        result = client.query(
            query,
            parameters={
                "project_id": project_id,
                "agent_name": agent_name,
                "severity": severity,
            },
        )

        if not result.result_rows:
            logger.debug("No matching alert rules found")
            return

        # Convert anomalies to dicts for formatting
        anomaly_dicts = [
            {
                "metric_name": a.metric_name,
                "current_value": a.current_value,
                "baseline_mean": a.baseline_mean,
                "change_percent": a.change_percent,
            }
            for a in anomalies
        ]

        # Send notifications for each matching rule
        for row in result.result_rows:
            rule_id = row[0]
            rule_name = row[1]
            notify_email = row[2]
            email_addresses = row[3]
            notify_slack = row[4]
            slack_webhook_url = row[5]
            slack_channel = row[6]

            logger.info(f"Matched alert rule: {rule_name} ({rule_id})")

            # Send email notification
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
                logger.info(f"Email notification sent to {len(email_addresses)} recipients")

            # Send Slack notification
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
                logger.info("Slack notification sent")

    except Exception as e:
        # Don't fail alert creation if notifications fail
        logger.error(f"Failed to send notifications: {e}")


# API Endpoints


@router.get("/projects/{project_id}/baselines")
async def list_baselines(
    project_id: str,
    agent_name: str | None = Query(None),
    limit: int = Query(10, ge=1, le=100),
    user: dict = Depends(require_tier("pro")),
) -> dict[str, Any]:
    """List baselines for a project.

    Args:
        project_id: Project identifier.
        agent_name: Optional agent name filter.
        limit: Maximum number of baselines to return.

    Returns:
        List of baseline metrics.
    """
    try:
        client = get_clickhouse_client()

        conditions = ["project_id = %(project_id)s"]
        params: dict[str, Any] = {"project_id": project_id, "limit": limit}

        if agent_name:
            conditions.append("agent_name = %(agent_name)s")
            params["agent_name"] = agent_name

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT
                baseline_id, agent_name, service_name,
                window_start, window_end, sample_size,
                duration_mean, duration_stddev, duration_p50, duration_p95,
                token_usage_mean, token_usage_stddev,
                success_rate, error_count,
                cost_mean, cost_total
            FROM agent_baselines
            WHERE {where_clause}
            ORDER BY window_end DESC
            LIMIT %(limit)s
        """

        result = client.query(query, parameters=params)

        baselines = []
        for row in result.result_rows:
            baselines.append(
                {
                    "baseline_id": row[0],
                    "agent_name": row[1],
                    "service_name": row[2],
                    "window_start": row[3].isoformat() if row[3] else None,
                    "window_end": row[4].isoformat() if row[4] else None,
                    "sample_size": int(row[5]),
                    "duration_mean": float(row[6]),
                    "duration_stddev": float(row[7]),
                    "duration_p50": float(row[8]),
                    "duration_p95": float(row[9]),
                    "token_usage_mean": float(row[10]),
                    "token_usage_stddev": float(row[11]),
                    "success_rate": float(row[12]),
                    "error_count": int(row[13]),
                    "cost_mean": float(row[14]),
                    "cost_total": float(row[15]),
                }
            )

        return {"baselines": baselines, "count": len(baselines)}

    except Exception as e:
        logger.error(f"Failed to list baselines: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/baselines/calculate")
async def calculate_baselines(
    project_id: str,
    agent_name: str | None = Query(None),
    window_days: int = Query(7, ge=1, le=30),
    user: dict = Depends(require_tier("pro")),
) -> dict[str, Any]:
    """Calculate baselines for agents in a project.

    Args:
        project_id: Project identifier.
        agent_name: Optional specific agent to calculate.
        window_days: Rolling window size in days.

    Returns:
        Number of baselines calculated.
    """
    try:
        client = get_clickhouse_client()
        calculator = BaselineCalculator(client, window_days=window_days)

        if agent_name:
            # Calculate for specific agent
            # Need to get service_name first
            service_query = """
                SELECT DISTINCT service_name
                FROM spans
                WHERE project_id = %(project_id)s
                  AND JSONExtractString(attributes, 'agent.name') = %(agent_name)s
                  AND span_type = 'agent'
                LIMIT 1
            """
            result = client.query(
                service_query,
                parameters={"project_id": project_id, "agent_name": agent_name},
            )
            rows = result.result_rows
            if not rows:
                raise HTTPException(
                    status_code=404, detail=f"Agent '{agent_name}' not found"
                )

            service_name = rows[0][0]
            baseline = calculator.calculate_agent_baseline(
                project_id, agent_name, service_name
            )
            if baseline:
                calculator.save_baseline(baseline)
                return {"baselines_calculated": 1, "agent_name": agent_name}
            else:
                return {"baselines_calculated": 0, "message": "Insufficient data"}
        else:
            # Calculate for all agents
            count = calculator.calculate_all_baselines(project_id)
            return {"baselines_calculated": count}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to calculate baselines: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/drift/check")
async def check_drift(
    project_id: str,
    agent_name: str | None = Query(None),
    lookback_hours: int = Query(24, ge=1, le=168),
    sensitivity: float = Query(2.0, ge=1.0, le=4.0),
    user: dict = Depends(require_tier("pro")),
) -> dict[str, Any]:
    """Check for drift in agent behavior.

    Args:
        project_id: Project identifier.
        agent_name: Optional agent name filter.
        lookback_hours: Hours to look back for current metrics.
        sensitivity: Detection sensitivity in standard deviations.

    Returns:
        List of drift alerts.
    """
    try:
        client = get_clickhouse_client()
        calculator = BaselineCalculator(client)
        detector = AnomalyDetector(client, sensitivity=sensitivity)

        # Get agents to check
        if agent_name:
            # Get service name for specific agent
            service_query = """
                SELECT DISTINCT service_name
                FROM spans
                WHERE project_id = %(project_id)s
                  AND JSONExtractString(attributes, 'agent.name') = %(agent_name)s
                  AND span_type = 'agent'
                LIMIT 1
            """
            result = client.query(
                service_query,
                parameters={"project_id": project_id, "agent_name": agent_name},
            )
            rows = result.result_rows
            if not rows:
                raise HTTPException(
                    status_code=404, detail=f"Agent '{agent_name}' not found"
                )
            agents_to_check = [(agent_name, rows[0][0])]
        else:
            # Get all agents with recent activity
            agents_query = """
                SELECT DISTINCT
                    JSONExtractString(attributes, 'agent.name') AS agent_name,
                    service_name
                FROM spans
                WHERE project_id = %(project_id)s
                  AND span_type = 'agent'
                  AND started_at >= now() - INTERVAL %(lookback_hours)s HOUR
                  AND JSONHas(attributes, 'agent.name')
            """
            result = client.query(
                agents_query,
                parameters={"project_id": project_id, "lookback_hours": lookback_hours},
            )
            agents_to_check = [
                (row[0], row[1]) for row in result.result_rows if row[0]
            ]

        # Check each agent for drift
        alerts = []
        for agent_name, service_name in agents_to_check:
            # Get latest baseline
            baseline = calculator.get_latest_baseline(
                project_id, agent_name, service_name
            )
            if not baseline:
                logger.info(f"No baseline found for agent '{agent_name}', skipping")
                continue

            # Detect anomalies
            anomalies = detector.detect_anomalies(
                project_id, agent_name, service_name, baseline, lookback_hours
            )

            if anomalies:
                # Analyze root causes
                root_causes = detector.analyze_root_causes(
                    project_id, agent_name, service_name, anomalies, lookback_hours
                )

                alerts.append(
                    {
                        "agent_name": agent_name,
                        "service_name": service_name,
                        "anomalies": [
                            {
                                "metric_name": a["metric_name"],
                                "current_value": a["current_value"],
                                "baseline_mean": a.get("baseline_mean", 0),
                                "change_percent": a["change_percent"],
                                "severity": a["severity"],
                                "direction": a["direction"],
                                "unit": a["unit"],
                                "sample_size": a["sample_size"],
                            }
                            for a in anomalies
                        ],
                        "root_causes": root_causes,
                        "baseline": {
                            "baseline_id": baseline["baseline_id"],
                            "agent_name": baseline["agent_name"],
                            "service_name": baseline["service_name"],
                            "window_start": baseline["window_start"].isoformat(),
                            "window_end": baseline["window_end"].isoformat(),
                            "sample_size": baseline["sample_size"],
                            "duration_mean": baseline["duration_mean"],
                            "duration_stddev": baseline["duration_stddev"],
                            "duration_p50": baseline["duration_p50"],
                            "duration_p95": baseline["duration_p95"],
                            "token_usage_mean": baseline["token_usage_mean"],
                            "token_usage_stddev": baseline["token_usage_stddev"],
                            "success_rate": baseline["success_rate"],
                            "error_count": baseline["error_count"],
                            "cost_mean": baseline["cost_mean"],
                            "cost_total": baseline["cost_total"],
                        },
                        "detected_at": datetime.utcnow().isoformat(),
                    }
                )

        return {"alerts": alerts, "count": len(alerts)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/alerts")
async def create_alert(project_id: str, alert: DriftAlert, user: dict = Depends(require_tier("pro"))) -> dict[str, Any]:
    """Store a drift alert.

    Args:
        project_id: Project identifier.
        alert: Drift alert data.

    Returns:
        Created alert with ID.
    """
    try:
        client = get_clickhouse_client()
        alert_id = str(uuid.uuid4())

        # Determine severity from anomalies
        severities = [a.severity for a in alert.anomalies]
        max_severity = max(severities) if severities else AnomalySeverity.LOW

        # Serialize anomalies and root causes to JSON
        anomalies_json = json.dumps(
            [
                {
                    "metric_name": a.metric_name,
                    "current_value": a.current_value,
                    "baseline_mean": a.baseline_mean,
                    "change_percent": a.change_percent,
                    "severity": a.severity,
                    "direction": a.direction,
                    "unit": a.unit,
                    "sample_size": a.sample_size,
                }
                for a in alert.anomalies
            ]
        )

        root_causes_json = json.dumps(
            [
                {
                    "type": rc.type,
                    "description": rc.description,
                    "confidence": rc.confidence,
                }
                for rc in alert.root_causes
            ]
        )

        # Insert alert
        insert_query = """
            INSERT INTO drift_alerts (
                alert_id, project_id, agent_name, service_name, baseline_id,
                detected_at, severity, status, anomalies, root_causes
            ) VALUES
        """

        client.insert(
            "drift_alerts",
            [
                [
                    alert_id,
                    project_id,
                    alert.agent_name,
                    alert.service_name,
                    alert.baseline.baseline_id,
                    alert.detected_at,
                    max_severity,
                    "active",
                    anomalies_json,
                    root_causes_json,
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

        # Check for alert rules and send notifications
        await _check_and_send_notifications(
            client, project_id, alert.agent_name, max_severity, alert.anomalies
        )

        return {
            "alert_id": alert_id,
            "project_id": project_id,
            "status": "active",
            "severity": max_severity,
        }

    except Exception as e:
        logger.error(f"Failed to create alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/alerts")
async def list_alerts(
    project_id: str,
    agent_name: str | None = Query(None),
    status: str | None = Query(None),
    severity: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    user: dict = Depends(require_tier("pro")),
) -> dict[str, Any]:
    """List drift alerts for a project.

    Args:
        project_id: Project identifier.
        agent_name: Optional agent name filter.
        status: Optional status filter (active, acknowledged, dismissed, muted).
        severity: Optional severity filter (low, medium, high, critical).
        limit: Maximum number of alerts to return.

    Returns:
        List of drift alerts.
    """
    try:
        client = get_clickhouse_client()

        conditions = ["project_id = %(project_id)s"]
        params: dict[str, Any] = {"project_id": project_id, "limit": limit}

        if agent_name:
            conditions.append("agent_name = %(agent_name)s")
            params["agent_name"] = agent_name

        if status:
            conditions.append("status = %(status)s")
            params["status"] = status

        if severity:
            conditions.append("severity = %(severity)s")
            params["severity"] = severity

        # Only show active and unmuted alerts by default
        if not status:
            conditions.append(
                "(status = 'active' OR (status = 'muted' AND mute_until > now()))"
            )

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT
                alert_id, project_id, agent_name, service_name, baseline_id,
                detected_at, severity, status, anomalies, root_causes,
                acknowledged_by, acknowledged_at, dismissed_by, dismissed_at,
                mute_until, notes, created_at, updated_at
            FROM drift_alerts
            WHERE {where_clause}
            ORDER BY severity DESC, detected_at DESC
            LIMIT %(limit)s
        """

        result = client.query(query, parameters=params)

        alerts = []
        for row in result.result_rows:
            alerts.append(
                {
                    "alert_id": row[0],
                    "project_id": row[1],
                    "agent_name": row[2],
                    "service_name": row[3],
                    "baseline_id": row[4],
                    "detected_at": row[5].isoformat() if row[5] else None,
                    "severity": row[6],
                    "status": row[7],
                    "anomalies": json.loads(row[8]) if row[8] else [],
                    "root_causes": json.loads(row[9]) if row[9] else [],
                    "acknowledged_by": row[10],
                    "acknowledged_at": row[11].isoformat() if row[11] else None,
                    "dismissed_by": row[12],
                    "dismissed_at": row[13].isoformat() if row[13] else None,
                    "mute_until": row[14].isoformat() if row[14] else None,
                    "notes": row[15],
                    "created_at": row[16].isoformat() if row[16] else None,
                    "updated_at": row[17].isoformat() if row[17] else None,
                }
            )

        return {"alerts": alerts, "count": len(alerts)}

    except Exception as e:
        logger.error(f"Failed to list alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/alerts/{alert_id}")
async def get_alert(project_id: str, alert_id: str, user: dict = Depends(require_tier("pro"))) -> dict[str, Any]:
    """Get a specific alert by ID.

    Args:
        project_id: Project identifier.
        alert_id: Alert identifier.

    Returns:
        Alert details.
    """
    try:
        client = get_clickhouse_client()

        query = """
            SELECT
                alert_id, project_id, agent_name, service_name, baseline_id,
                detected_at, severity, status, anomalies, root_causes,
                acknowledged_by, acknowledged_at, dismissed_by, dismissed_at,
                mute_until, notes, created_at, updated_at
            FROM drift_alerts
            WHERE project_id = %(project_id)s AND alert_id = %(alert_id)s
            LIMIT 1
        """

        result = client.query(
            query, parameters={"project_id": project_id, "alert_id": alert_id}
        )

        if not result.result_rows:
            raise HTTPException(status_code=404, detail="Alert not found")

        row = result.result_rows[0]
        return {
            "alert_id": row[0],
            "project_id": row[1],
            "agent_name": row[2],
            "service_name": row[3],
            "baseline_id": row[4],
            "detected_at": row[5].isoformat() if row[5] else None,
            "severity": row[6],
            "status": row[7],
            "anomalies": json.loads(row[8]) if row[8] else [],
            "root_causes": json.loads(row[9]) if row[9] else [],
            "acknowledged_by": row[10],
            "acknowledged_at": row[11].isoformat() if row[11] else None,
            "dismissed_by": row[12],
            "dismissed_at": row[13].isoformat() if row[13] else None,
            "mute_until": row[14].isoformat() if row[14] else None,
            "notes": row[15],
            "created_at": row[16].isoformat() if row[16] else None,
            "updated_at": row[17].isoformat() if row[17] else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/projects/{project_id}/alerts/{alert_id}")
async def update_alert(
    project_id: str, alert_id: str, update: UpdateAlertRequest, user: dict = Depends(require_tier("pro"))
) -> dict[str, Any]:
    """Update an alert's status.

    Args:
        project_id: Project identifier.
        alert_id: Alert identifier.
        update: Update request with status and metadata.

    Returns:
        Updated alert.
    """
    try:
        client = get_clickhouse_client()

        # Build update fields based on status
        update_fields = ["status = %(status)s"]
        params: dict[str, Any] = {
            "project_id": project_id,
            "alert_id": alert_id,
            "status": update.status,
            "now": datetime.utcnow(),
        }

        if update.notes:
            update_fields.append("notes = %(notes)s")
            params["notes"] = update.notes

        if update.status == "acknowledged":
            update_fields.append("acknowledged_by = %(user_id)s")
            update_fields.append("acknowledged_at = %(now)s")
            params["user_id"] = update.user_id or "system"
        elif update.status == "dismissed":
            update_fields.append("dismissed_by = %(user_id)s")
            update_fields.append("dismissed_at = %(now)s")
            params["user_id"] = update.user_id or "system"
        elif update.status == "muted" and update.mute_hours:
            from datetime import timedelta

            mute_until = datetime.utcnow() + timedelta(hours=update.mute_hours)
            update_fields.append("mute_until = %(mute_until)s")
            params["mute_until"] = mute_until

        update_fields.append("updated_at = %(now)s")

        # ClickHouse doesn't support UPDATE with WHERE, so we need to use ALTER TABLE
        # For simplicity, we'll use ReplacingMergeTree's deduplication by re-inserting
        # First, get the current alert
        get_query = """
            SELECT
                alert_id, project_id, agent_name, service_name, baseline_id,
                detected_at, severity, status, anomalies, root_causes,
                acknowledged_by, acknowledged_at, dismissed_by, dismissed_at,
                mute_until, notes, created_at
            FROM drift_alerts
            WHERE project_id = %(project_id)s AND alert_id = %(alert_id)s
            LIMIT 1
        """

        result = client.query(
            get_query, parameters={"project_id": project_id, "alert_id": alert_id}
        )

        if not result.result_rows:
            raise HTTPException(status_code=404, detail="Alert not found")

        row = result.result_rows[0]

        # Prepare updated values
        updated_row = list(row)  # Copy existing values
        updated_row[7] = update.status  # status

        if update.notes:
            updated_row[15] = update.notes

        if update.status == "acknowledged":
            updated_row[10] = update.user_id or "system"  # acknowledged_by
            updated_row[11] = datetime.utcnow()  # acknowledged_at
        elif update.status == "dismissed":
            updated_row[12] = update.user_id or "system"  # dismissed_by
            updated_row[13] = datetime.utcnow()  # dismissed_at
        elif update.status == "muted" and update.mute_hours:
            from datetime import timedelta

            updated_row[14] = datetime.utcnow() + timedelta(
                hours=update.mute_hours
            )  # mute_until

        # Append updated_at
        updated_row.append(datetime.utcnow())

        # Re-insert with updated values (ReplacingMergeTree will handle deduplication)
        client.insert(
            "drift_alerts",
            [updated_row],
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
                "acknowledged_by",
                "acknowledged_at",
                "dismissed_by",
                "dismissed_at",
                "mute_until",
                "notes",
                "created_at",
                "updated_at",
            ],
        )

        return {"alert_id": alert_id, "status": update.status, "updated": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Alert Rules Endpoints


@router.post("/projects/{project_id}/alert-rules")
async def create_alert_rule(
    project_id: str, rule: CreateAlertRuleRequest, user: dict = Depends(require_tier("pro"))
) -> dict[str, Any]:
    """Create a new alert rule.

    Args:
        project_id: Project identifier.
        rule: Alert rule configuration.

    Returns:
        Created rule with ID.
    """
    try:
        client = get_clickhouse_client()
        rule_id = str(uuid.uuid4())

        # Insert rule
        client.insert(
            "alert_rules",
            [
                [
                    rule_id,
                    project_id,
                    rule.name,
                    rule.description,
                    rule.enabled,
                    rule.agent_name,
                    rule.metric_name,
                    rule.severity_threshold,
                    rule.change_percent_min,
                    rule.notify_email,
                    rule.email_addresses,
                    rule.notify_slack,
                    rule.slack_webhook_url,
                    rule.slack_channel,
                    rule.user_id,
                ]
            ],
            column_names=[
                "rule_id",
                "project_id",
                "name",
                "description",
                "enabled",
                "agent_name",
                "metric_name",
                "severity_threshold",
                "change_percent_min",
                "notify_email",
                "email_addresses",
                "notify_slack",
                "slack_webhook_url",
                "slack_channel",
                "created_by",
            ],
        )

        return {
            "rule_id": rule_id,
            "project_id": project_id,
            "name": rule.name,
        }

    except Exception as e:
        logger.error(f"Failed to create alert rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/alert-rules")
async def list_alert_rules(
    project_id: str,
    enabled: bool | None = Query(None),
    limit: int = Query(50, ge=1, le=100),
    user: dict = Depends(require_tier("pro")),
) -> dict[str, Any]:
    """List alert rules for a project.

    Args:
        project_id: Project identifier.
        enabled: Optional filter by enabled status.
        limit: Maximum number of rules to return.

    Returns:
        List of alert rules.
    """
    try:
        client = get_clickhouse_client()

        conditions = ["project_id = %(project_id)s"]
        params: dict[str, Any] = {"project_id": project_id, "limit": limit}

        if enabled is not None:
            conditions.append("enabled = %(enabled)s")
            params["enabled"] = enabled

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT
                rule_id, project_id, name, description, enabled,
                agent_name, metric_name, severity_threshold, change_percent_min,
                notify_email, email_addresses, notify_slack, slack_webhook_url, slack_channel,
                created_by, created_at, updated_at
            FROM alert_rules
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT %(limit)s
        """

        result = client.query(query, parameters=params)

        rules = []
        for row in result.result_rows:
            rules.append(
                {
                    "rule_id": row[0],
                    "project_id": row[1],
                    "name": row[2],
                    "description": row[3],
                    "enabled": row[4],
                    "agent_name": row[5],
                    "metric_name": row[6],
                    "severity_threshold": row[7],
                    "change_percent_min": row[8],
                    "notify_email": row[9],
                    "email_addresses": row[10],
                    "notify_slack": row[11],
                    "slack_webhook_url": row[12],
                    "slack_channel": row[13],
                    "created_by": row[14],
                    "created_at": row[15].isoformat() if row[15] else None,
                    "updated_at": row[16].isoformat() if row[16] else None,
                }
            )

        return {"rules": rules, "count": len(rules)}

    except Exception as e:
        logger.error(f"Failed to list alert rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/alert-rules/{rule_id}")
async def get_alert_rule(project_id: str, rule_id: str, user: dict = Depends(require_tier("pro"))) -> dict[str, Any]:
    """Get a specific alert rule.

    Args:
        project_id: Project identifier.
        rule_id: Rule identifier.

    Returns:
        Alert rule details.
    """
    try:
        client = get_clickhouse_client()

        query = """
            SELECT
                rule_id, project_id, name, description, enabled,
                agent_name, metric_name, severity_threshold, change_percent_min,
                notify_email, email_addresses, notify_slack, slack_webhook_url, slack_channel,
                created_by, created_at, updated_at
            FROM alert_rules
            WHERE project_id = %(project_id)s AND rule_id = %(rule_id)s
            LIMIT 1
        """

        result = client.query(
            query, parameters={"project_id": project_id, "rule_id": rule_id}
        )

        if not result.result_rows:
            raise HTTPException(status_code=404, detail="Alert rule not found")

        row = result.result_rows[0]
        return {
            "rule_id": row[0],
            "project_id": row[1],
            "name": row[2],
            "description": row[3],
            "enabled": row[4],
            "agent_name": row[5],
            "metric_name": row[6],
            "severity_threshold": row[7],
            "change_percent_min": row[8],
            "notify_email": row[9],
            "email_addresses": row[10],
            "notify_slack": row[11],
            "slack_webhook_url": row[12],
            "slack_channel": row[13],
            "created_by": row[14],
            "created_at": row[15].isoformat() if row[15] else None,
            "updated_at": row[16].isoformat() if row[16] else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get alert rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/projects/{project_id}/alert-rules/{rule_id}")
async def update_alert_rule(
    project_id: str, rule_id: str, update: UpdateAlertRuleRequest, user: dict = Depends(require_tier("pro"))
) -> dict[str, Any]:
    """Update an alert rule.

    Args:
        project_id: Project identifier.
        rule_id: Rule identifier.
        update: Update request.

    Returns:
        Updated rule.
    """
    try:
        client = get_clickhouse_client()

        # Get current rule
        get_query = """
            SELECT
                rule_id, project_id, name, description, enabled,
                agent_name, metric_name, severity_threshold, change_percent_min,
                notify_email, email_addresses, notify_slack, slack_webhook_url, slack_channel,
                created_by, created_at
            FROM alert_rules
            WHERE project_id = %(project_id)s AND rule_id = %(rule_id)s
            LIMIT 1
        """

        result = client.query(
            get_query, parameters={"project_id": project_id, "rule_id": rule_id}
        )

        if not result.result_rows:
            raise HTTPException(status_code=404, detail="Alert rule not found")

        row = result.result_rows[0]

        # Prepare updated values
        updated_row = list(row)

        # Update fields if provided
        if update.name is not None:
            updated_row[2] = update.name
        if update.description is not None:
            updated_row[3] = update.description
        if update.enabled is not None:
            updated_row[4] = update.enabled
        if update.agent_name is not None:
            updated_row[5] = update.agent_name
        if update.metric_name is not None:
            updated_row[6] = update.metric_name
        if update.severity_threshold is not None:
            updated_row[7] = update.severity_threshold
        if update.change_percent_min is not None:
            updated_row[8] = update.change_percent_min
        if update.notify_email is not None:
            updated_row[9] = update.notify_email
        if update.email_addresses is not None:
            updated_row[10] = update.email_addresses
        if update.notify_slack is not None:
            updated_row[11] = update.notify_slack
        if update.slack_webhook_url is not None:
            updated_row[12] = update.slack_webhook_url
        if update.slack_channel is not None:
            updated_row[13] = update.slack_channel

        # Append updated_at
        updated_row.append(datetime.utcnow())

        # Re-insert with updated values
        client.insert(
            "alert_rules",
            [updated_row],
            column_names=[
                "rule_id",
                "project_id",
                "name",
                "description",
                "enabled",
                "agent_name",
                "metric_name",
                "severity_threshold",
                "change_percent_min",
                "notify_email",
                "email_addresses",
                "notify_slack",
                "slack_webhook_url",
                "slack_channel",
                "created_by",
                "created_at",
                "updated_at",
            ],
        )

        return {"rule_id": rule_id, "updated": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update alert rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/projects/{project_id}/alert-rules/{rule_id}")
async def delete_alert_rule(project_id: str, rule_id: str, user: dict = Depends(require_tier("pro"))) -> dict[str, Any]:
    """Delete an alert rule.

    Args:
        project_id: Project identifier.
        rule_id: Rule identifier.

    Returns:
        Deletion confirmation.
    """
    try:
        client = get_clickhouse_client()

        # ClickHouse doesn't support DELETE, so we'll insert with enabled=false
        # or use ALTER TABLE DELETE (which is async)
        query = """
            ALTER TABLE alert_rules
            DELETE WHERE project_id = %(project_id)s AND rule_id = %(rule_id)s
        """

        client.command(query, parameters={"project_id": project_id, "rule_id": rule_id})

        return {"rule_id": rule_id, "deleted": True}

    except Exception as e:
        logger.error(f"Failed to delete alert rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))
