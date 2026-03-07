"""Generic alerting API endpoints.

Provides CRUD for alert rules that trigger on metric thresholds
(error rate, latency, cost, success rate, token usage) and
send notifications via email, Slack, or PagerDuty.
"""

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..auth import require_tier
from shared import get_clickhouse_client

logger = logging.getLogger(__name__)

router = APIRouter()

# Valid metric types for alert rules
VALID_METRIC_TYPES = {
    "error_rate",
    "latency_p95",
    "latency_mean",
    "cost_per_trace",
    "success_rate",
    "token_usage",
}

VALID_CONDITIONS = {"gt", "lt", "gte", "lte"}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CreateAlertRuleRequest(BaseModel):
    """Request to create a generic alert rule."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    enabled: bool = True
    metric_type: str = Field(..., description="One of: error_rate, latency_p95, latency_mean, cost_per_trace, success_rate, token_usage")
    condition: str = Field(..., description="One of: gt, lt, gte, lte")
    threshold: float
    evaluation_window_minutes: int = Field(default=60, ge=5, le=1440)
    agent_name: str | None = None
    # Notification channels
    notify_email: bool = False
    email_addresses: list[str] = Field(default_factory=list)
    notify_slack: bool = False
    slack_webhook_url: str | None = None
    notify_pagerduty: bool = False
    pagerduty_routing_key: str | None = None
    cooldown_minutes: int = Field(default=30, ge=5, le=1440)
    severity: str = Field(default="medium", description="low, medium, high, critical")


class UpdateAlertRuleRequest(BaseModel):
    """Request to update an alert rule."""

    name: str | None = None
    description: str | None = None
    enabled: bool | None = None
    metric_type: str | None = None
    condition: str | None = None
    threshold: float | None = None
    evaluation_window_minutes: int | None = None
    agent_name: str | None = None
    notify_email: bool | None = None
    email_addresses: list[str] | None = None
    notify_slack: bool | None = None
    slack_webhook_url: str | None = None
    notify_pagerduty: bool | None = None
    pagerduty_routing_key: str | None = None
    cooldown_minutes: int | None = None
    severity: str | None = None


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------

_SCHEMA_INITIALIZED = False


def _ensure_alert_tables() -> None:
    """Create alert tables in ClickHouse if they don't exist."""
    global _SCHEMA_INITIALIZED
    if _SCHEMA_INITIALIZED:
        return

    client = get_clickhouse_client()

    client.command("""
        CREATE TABLE IF NOT EXISTS generic_alert_rules (
            rule_id String,
            project_id String,
            name String,
            description String DEFAULT '',
            enabled UInt8 DEFAULT 1,
            metric_type String,
            `condition` String,
            threshold Float64,
            evaluation_window_minutes UInt32 DEFAULT 60,
            agent_name Nullable(String),
            notify_email UInt8 DEFAULT 0,
            email_addresses Array(String) DEFAULT [],
            notify_slack UInt8 DEFAULT 0,
            slack_webhook_url Nullable(String),
            notify_pagerduty UInt8 DEFAULT 0,
            pagerduty_routing_key Nullable(String),
            cooldown_minutes UInt32 DEFAULT 30,
            severity String DEFAULT 'medium',
            created_at DateTime DEFAULT now(),
            updated_at DateTime DEFAULT now()
        ) ENGINE = ReplacingMergeTree(updated_at)
        ORDER BY (project_id, rule_id)
    """)

    client.command("""
        CREATE TABLE IF NOT EXISTS generic_alert_history (
            alert_id String,
            rule_id String,
            project_id String,
            rule_name String,
            metric_type String,
            current_value Float64,
            threshold Float64,
            `condition` String,
            severity String,
            notification_results String DEFAULT '{}',
            triggered_at DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (project_id, triggered_at)
        TTL triggered_at + INTERVAL 90 DAY
    """)

    _SCHEMA_INITIALIZED = True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/projects/{project_id}/rules")
async def create_alert_rule(
    project_id: str,
    rule: CreateAlertRuleRequest,
    user: dict = Depends(require_tier("free")),
) -> dict[str, Any]:
    """Create a new generic alert rule."""
    if rule.metric_type not in VALID_METRIC_TYPES:
        raise HTTPException(400, f"Invalid metric_type. Must be one of: {VALID_METRIC_TYPES}")
    if rule.condition not in VALID_CONDITIONS:
        raise HTTPException(400, f"Invalid condition. Must be one of: {VALID_CONDITIONS}")

    _ensure_alert_tables()

    rule_id = str(uuid.uuid4())

    try:
        client = get_clickhouse_client()
        client.insert(
            "generic_alert_rules",
            [[
                rule_id,
                project_id,
                rule.name,
                rule.description or "",
                1 if rule.enabled else 0,
                rule.metric_type,
                rule.condition,
                rule.threshold,
                rule.evaluation_window_minutes,
                rule.agent_name,
                1 if rule.notify_email else 0,
                rule.email_addresses,
                1 if rule.notify_slack else 0,
                rule.slack_webhook_url,
                1 if rule.notify_pagerduty else 0,
                rule.pagerduty_routing_key,
                rule.cooldown_minutes,
                rule.severity,
            ]],
            column_names=[
                "rule_id", "project_id", "name", "description", "enabled",
                "metric_type", "condition", "threshold", "evaluation_window_minutes",
                "agent_name", "notify_email", "email_addresses", "notify_slack",
                "slack_webhook_url", "notify_pagerduty", "pagerduty_routing_key",
                "cooldown_minutes", "severity",
            ],
        )

        return {
            "rule_id": rule_id,
            "project_id": project_id,
            "name": rule.name,
            "metric_type": rule.metric_type,
            "condition": rule.condition,
            "threshold": rule.threshold,
            "enabled": rule.enabled,
        }

    except Exception as e:
        logger.error(f"Failed to create alert rule: {e}")
        raise HTTPException(500, detail=str(e))


@router.get("/projects/{project_id}/rules")
async def list_alert_rules(
    project_id: str,
    enabled: bool | None = Query(None),
    limit: int = Query(50, ge=1, le=100),
    user: dict = Depends(require_tier("free")),
) -> dict[str, Any]:
    """List alert rules for a project."""
    _ensure_alert_tables()

    conditions = ["project_id = %(project_id)s"]
    params: dict[str, Any] = {"project_id": project_id, "limit": limit}

    if enabled is not None:
        conditions.append(f"enabled = {1 if enabled else 0}")

    where = " AND ".join(conditions)

    try:
        client = get_clickhouse_client()
        result = client.query(
            f"""
            SELECT rule_id, project_id, name, description, enabled,
                   metric_type, `condition`, threshold, evaluation_window_minutes,
                   agent_name, notify_email, email_addresses, notify_slack,
                   slack_webhook_url, notify_pagerduty, pagerduty_routing_key,
                   cooldown_minutes, severity, created_at
            FROM generic_alert_rules FINAL
            WHERE {where}
            ORDER BY created_at DESC
            LIMIT %(limit)s
            """,
            parameters=params,
        )

        rules = []
        for row in result.result_rows:
            rules.append({
                "rule_id": row[0],
                "project_id": row[1],
                "name": row[2],
                "description": row[3],
                "enabled": bool(row[4]),
                "metric_type": row[5],
                "condition": row[6],
                "threshold": row[7],
                "evaluation_window_minutes": row[8],
                "agent_name": row[9],
                "notify_email": bool(row[10]),
                "email_addresses": row[11],
                "notify_slack": bool(row[12]),
                "slack_webhook_url": row[13],
                "notify_pagerduty": bool(row[14]),
                "pagerduty_routing_key": row[15],
                "cooldown_minutes": row[16],
                "severity": row[17],
                "created_at": str(row[18]),
            })

        return {"rules": rules, "total": len(rules)}

    except Exception as e:
        logger.error(f"Failed to list alert rules: {e}")
        raise HTTPException(500, detail=str(e))


@router.delete("/projects/{project_id}/rules/{rule_id}")
async def delete_alert_rule(
    project_id: str,
    rule_id: str,
    user: dict = Depends(require_tier("free")),
) -> dict[str, str]:
    """Delete an alert rule."""
    _ensure_alert_tables()

    try:
        client = get_clickhouse_client()
        client.command(
            """
            ALTER TABLE generic_alert_rules
            DELETE WHERE project_id = %(project_id)s AND rule_id = %(rule_id)s
            """,
            parameters={"project_id": project_id, "rule_id": rule_id},
        )
        return {"status": "deleted", "rule_id": rule_id}

    except Exception as e:
        logger.error(f"Failed to delete alert rule: {e}")
        raise HTTPException(500, detail=str(e))


@router.put("/projects/{project_id}/rules/{rule_id}")
async def update_alert_rule(
    project_id: str,
    rule_id: str,
    update: UpdateAlertRuleRequest,
    user: dict = Depends(require_tier("free")),
) -> dict[str, Any]:
    """Update an existing alert rule (full replacement)."""
    _ensure_alert_tables()

    try:
        client = get_clickhouse_client()

        # Fetch existing rule
        result = client.query(
            """
            SELECT rule_id, project_id, name, description, enabled,
                   metric_type, `condition`, threshold, evaluation_window_minutes,
                   agent_name, notify_email, email_addresses, notify_slack,
                   slack_webhook_url, notify_pagerduty, pagerduty_routing_key,
                   cooldown_minutes, severity, created_at
            FROM generic_alert_rules FINAL
            WHERE project_id = %(project_id)s AND rule_id = %(rule_id)s
            LIMIT 1
            """,
            parameters={"project_id": project_id, "rule_id": rule_id},
        )

        if not result.result_rows:
            raise HTTPException(404, "Alert rule not found")

        row = result.result_rows[0]
        current = {
            "name": row[2], "description": row[3], "enabled": bool(row[4]),
            "metric_type": row[5], "condition": row[6], "threshold": row[7],
            "evaluation_window_minutes": row[8], "agent_name": row[9],
            "notify_email": bool(row[10]), "email_addresses": row[11],
            "notify_slack": bool(row[12]), "slack_webhook_url": row[13],
            "notify_pagerduty": bool(row[14]), "pagerduty_routing_key": row[15],
            "cooldown_minutes": row[16], "severity": row[17],
        }

        # Apply updates
        update_data = update.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            if value is not None:
                current[key] = value

        # Validate updated values
        if current["metric_type"] not in VALID_METRIC_TYPES:
            raise HTTPException(400, f"Invalid metric_type: {current['metric_type']}")
        if current["condition"] not in VALID_CONDITIONS:
            raise HTTPException(400, f"Invalid condition: {current['condition']}")

        # Insert updated row (ReplacingMergeTree handles dedup)
        client.insert(
            "generic_alert_rules",
            [[
                rule_id, project_id,
                current["name"], current["description"],
                1 if current["enabled"] else 0,
                current["metric_type"], current["condition"],
                current["threshold"], current["evaluation_window_minutes"],
                current["agent_name"],
                1 if current["notify_email"] else 0, current["email_addresses"],
                1 if current["notify_slack"] else 0, current["slack_webhook_url"],
                1 if current["notify_pagerduty"] else 0, current["pagerduty_routing_key"],
                current["cooldown_minutes"], current["severity"],
            ]],
            column_names=[
                "rule_id", "project_id", "name", "description", "enabled",
                "metric_type", "condition", "threshold", "evaluation_window_minutes",
                "agent_name", "notify_email", "email_addresses", "notify_slack",
                "slack_webhook_url", "notify_pagerduty", "pagerduty_routing_key",
                "cooldown_minutes", "severity",
            ],
        )

        return {"status": "updated", "rule_id": rule_id, **current}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update alert rule: {e}")
        raise HTTPException(500, detail=str(e))


@router.get("/projects/{project_id}/history")
async def list_alert_history(
    project_id: str,
    rule_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    user: dict = Depends(require_tier("free")),
) -> dict[str, Any]:
    """List triggered alert history."""
    _ensure_alert_tables()

    conditions = ["project_id = %(project_id)s"]
    params: dict[str, Any] = {"project_id": project_id, "limit": limit}

    if rule_id:
        conditions.append("rule_id = %(rule_id)s")
        params["rule_id"] = rule_id

    where = " AND ".join(conditions)

    try:
        client = get_clickhouse_client()
        result = client.query(
            f"""
            SELECT alert_id, rule_id, project_id, rule_name, metric_type,
                   current_value, threshold, `condition`, severity,
                   notification_results, triggered_at
            FROM generic_alert_history
            WHERE {where}
            ORDER BY triggered_at DESC
            LIMIT %(limit)s
            """,
            parameters=params,
        )

        alerts = []
        for row in result.result_rows:
            alerts.append({
                "alert_id": row[0],
                "rule_id": row[1],
                "project_id": row[2],
                "rule_name": row[3],
                "metric_type": row[4],
                "current_value": row[5],
                "threshold": row[6],
                "condition": row[7],
                "severity": row[8],
                "notification_results": row[9],
                "triggered_at": str(row[10]),
            })

        return {"alerts": alerts, "total": len(alerts)}

    except Exception as e:
        logger.error(f"Failed to list alert history: {e}")
        raise HTTPException(500, detail=str(e))
