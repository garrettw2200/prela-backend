"""Background service that evaluates generic alert rules against ClickHouse metrics.

Runs periodically, queries spans for each active rule's metric over its
evaluation window, compares against threshold, and dispatches notifications
with cooldown-based deduplication.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

from shared import get_clickhouse_client, settings
from shared.alerting import dispatch_alert

logger = logging.getLogger(__name__)

ALERT_CHECK_INTERVAL_MINUTES = settings.alert_check_interval_minutes

# In-memory cooldown tracker: rule_id -> last_fired_at
_cooldowns: dict[str, datetime] = {}

# SQL templates for each metric type.
# Each returns a single float value for the given project/agent/window.
_METRIC_QUERIES: dict[str, str] = {
    "error_rate": """
        SELECT countIf(status_code >= 400 OR status = 'ERROR') / count() AS value
        FROM spans
        WHERE project_id = %(project_id)s
          AND start_time >= now() - INTERVAL %(window)s MINUTE
          {agent_filter}
    """,
    "latency_p95": """
        SELECT quantile(0.95)(duration_ns / 1e6) AS value
        FROM spans
        WHERE project_id = %(project_id)s
          AND start_time >= now() - INTERVAL %(window)s MINUTE
          AND span_type = 'llm'
          {agent_filter}
    """,
    "latency_mean": """
        SELECT avg(duration_ns / 1e6) AS value
        FROM spans
        WHERE project_id = %(project_id)s
          AND start_time >= now() - INTERVAL %(window)s MINUTE
          AND span_type = 'llm'
          {agent_filter}
    """,
    "cost_per_trace": """
        SELECT sum(cost) / countDistinct(trace_id) AS value
        FROM spans
        WHERE project_id = %(project_id)s
          AND start_time >= now() - INTERVAL %(window)s MINUTE
          {agent_filter}
    """,
    "success_rate": """
        SELECT countIf(status_code < 400 AND status != 'ERROR') / count() AS value
        FROM spans
        WHERE project_id = %(project_id)s
          AND start_time >= now() - INTERVAL %(window)s MINUTE
          {agent_filter}
    """,
    "token_usage": """
        SELECT sum(total_tokens) AS value
        FROM spans
        WHERE project_id = %(project_id)s
          AND start_time >= now() - INTERVAL %(window)s MINUTE
          AND span_type = 'llm'
          {agent_filter}
    """,
}

_CONDITION_OPS = {
    "gt": lambda v, t: v > t,
    "lt": lambda v, t: v < t,
    "gte": lambda v, t: v >= t,
    "lte": lambda v, t: v <= t,
}


def _evaluate_condition(value: float, condition: str, threshold: float) -> bool:
    """Check if value meets the alert condition."""
    op = _CONDITION_OPS.get(condition)
    if op is None:
        return False
    return op(value, threshold)


async def _evaluate_rule(rule: dict) -> None:
    """Evaluate a single alert rule and dispatch notification if triggered."""
    rule_id = rule["rule_id"]
    project_id = rule["project_id"]
    metric_type = rule["metric_type"]
    condition = rule["condition"]
    threshold = rule["threshold"]
    window = rule["evaluation_window_minutes"]
    cooldown = rule["cooldown_minutes"]

    # Check cooldown
    now = datetime.now(timezone.utc)
    last_fired = _cooldowns.get(rule_id)
    if last_fired and (now - last_fired).total_seconds() < cooldown * 60:
        return

    # Get metric query template
    query_template = _METRIC_QUERIES.get(metric_type)
    if not query_template:
        logger.warning(f"[ALERT] Unknown metric_type: {metric_type}")
        return

    # Build agent filter
    agent_filter = ""
    if rule.get("agent_name"):
        agent_filter = "AND agent_name = %(agent_name)s"

    query = query_template.format(agent_filter=agent_filter)

    try:
        client = get_clickhouse_client()
        params = {
            "project_id": project_id,
            "window": window,
        }
        if rule.get("agent_name"):
            params["agent_name"] = rule["agent_name"]

        result = client.query(query, parameters=params)

        if not result.result_rows or result.result_rows[0][0] is None:
            return

        current_value = float(result.result_rows[0][0])

    except Exception as e:
        logger.error(f"[ALERT] Failed to query metric for rule {rule_id}: {e}")
        return

    # Check condition
    if not _evaluate_condition(current_value, condition, threshold):
        return

    # Condition met — fire alert
    logger.info(
        f"[ALERT] Rule '{rule['name']}' triggered: "
        f"{metric_type}={current_value:.4f} {condition} {threshold:.4f}"
    )

    _cooldowns[rule_id] = now

    # Dispatch notifications
    try:
        notification_results = await dispatch_alert(
            rule_name=rule["name"],
            metric_type=metric_type,
            current_value=current_value,
            threshold=threshold,
            condition=condition,
            project_id=project_id,
            severity=rule.get("severity", "medium"),
            notify_email=rule.get("notify_email", False),
            email_addresses=rule.get("email_addresses", []),
            notify_slack=rule.get("notify_slack", False),
            slack_webhook_url=rule.get("slack_webhook_url"),
            notify_pagerduty=rule.get("notify_pagerduty", False),
            pagerduty_routing_key=rule.get("pagerduty_routing_key"),
            dedup_key=f"prela-alert-{rule_id}",
        )
    except Exception as e:
        logger.error(f"[ALERT] Failed to dispatch notifications for rule {rule_id}: {e}")
        notification_results = {"error": str(e)}

    # Record in alert history
    try:
        alert_id = str(uuid.uuid4())
        client = get_clickhouse_client()
        client.insert(
            "generic_alert_history",
            [[
                alert_id, rule_id, project_id, rule["name"],
                metric_type, current_value, threshold, condition,
                rule.get("severity", "medium"),
                json.dumps(notification_results),
            ]],
            column_names=[
                "alert_id", "rule_id", "project_id", "rule_name",
                "metric_type", "current_value", "threshold", "condition",
                "severity", "notification_results",
            ],
        )
    except Exception as e:
        logger.error(f"[ALERT] Failed to record alert history: {e}")


async def _run_evaluation_cycle() -> dict:
    """Run a single evaluation cycle across all projects and rules."""
    try:
        client = get_clickhouse_client()

        # Fetch all enabled rules
        result = client.query("""
            SELECT rule_id, project_id, name, description, enabled,
                   metric_type, `condition`, threshold, evaluation_window_minutes,
                   agent_name, notify_email, email_addresses, notify_slack,
                   slack_webhook_url, notify_pagerduty, pagerduty_routing_key,
                   cooldown_minutes, severity
            FROM generic_alert_rules FINAL
            WHERE enabled = 1
        """)

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
            })

        if not rules:
            return {"rules_checked": 0}

        # Evaluate each rule
        for rule in rules:
            await _evaluate_rule(rule)

        return {"rules_checked": len(rules)}

    except Exception as e:
        logger.error(f"[ALERT] Evaluation cycle failed: {e}", exc_info=True)
        return {"error": str(e)}


async def background_alert_evaluation_loop() -> None:
    """Background loop that evaluates alert rules periodically."""
    logger.info(
        f"[ALERT] Background evaluation loop started "
        f"(interval={ALERT_CHECK_INTERVAL_MINUTES}min)"
    )

    # Initial delay to let the app fully start
    await asyncio.sleep(90)

    while True:
        try:
            result = await _run_evaluation_cycle()
            logger.debug(f"[ALERT] Evaluation cycle complete: {result}")
        except Exception as e:
            logger.error(f"[ALERT] Background loop error: {e}", exc_info=True)

        await asyncio.sleep(ALERT_CHECK_INTERVAL_MINUTES * 60)
