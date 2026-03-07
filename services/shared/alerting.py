"""Unified alerting service for dispatching notifications across channels."""

import logging
from typing import Any

from .notifications import (
    send_email_notification,
    send_pagerduty_notification,
    send_slack_notification,
)

logger = logging.getLogger(__name__)

# Map metric alert severities to PagerDuty severity values
_PD_SEVERITY_MAP = {
    "critical": "critical",
    "high": "error",
    "medium": "warning",
    "low": "info",
}


def format_alert_email(
    rule_name: str,
    metric_type: str,
    current_value: float,
    threshold: float,
    condition: str,
    project_id: str,
    dashboard_url: str | None = None,
) -> tuple[str, str]:
    """Format a generic metric alert as HTML and plain text email.

    Returns:
        Tuple of (html_body, text_body).
    """
    text_body = (
        f"Alert: {rule_name}\n"
        f"Metric: {metric_type}\n"
        f"Current value: {current_value:.4f}\n"
        f"Condition: {condition} {threshold:.4f}\n"
        f"Project: {project_id}\n"
    )
    if dashboard_url:
        text_body += f"\nView in dashboard: {dashboard_url}"

    html_body = f"""
    <html>
        <body style="font-family: sans-serif; line-height: 1.6; color: #374151;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #111827;">Alert: {rule_name}</h2>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #e5e7eb; font-weight: bold;">Metric</td>
                        <td style="padding: 8px; border-bottom: 1px solid #e5e7eb;">{metric_type}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #e5e7eb; font-weight: bold;">Current Value</td>
                        <td style="padding: 8px; border-bottom: 1px solid #e5e7eb;">{current_value:.4f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #e5e7eb; font-weight: bold;">Threshold</td>
                        <td style="padding: 8px; border-bottom: 1px solid #e5e7eb;">{condition} {threshold:.4f}</td>
                    </tr>
                </table>
                {f'<p style="margin-top: 20px;"><a href="{dashboard_url}">View in Dashboard</a></p>' if dashboard_url else ''}
                <p style="margin-top: 30px; font-size: 12px; color: #6b7280;">
                    This is an automated alert from Prela.
                </p>
            </div>
        </body>
    </html>
    """

    return html_body, text_body


def format_alert_slack(
    rule_name: str,
    metric_type: str,
    current_value: float,
    threshold: float,
    condition: str,
    severity: str = "medium",
    dashboard_url: str | None = None,
) -> dict[str, Any]:
    """Format a generic metric alert as Slack message with blocks."""
    severity_emojis = {
        "low": ":large_blue_circle:",
        "medium": ":warning:",
        "high": ":large_orange_circle:",
        "critical": ":red_circle:",
    }
    emoji = severity_emojis.get(severity, ":gray_circle:")

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} Alert: {rule_name}",
            },
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Metric:*\n{metric_type}"},
                {"type": "mrkdwn", "text": f"*Condition:*\n{condition} {threshold:.4f}"},
                {"type": "mrkdwn", "text": f"*Current Value:*\n{current_value:.4f}"},
                {"type": "mrkdwn", "text": f"*Severity:*\n{severity.upper()}"},
            ],
        },
    ]

    if dashboard_url:
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View in Dashboard"},
                    "url": dashboard_url,
                    "style": "primary",
                }
            ],
        })

    return {"blocks": blocks}


async def dispatch_alert(
    rule_name: str,
    metric_type: str,
    current_value: float,
    threshold: float,
    condition: str,
    project_id: str,
    severity: str = "medium",
    # Notification channels
    notify_email: bool = False,
    email_addresses: list[str] | None = None,
    notify_slack: bool = False,
    slack_webhook_url: str | None = None,
    notify_pagerduty: bool = False,
    pagerduty_routing_key: str | None = None,
    # Optional
    dashboard_url: str | None = None,
    dedup_key: str | None = None,
) -> dict[str, bool]:
    """Dispatch an alert to all configured notification channels.

    Returns:
        Dict mapping channel name to success status.
    """
    results: dict[str, bool] = {}

    if notify_email and email_addresses:
        html_body, text_body = format_alert_email(
            rule_name, metric_type, current_value, threshold, condition,
            project_id, dashboard_url,
        )
        results["email"] = await send_email_notification(
            to_addresses=email_addresses,
            subject=f"[Prela Alert] {rule_name}: {metric_type}",
            body_html=html_body,
            body_text=text_body,
        )

    if notify_slack and slack_webhook_url:
        message = format_alert_slack(
            rule_name, metric_type, current_value, threshold, condition,
            severity, dashboard_url,
        )
        results["slack"] = await send_slack_notification(
            webhook_url=slack_webhook_url,
            message=message,
        )

    if notify_pagerduty and pagerduty_routing_key:
        pd_severity = _PD_SEVERITY_MAP.get(severity, "warning")
        results["pagerduty"] = await send_pagerduty_notification(
            routing_key=pagerduty_routing_key,
            summary=f"{rule_name}: {metric_type} is {current_value:.4f} ({condition} {threshold:.4f})",
            severity=pd_severity,
            source=f"prela/{project_id}",
            details={
                "rule_name": rule_name,
                "metric_type": metric_type,
                "current_value": current_value,
                "threshold": threshold,
                "condition": condition,
                "project_id": project_id,
            },
            dedup_key=dedup_key,
        )

    return results
