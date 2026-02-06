"""Notification service for sending email and Slack alerts."""

import asyncio
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import httpx

from .config import settings

logger = logging.getLogger(__name__)


async def send_email_notification(
    to_addresses: list[str],
    subject: str,
    body_html: str,
    body_text: str | None = None,
) -> bool:
    """Send email notification via SMTP.

    Args:
        to_addresses: List of recipient email addresses.
        subject: Email subject line.
        body_html: HTML body content.
        body_text: Optional plain text body.

    Returns:
        True if sent successfully, False otherwise.
    """
    try:
        # Get SMTP settings from environment
        smtp_host = settings.SMTP_HOST
        smtp_port = settings.SMTP_PORT
        smtp_user = settings.SMTP_USER
        smtp_password = settings.SMTP_PASSWORD
        from_address = settings.SMTP_FROM_ADDRESS

        if not all([smtp_host, smtp_user, smtp_password, from_address]):
            logger.warning("SMTP not configured, skipping email notification")
            return False

        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_address
        msg["To"] = ", ".join(to_addresses)

        # Add text and HTML parts
        if body_text:
            msg.attach(MIMEText(body_text, "plain"))
        msg.attach(MIMEText(body_html, "html"))

        # Send via SMTP
        # Use asyncio to run blocking SMTP in thread pool
        await asyncio.to_thread(_send_smtp, smtp_host, smtp_port, smtp_user, smtp_password, from_address, to_addresses, msg)

        logger.info(f"Email sent to {len(to_addresses)} recipients")
        return True

    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


def _send_smtp(
    host: str,
    port: int,
    user: str,
    password: str,
    from_address: str,
    to_addresses: list[str],
    msg: MIMEMultipart,
) -> None:
    """Blocking SMTP send (called in thread pool)."""
    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(user, password)
        server.sendmail(from_address, to_addresses, msg.as_string())


async def send_slack_notification(
    webhook_url: str,
    message: dict[str, Any],
    channel: str | None = None,
) -> bool:
    """Send Slack notification via webhook.

    Args:
        webhook_url: Slack webhook URL.
        message: Slack message payload (blocks or text).
        channel: Optional channel override.

    Returns:
        True if sent successfully, False otherwise.
    """
    try:
        payload = message.copy()
        if channel:
            payload["channel"] = channel

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                logger.error(f"Slack webhook failed: {response.status_code} {response.text}")
                return False

            logger.info("Slack notification sent successfully")
            return True

    except Exception as e:
        logger.error(f"Failed to send Slack notification: {e}")
        return False


def format_drift_alert_email(
    agent_name: str,
    severity: str,
    anomalies: list[dict[str, Any]],
    dashboard_url: str | None = None,
) -> tuple[str, str]:
    """Format drift alert as HTML and plain text email.

    Args:
        agent_name: Name of the agent.
        severity: Alert severity (low/medium/high/critical).
        anomalies: List of anomaly dictionaries.
        dashboard_url: Optional link to dashboard.

    Returns:
        Tuple of (html_body, text_body).
    """
    # Plain text version
    text_lines = [
        f"Drift Alert: {agent_name}",
        f"Severity: {severity.upper()}",
        "",
        "Anomalies detected:",
    ]

    for anomaly in anomalies:
        text_lines.append(
            f"  • {anomaly['metric_name']}: {anomaly['current_value']:.2f} "
            f"(baseline: {anomaly['baseline_mean']:.2f}, "
            f"change: {anomaly['change_percent']:+.1f}%)"
        )

    if dashboard_url:
        text_lines.append("")
        text_lines.append(f"View in dashboard: {dashboard_url}")

    text_body = "\n".join(text_lines)

    # HTML version
    severity_colors = {
        "low": "#3b82f6",  # blue
        "medium": "#eab308",  # yellow
        "high": "#f97316",  # orange
        "critical": "#ef4444",  # red
    }
    color = severity_colors.get(severity, "#6b7280")

    anomaly_rows = ""
    for anomaly in anomalies:
        anomaly_rows += f"""
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #e5e7eb;">{anomaly['metric_name']}</td>
            <td style="padding: 8px; border-bottom: 1px solid #e5e7eb;">{anomaly['baseline_mean']:.2f}</td>
            <td style="padding: 8px; border-bottom: 1px solid #e5e7eb;">{anomaly['current_value']:.2f}</td>
            <td style="padding: 8px; border-bottom: 1px solid #e5e7eb; font-weight: bold; color: {color};">
                {anomaly['change_percent']:+.1f}%
            </td>
        </tr>
        """

    dashboard_link = ""
    if dashboard_url:
        dashboard_link = f"""
        <p style="margin-top: 20px;">
            <a href="{dashboard_url}" style="color: #4f46e5; text-decoration: none;">
                View in Dashboard →
            </a>
        </p>
        """

    html_body = f"""
    <html>
        <body style="font-family: sans-serif; line-height: 1.6; color: #374151;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #111827; margin-bottom: 10px;">
                    Drift Alert: {agent_name}
                </h2>
                <p style="margin: 10px 0;">
                    <strong style="color: {color}; text-transform: uppercase;">
                        {severity} severity
                    </strong>
                </p>

                <h3 style="color: #111827; margin-top: 20px; margin-bottom: 10px;">
                    Anomalies Detected
                </h3>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                    <thead>
                        <tr style="background-color: #f3f4f6;">
                            <th style="padding: 8px; text-align: left; border-bottom: 2px solid #d1d5db;">Metric</th>
                            <th style="padding: 8px; text-align: left; border-bottom: 2px solid #d1d5db;">Baseline</th>
                            <th style="padding: 8px; text-align: left; border-bottom: 2px solid #d1d5db;">Current</th>
                            <th style="padding: 8px; text-align: left; border-bottom: 2px solid #d1d5db;">Change</th>
                        </tr>
                    </thead>
                    <tbody>
                        {anomaly_rows}
                    </tbody>
                </table>

                {dashboard_link}

                <p style="margin-top: 30px; font-size: 12px; color: #6b7280;">
                    This is an automated alert from Prela drift detection.
                </p>
            </div>
        </body>
    </html>
    """

    return html_body, text_body


def format_drift_alert_slack(
    agent_name: str,
    severity: str,
    anomalies: list[dict[str, Any]],
    dashboard_url: str | None = None,
) -> dict[str, Any]:
    """Format drift alert as Slack message with blocks.

    Args:
        agent_name: Name of the agent.
        severity: Alert severity (low/medium/high/critical).
        anomalies: List of anomaly dictionaries.
        dashboard_url: Optional link to dashboard.

    Returns:
        Slack message payload with blocks.
    """
    severity_emojis = {
        "low": ":large_blue_circle:",
        "medium": ":warning:",
        "high": ":large_orange_circle:",
        "critical": ":red_circle:",
    }
    emoji = severity_emojis.get(severity, ":gray_circle:")

    # Build anomaly fields
    fields = []
    for anomaly in anomalies[:3]:  # Max 3 anomalies in fields
        fields.append({
            "type": "mrkdwn",
            "text": f"*{anomaly['metric_name']}*\n"
                   f"Baseline: {anomaly['baseline_mean']:.2f} → Current: {anomaly['current_value']:.2f} "
                   f"({anomaly['change_percent']:+.1f}%)"
        })

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} Drift Alert: {agent_name}",
            }
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Severity:*\n{severity.upper()}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Anomalies:*\n{len(anomalies)} detected"
                }
            ]
        },
        {
            "type": "section",
            "fields": fields
        }
    ]

    if dashboard_url:
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "View in Dashboard"
                    },
                    "url": dashboard_url,
                    "style": "primary"
                }
            ]
        })

    return {"blocks": blocks}
