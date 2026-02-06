"""Anomaly detection service for drift alerts.

This module detects anomalies in agent behavior by comparing current metrics
against established baselines using statistical methods.
"""

import logging
import math
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from clickhouse_connect.driver import Client

logger = logging.getLogger(__name__)


class AnomalyMethod(str, Enum):
    """Anomaly detection method."""

    Z_SCORE = "z_score"
    IQR = "iqr"


class AnomalySeverity(str, Enum):
    """Anomaly severity level."""

    LOW = "low"  # 1σ - 2σ
    MEDIUM = "medium"  # 2σ - 3σ
    HIGH = "high"  # > 3σ
    CRITICAL = "critical"  # > 4σ


class AnomalyDetector:
    """Detect anomalies in agent behavior using statistical methods."""

    def __init__(
        self,
        client: Client,
        method: AnomalyMethod = AnomalyMethod.Z_SCORE,
        sensitivity: float = 2.0,
    ):
        """Initialize anomaly detector.

        Args:
            client: ClickHouse client instance.
            method: Detection method (z_score or iqr).
            sensitivity: Sensitivity threshold in standard deviations (default: 2.0).
        """
        self.client = client
        self.method = method
        self.sensitivity = sensitivity

    def detect_anomalies(
        self,
        project_id: str,
        agent_name: str,
        service_name: str,
        baseline: dict[str, Any],
        lookback_hours: int = 24,
    ) -> list[dict[str, Any]]:
        """Detect anomalies in recent agent behavior.

        Args:
            project_id: Project identifier.
            agent_name: Agent name to check.
            service_name: Service name.
            baseline: Baseline metrics dictionary.
            lookback_hours: Hours to look back for current metrics (default: 24).

        Returns:
            List of detected anomalies with details.
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=lookback_hours)

        logger.info(
            f"Detecting anomalies for agent '{agent_name}' "
            f"in project '{project_id}' "
            f"from {start_time} to {end_time}"
        )

        # Query current metrics
        query = """
            SELECT
                count() AS sample_size,
                avg(duration_ms) AS duration_mean,
                avg(JSONExtractUInt(attributes, 'llm.total_tokens')) AS token_usage_mean,
                avg(JSONExtractUInt(attributes, 'tool.call_count')) AS tool_calls_mean,
                avg(length(JSONExtractString(attributes, 'llm.response'))) AS response_length_mean,
                countIf(status = 'completed') / count() AS success_rate,
                avg(JSONExtractFloat(attributes, 'llm.cost_usd')) AS cost_mean
            FROM spans
            WHERE project_id = %(project_id)s
              AND JSONExtractString(attributes, 'agent.name') = %(agent_name)s
              AND service_name = %(service_name)s
              AND started_at >= %(start_time)s
              AND started_at <= %(end_time)s
              AND span_type = 'agent'
        """

        result = self.client.query(
            query,
            parameters={
                "project_id": project_id,
                "agent_name": agent_name,
                "service_name": service_name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
            },
        )

        rows = result.result_rows
        if not rows or rows[0][0] == 0:  # No recent data
            logger.info(f"No recent data for agent '{agent_name}'")
            return []

        row = rows[0]
        current_metrics = {
            "sample_size": int(row[0]),
            "duration_mean": float(row[1]) if row[1] else 0.0,
            "token_usage_mean": float(row[2]) if row[2] else 0.0,
            "tool_calls_mean": float(row[3]) if row[3] else 0.0,
            "response_length_mean": float(row[4]) if row[4] else 0.0,
            "success_rate": float(row[5]) if row[5] else 0.0,
            "cost_mean": float(row[6]) if row[6] else 0.0,
        }

        # Check each metric for anomalies
        anomalies = []

        # Duration
        if baseline["duration_stddev"] > 0:
            anomaly = self._check_metric_anomaly(
                metric_name="duration",
                current_value=current_metrics["duration_mean"],
                baseline_mean=baseline["duration_mean"],
                baseline_stddev=baseline["duration_stddev"],
                unit="ms",
                higher_is_worse=True,
            )
            if anomaly:
                anomaly["agent_name"] = agent_name
                anomaly["service_name"] = service_name
                anomaly["project_id"] = project_id
                anomaly["baseline_id"] = baseline["baseline_id"]
                anomaly["lookback_hours"] = lookback_hours
                anomaly["sample_size"] = current_metrics["sample_size"]
                anomalies.append(anomaly)

        # Token usage
        if baseline["token_usage_stddev"] > 0:
            anomaly = self._check_metric_anomaly(
                metric_name="token_usage",
                current_value=current_metrics["token_usage_mean"],
                baseline_mean=baseline["token_usage_mean"],
                baseline_stddev=baseline["token_usage_stddev"],
                unit="tokens",
                higher_is_worse=True,
            )
            if anomaly:
                anomaly["agent_name"] = agent_name
                anomaly["service_name"] = service_name
                anomaly["project_id"] = project_id
                anomaly["baseline_id"] = baseline["baseline_id"]
                anomaly["lookback_hours"] = lookback_hours
                anomaly["sample_size"] = current_metrics["sample_size"]
                anomalies.append(anomaly)

        # Tool calls
        if baseline["tool_calls_stddev"] > 0:
            anomaly = self._check_metric_anomaly(
                metric_name="tool_calls",
                current_value=current_metrics["tool_calls_mean"],
                baseline_mean=baseline["tool_calls_mean"],
                baseline_stddev=baseline["tool_calls_stddev"],
                unit="calls",
                higher_is_worse=False,  # Could be positive or negative
            )
            if anomaly:
                anomaly["agent_name"] = agent_name
                anomaly["service_name"] = service_name
                anomaly["project_id"] = project_id
                anomaly["baseline_id"] = baseline["baseline_id"]
                anomaly["lookback_hours"] = lookback_hours
                anomaly["sample_size"] = current_metrics["sample_size"]
                anomalies.append(anomaly)

        # Success rate
        success_rate_diff = abs(
            current_metrics["success_rate"] - baseline["success_rate"]
        )
        if success_rate_diff > 0.1:  # 10% change
            anomaly = {
                "metric_name": "success_rate",
                "current_value": current_metrics["success_rate"],
                "baseline_mean": baseline["success_rate"],
                "change_percent": (
                    (current_metrics["success_rate"] - baseline["success_rate"])
                    / baseline["success_rate"]
                    * 100
                    if baseline["success_rate"] > 0
                    else 0
                ),
                "severity": (
                    AnomalySeverity.CRITICAL
                    if success_rate_diff > 0.3
                    else AnomalySeverity.HIGH
                    if success_rate_diff > 0.2
                    else AnomalySeverity.MEDIUM
                ),
                "direction": (
                    "increased"
                    if current_metrics["success_rate"] > baseline["success_rate"]
                    else "decreased"
                ),
                "unit": "%",
                "agent_name": agent_name,
                "service_name": service_name,
                "project_id": project_id,
                "baseline_id": baseline["baseline_id"],
                "lookback_hours": lookback_hours,
                "sample_size": current_metrics["sample_size"],
            }
            anomalies.append(anomaly)

        # Cost
        if baseline["cost_mean"] > 0:
            anomaly = self._check_metric_anomaly(
                metric_name="cost",
                current_value=current_metrics["cost_mean"],
                baseline_mean=baseline["cost_mean"],
                baseline_stddev=baseline.get("cost_mean", 0) * 0.3,  # Estimate stddev
                unit="USD",
                higher_is_worse=True,
            )
            if anomaly:
                anomaly["agent_name"] = agent_name
                anomaly["service_name"] = service_name
                anomaly["project_id"] = project_id
                anomaly["baseline_id"] = baseline["baseline_id"]
                anomaly["lookback_hours"] = lookback_hours
                anomaly["sample_size"] = current_metrics["sample_size"]
                anomalies.append(anomaly)

        logger.info(f"Detected {len(anomalies)} anomalies for agent '{agent_name}'")
        return anomalies

    def _check_metric_anomaly(
        self,
        metric_name: str,
        current_value: float,
        baseline_mean: float,
        baseline_stddev: float,
        unit: str,
        higher_is_worse: bool = True,
    ) -> dict[str, Any] | None:
        """Check if a metric is anomalous using Z-score method.

        Args:
            metric_name: Name of the metric.
            current_value: Current metric value.
            baseline_mean: Baseline mean value.
            baseline_stddev: Baseline standard deviation.
            unit: Unit of measurement.
            higher_is_worse: Whether higher values indicate problems.

        Returns:
            Anomaly dictionary if detected, None otherwise.
        """
        if baseline_stddev == 0:
            return None

        # Calculate Z-score
        z_score = (current_value - baseline_mean) / baseline_stddev

        # Check if anomalous
        if abs(z_score) < self.sensitivity:
            return None

        # Determine severity
        abs_z = abs(z_score)
        if abs_z >= 4.0:
            severity = AnomalySeverity.CRITICAL
        elif abs_z >= 3.0:
            severity = AnomalySeverity.HIGH
        elif abs_z >= 2.0:
            severity = AnomalySeverity.MEDIUM
        else:
            severity = AnomalySeverity.LOW

        # Calculate change percentage
        change_percent = (
            (current_value - baseline_mean) / baseline_mean * 100
            if baseline_mean != 0
            else 0
        )

        return {
            "metric_name": metric_name,
            "current_value": current_value,
            "baseline_mean": baseline_mean,
            "baseline_stddev": baseline_stddev,
            "z_score": z_score,
            "severity": severity,
            "change_percent": change_percent,
            "direction": "increased" if z_score > 0 else "decreased",
            "unit": unit,
        }

    def analyze_root_causes(
        self,
        project_id: str,
        agent_name: str,
        service_name: str,
        anomalies: list[dict[str, Any]],
        lookback_hours: int = 24,
    ) -> list[dict[str, Any]]:
        """Analyze potential root causes for detected anomalies.

        Args:
            project_id: Project identifier.
            agent_name: Agent name.
            service_name: Service name.
            anomalies: List of detected anomalies.
            lookback_hours: Hours to analyze.

        Returns:
            List of potential root causes with confidence scores.
        """
        if not anomalies:
            return []

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=lookback_hours)

        root_causes = []

        # Check for model changes
        model_query = """
            SELECT DISTINCT JSONExtractString(attributes, 'llm.model') AS model
            FROM spans
            WHERE project_id = %(project_id)s
              AND JSONExtractString(attributes, 'agent.name') = %(agent_name)s
              AND service_name = %(service_name)s
              AND started_at >= %(start_time)s
              AND span_type IN ('agent', 'llm')
        """

        result = self.client.query(
            model_query,
            parameters={
                "project_id": project_id,
                "agent_name": agent_name,
                "service_name": service_name,
                "start_time": start_time.isoformat(),
            },
        )

        models = [row[0] for row in result.result_rows if row[0]]
        if len(models) > 1:
            root_causes.append(
                {
                    "type": "model_change",
                    "description": f"Model changed: {', '.join(models)}",
                    "confidence": 0.9,
                    "models": models,
                }
            )

        # Check for input complexity changes (token count increase)
        for anomaly in anomalies:
            if anomaly["metric_name"] == "token_usage" and anomaly["z_score"] > 0:
                root_causes.append(
                    {
                        "type": "input_complexity_increase",
                        "description": f"Input complexity increased by {anomaly['change_percent']:.1f}%",
                        "confidence": 0.8,
                        "change_percent": anomaly["change_percent"],
                    }
                )

        # Check for increased error rate
        for anomaly in anomalies:
            if (
                anomaly["metric_name"] == "success_rate"
                and anomaly["direction"] == "decreased"
            ):
                root_causes.append(
                    {
                        "type": "error_rate_increase",
                        "description": f"Error rate increased (success rate dropped {abs(anomaly['change_percent']):.1f}%)",
                        "confidence": 0.95,
                        "change_percent": anomaly["change_percent"],
                    }
                )

        return root_causes
