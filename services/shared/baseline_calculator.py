"""Baseline calculation service for drift detection.

This module calculates rolling window baselines for agent behavior metrics
to enable anomaly detection and drift alerts.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any

from clickhouse_connect.driver import Client

logger = logging.getLogger(__name__)


class BaselineCalculator:
    """Calculate agent behavior baselines from historical trace data."""

    def __init__(self, client: Client, window_days: int = 7):
        """Initialize baseline calculator.

        Args:
            client: ClickHouse client instance.
            window_days: Rolling window size in days (default: 7).
        """
        self.client = client
        self.window_days = window_days

    def calculate_agent_baseline(
        self,
        project_id: str,
        agent_name: str,
        service_name: str,
        end_time: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Calculate baseline metrics for a specific agent.

        Args:
            project_id: Project identifier.
            agent_name: Agent name to calculate baseline for.
            service_name: Service name.
            end_time: End of rolling window (default: now).

        Returns:
            Dictionary with baseline metrics, or None if insufficient data.
        """
        if end_time is None:
            end_time = datetime.utcnow()

        window_start = end_time - timedelta(days=self.window_days)

        logger.info(
            f"Calculating baseline for agent '{agent_name}' "
            f"in project '{project_id}' "
            f"from {window_start} to {end_time}"
        )

        # Query agent executions in window
        query = """
            SELECT
                count() AS sample_size,

                -- Duration metrics
                avg(duration_ms) AS duration_mean,
                stddevPop(duration_ms) AS duration_stddev,
                quantile(0.5)(duration_ms) AS duration_p50,
                quantile(0.95)(duration_ms) AS duration_p95,
                quantile(0.99)(duration_ms) AS duration_p99,
                min(duration_ms) AS duration_min,
                max(duration_ms) AS duration_max,

                -- Token usage metrics (from LLM spans)
                avg(JSONExtractUInt(attributes, 'llm.total_tokens')) AS token_usage_mean,
                stddevPop(JSONExtractUInt(attributes, 'llm.total_tokens')) AS token_usage_stddev,
                quantile(0.5)(JSONExtractUInt(attributes, 'llm.total_tokens')) AS token_usage_p50,
                quantile(0.95)(JSONExtractUInt(attributes, 'llm.total_tokens')) AS token_usage_p95,

                -- Tool call metrics
                avg(JSONExtractUInt(attributes, 'tool.call_count')) AS tool_calls_mean,
                stddevPop(JSONExtractUInt(attributes, 'tool.call_count')) AS tool_calls_stddev,

                -- Response length metrics
                avg(length(JSONExtractString(attributes, 'llm.response'))) AS response_length_mean,
                stddevPop(length(JSONExtractString(attributes, 'llm.response'))) AS response_length_stddev,

                -- Success metrics
                countIf(status = 'completed') / count() AS success_rate,
                countIf(status = 'error') AS error_count,

                -- Cost metrics
                avg(JSONExtractFloat(attributes, 'llm.cost_usd')) AS cost_mean,
                sum(JSONExtractFloat(attributes, 'llm.cost_usd')) AS cost_total

            FROM spans
            WHERE project_id = %(project_id)s
              AND JSONExtractString(attributes, 'agent.name') = %(agent_name)s
              AND service_name = %(service_name)s
              AND started_at >= %(window_start)s
              AND started_at <= %(end_time)s
              AND span_type = 'agent'
        """

        result = self.client.query(
            query,
            parameters={
                "project_id": project_id,
                "agent_name": agent_name,
                "service_name": service_name,
                "window_start": window_start.isoformat(),
                "end_time": end_time.isoformat(),
            },
        )

        rows = result.result_rows
        if not rows or rows[0][0] == 0:  # sample_size == 0
            logger.warning(
                f"Insufficient data for baseline calculation: "
                f"agent='{agent_name}', project='{project_id}'"
            )
            return None

        # Parse results
        row = rows[0]
        baseline = {
            "baseline_id": str(uuid.uuid4()),
            "project_id": project_id,
            "agent_name": agent_name,
            "service_name": service_name,
            "window_start": window_start,
            "window_end": end_time,
            "sample_size": int(row[0]),
            # Duration metrics
            "duration_mean": float(row[1]) if row[1] else 0.0,
            "duration_stddev": float(row[2]) if row[2] else 0.0,
            "duration_p50": float(row[3]) if row[3] else 0.0,
            "duration_p95": float(row[4]) if row[4] else 0.0,
            "duration_p99": float(row[5]) if row[5] else 0.0,
            "duration_min": float(row[6]) if row[6] else 0.0,
            "duration_max": float(row[7]) if row[7] else 0.0,
            # Token usage
            "token_usage_mean": float(row[8]) if row[8] else 0.0,
            "token_usage_stddev": float(row[9]) if row[9] else 0.0,
            "token_usage_p50": float(row[10]) if row[10] else 0.0,
            "token_usage_p95": float(row[11]) if row[11] else 0.0,
            # Tool calls
            "tool_calls_mean": float(row[12]) if row[12] else 0.0,
            "tool_calls_stddev": float(row[13]) if row[13] else 0.0,
            # Response length
            "response_length_mean": float(row[14]) if row[14] else 0.0,
            "response_length_stddev": float(row[15]) if row[15] else 0.0,
            # Success
            "success_rate": float(row[16]) if row[16] else 0.0,
            "error_count": int(row[17]),
            # Cost
            "cost_mean": float(row[18]) if row[18] else 0.0,
            "cost_total": float(row[19]) if row[19] else 0.0,
        }

        logger.info(
            f"Baseline calculated for agent '{agent_name}': "
            f"sample_size={baseline['sample_size']}, "
            f"duration_mean={baseline['duration_mean']:.2f}ms, "
            f"success_rate={baseline['success_rate']:.2%}"
        )

        return baseline

    def save_baseline(self, baseline: dict[str, Any]) -> None:
        """Save baseline to ClickHouse.

        Args:
            baseline: Baseline metrics dictionary.
        """
        insert_query = """
            INSERT INTO agent_baselines (
                baseline_id, project_id, agent_name, service_name,
                window_start, window_end, sample_size,
                duration_mean, duration_stddev, duration_p50, duration_p95, duration_p99,
                duration_min, duration_max,
                token_usage_mean, token_usage_stddev, token_usage_p50, token_usage_p95,
                tool_calls_mean, tool_calls_stddev,
                response_length_mean, response_length_stddev,
                success_rate, error_count,
                cost_mean, cost_total
            )
            VALUES
        """

        self.client.insert(
            "agent_baselines",
            [
                [
                    baseline["baseline_id"],
                    baseline["project_id"],
                    baseline["agent_name"],
                    baseline["service_name"],
                    baseline["window_start"],
                    baseline["window_end"],
                    baseline["sample_size"],
                    baseline["duration_mean"],
                    baseline["duration_stddev"],
                    baseline["duration_p50"],
                    baseline["duration_p95"],
                    baseline["duration_p99"],
                    baseline["duration_min"],
                    baseline["duration_max"],
                    baseline["token_usage_mean"],
                    baseline["token_usage_stddev"],
                    baseline["token_usage_p50"],
                    baseline["token_usage_p95"],
                    baseline["tool_calls_mean"],
                    baseline["tool_calls_stddev"],
                    baseline["response_length_mean"],
                    baseline["response_length_stddev"],
                    baseline["success_rate"],
                    baseline["error_count"],
                    baseline["cost_mean"],
                    baseline["cost_total"],
                ]
            ],
        )

        logger.info(
            f"Saved baseline {baseline['baseline_id']} for agent '{baseline['agent_name']}'"
        )

    def get_latest_baseline(
        self, project_id: str, agent_name: str, service_name: str
    ) -> dict[str, Any] | None:
        """Retrieve the most recent baseline for an agent.

        Args:
            project_id: Project identifier.
            agent_name: Agent name.
            service_name: Service name.

        Returns:
            Latest baseline dictionary, or None if not found.
        """
        query = """
            SELECT
                baseline_id, project_id, agent_name, service_name,
                window_start, window_end, sample_size,
                duration_mean, duration_stddev, duration_p50, duration_p95, duration_p99,
                duration_min, duration_max,
                token_usage_mean, token_usage_stddev, token_usage_p50, token_usage_p95,
                tool_calls_mean, tool_calls_stddev,
                response_length_mean, response_length_stddev,
                success_rate, error_count,
                cost_mean, cost_total,
                created_at, updated_at
            FROM agent_baselines
            WHERE project_id = %(project_id)s
              AND agent_name = %(agent_name)s
              AND service_name = %(service_name)s
            ORDER BY window_end DESC
            LIMIT 1
        """

        result = self.client.query(
            query,
            parameters={
                "project_id": project_id,
                "agent_name": agent_name,
                "service_name": service_name,
            },
        )

        rows = result.result_rows
        if not rows:
            return None

        row = rows[0]
        return {
            "baseline_id": row[0],
            "project_id": row[1],
            "agent_name": row[2],
            "service_name": row[3],
            "window_start": row[4],
            "window_end": row[5],
            "sample_size": int(row[6]),
            "duration_mean": float(row[7]),
            "duration_stddev": float(row[8]),
            "duration_p50": float(row[9]),
            "duration_p95": float(row[10]),
            "duration_p99": float(row[11]),
            "duration_min": float(row[12]),
            "duration_max": float(row[13]),
            "token_usage_mean": float(row[14]),
            "token_usage_stddev": float(row[15]),
            "token_usage_p50": float(row[16]),
            "token_usage_p95": float(row[17]),
            "tool_calls_mean": float(row[18]),
            "tool_calls_stddev": float(row[19]),
            "response_length_mean": float(row[20]),
            "response_length_stddev": float(row[21]),
            "success_rate": float(row[22]),
            "error_count": int(row[23]),
            "cost_mean": float(row[24]),
            "cost_total": float(row[25]),
            "created_at": row[26],
            "updated_at": row[27],
        }

    def calculate_all_baselines(self, project_id: str) -> int:
        """Calculate baselines for all agents in a project.

        Args:
            project_id: Project identifier.

        Returns:
            Number of baselines calculated.
        """
        # Get unique agents from recent traces
        query = """
            SELECT DISTINCT
                JSONExtractString(attributes, 'agent.name') AS agent_name,
                service_name
            FROM spans
            WHERE project_id = %(project_id)s
              AND span_type = 'agent'
              AND started_at >= now() - INTERVAL %(window_days)s DAY
              AND JSONHas(attributes, 'agent.name')
        """

        result = self.client.query(
            query,
            parameters={"project_id": project_id, "window_days": self.window_days},
        )

        agents = [(row[0], row[1]) for row in result.result_rows if row[0]]

        count = 0
        for agent_name, service_name in agents:
            try:
                baseline = self.calculate_agent_baseline(
                    project_id, agent_name, service_name
                )
                if baseline:
                    self.save_baseline(baseline)
                    count += 1
            except Exception as e:
                logger.error(
                    f"Failed to calculate baseline for agent '{agent_name}': {e}"
                )

        logger.info(f"Calculated {count} baselines for project '{project_id}'")
        return count
