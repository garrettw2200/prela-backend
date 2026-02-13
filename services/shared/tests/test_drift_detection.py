"""Tests for background drift detection service."""

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# We need to mock the imports before importing the module
# since it imports from shared which may not be available in test env


@pytest.fixture
def mock_clickhouse_client():
    """Create a mock ClickHouse client."""
    client = MagicMock()
    return client


@pytest.fixture
def mock_baseline():
    """Create a sample baseline dict."""
    return {
        "baseline_id": str(uuid.uuid4()),
        "project_id": "test-project",
        "agent_name": "researcher",
        "service_name": "my-service",
        "window_start": datetime.now(timezone.utc) - timedelta(days=7),
        "window_end": datetime.now(timezone.utc),
        "sample_size": 100,
        "duration_mean": 5000.0,
        "duration_stddev": 800.0,
        "duration_p50": 4500.0,
        "duration_p95": 7200.0,
        "duration_p99": 8500.0,
        "duration_min": 2100.0,
        "duration_max": 12000.0,
        "token_usage_mean": 1800.0,
        "token_usage_stddev": 250.0,
        "token_usage_p50": 1750.0,
        "token_usage_p95": 2300.0,
        "tool_calls_mean": 3.0,
        "tool_calls_stddev": 1.0,
        "response_length_mean": 500.0,
        "response_length_stddev": 80.0,
        "success_rate": 0.95,
        "error_count": 5,
        "cost_mean": 0.025,
        "cost_total": 2.5,
    }


@pytest.fixture
def mock_anomalies():
    """Create sample anomaly dicts."""
    from shared.anomaly_detector import AnomalySeverity

    return [
        {
            "metric_name": "duration",
            "current_value": 8500.0,
            "baseline_mean": 5000.0,
            "baseline_stddev": 800.0,
            "z_score": 4.375,
            "severity": AnomalySeverity.CRITICAL,
            "change_percent": 70.0,
            "direction": "increased",
            "unit": "ms",
            "agent_name": "researcher",
            "service_name": "my-service",
            "project_id": "test-project",
            "baseline_id": "baseline-123",
            "lookback_hours": 24,
            "sample_size": 50,
        },
        {
            "metric_name": "token_usage",
            "current_value": 2800.0,
            "baseline_mean": 1800.0,
            "baseline_stddev": 250.0,
            "z_score": 4.0,
            "severity": AnomalySeverity.CRITICAL,
            "change_percent": 55.6,
            "direction": "increased",
            "unit": "tokens",
            "agent_name": "researcher",
            "service_name": "my-service",
            "project_id": "test-project",
            "baseline_id": "baseline-123",
            "lookback_hours": 24,
            "sample_size": 50,
        },
    ]


class TestGetActiveProjects:
    """Test project discovery."""

    @pytest.mark.asyncio
    async def test_returns_project_ids(self):
        """Test getting active projects from ClickHouse."""
        mock_result = MagicMock()
        mock_result.result_rows = [
            ("project-1",),
            ("project-2",),
            ("project-3",),
        ]

        with patch(
            "shared.clickhouse.get_clickhouse_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client.query.return_value = mock_result
            mock_get_client.return_value = mock_client

            # Import after patching
            from app.services.drift_detection import _get_active_projects

            projects = await _get_active_projects()
            assert len(projects) == 3
            assert "project-1" in projects

    @pytest.mark.asyncio
    async def test_handles_empty_results(self):
        """Test empty project list."""
        mock_result = MagicMock()
        mock_result.result_rows = []

        with patch(
            "shared.clickhouse.get_clickhouse_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client.query.return_value = mock_result
            mock_get_client.return_value = mock_client

            from app.services.drift_detection import _get_active_projects

            projects = await _get_active_projects()
            assert len(projects) == 0


class TestGetRecentAlertIds:
    """Test deduplication via recent alerts."""

    @pytest.mark.asyncio
    async def test_returns_recently_alerted_agents(self, mock_clickhouse_client):
        """Test that recently alerted agent names are returned."""
        mock_result = MagicMock()
        mock_result.result_rows = [
            ("agent-a",),
            ("agent-b",),
        ]
        mock_clickhouse_client.query.return_value = mock_result

        from app.services.drift_detection import _get_recent_alert_ids

        result = await _get_recent_alert_ids(
            mock_clickhouse_client, "test-project"
        )
        assert result == {"agent-a", "agent-b"}

    @pytest.mark.asyncio
    async def test_handles_no_recent_alerts(self, mock_clickhouse_client):
        """Test empty result when no recent alerts exist."""
        mock_result = MagicMock()
        mock_result.result_rows = []
        mock_clickhouse_client.query.return_value = mock_result

        from app.services.drift_detection import _get_recent_alert_ids

        result = await _get_recent_alert_ids(
            mock_clickhouse_client, "test-project"
        )
        assert result == set()


class TestDetectDriftForProject:
    """Test per-project drift detection."""

    @pytest.mark.asyncio
    async def test_skips_agents_with_recent_alerts(self, mock_baseline):
        """Test that agents with recent alerts are skipped."""
        # Mock: recent alerts include "researcher"
        recent_alerts_result = MagicMock()
        recent_alerts_result.result_rows = [("researcher",)]

        # Mock: agents with activity
        agents_result = MagicMock()
        agents_result.result_rows = [("researcher", "my-service")]

        mock_client = MagicMock()
        mock_client.query.side_effect = [
            recent_alerts_result,  # _get_recent_alert_ids
            agents_result,  # agents query
        ]

        with patch(
            "shared.clickhouse.get_clickhouse_client",
            return_value=mock_client,
        ):
            from app.services.drift_detection import detect_drift_for_project

            result = await detect_drift_for_project("test-project")
            assert result["agents_checked"] == 0
            assert result["alerts_created"] == 0

    @pytest.mark.asyncio
    async def test_skips_agents_without_baseline(self):
        """Test that agents without baselines are skipped."""
        recent_alerts_result = MagicMock()
        recent_alerts_result.result_rows = []

        agents_result = MagicMock()
        agents_result.result_rows = [("new-agent", "my-service")]

        mock_client = MagicMock()
        mock_client.query.side_effect = [
            recent_alerts_result,
            agents_result,
        ]

        with (
            patch(
                "shared.clickhouse.get_clickhouse_client",
                return_value=mock_client,
            ),
            patch(
                "app.services.drift_detection.BaselineCalculator"
            ) as MockCalc,
            patch(
                "app.services.drift_detection.AnomalyDetector"
            ),
        ):
            mock_calc_instance = MockCalc.return_value
            mock_calc_instance.get_latest_baseline.return_value = None

            from app.services.drift_detection import detect_drift_for_project

            result = await detect_drift_for_project("test-project")
            assert result["agents_checked"] == 1
            assert result["alerts_created"] == 0

    @pytest.mark.asyncio
    async def test_creates_alert_when_anomalies_detected(
        self, mock_baseline, mock_anomalies
    ):
        """Test that alerts are created when anomalies are found."""
        recent_alerts_result = MagicMock()
        recent_alerts_result.result_rows = []

        agents_result = MagicMock()
        agents_result.result_rows = [("researcher", "my-service")]

        mock_client = MagicMock()
        mock_client.query.side_effect = [
            recent_alerts_result,
            agents_result,
        ]

        with (
            patch(
                "shared.clickhouse.get_clickhouse_client",
                return_value=mock_client,
            ),
            patch(
                "app.services.drift_detection.BaselineCalculator"
            ) as MockCalc,
            patch(
                "app.services.drift_detection.AnomalyDetector"
            ) as MockDetector,
            patch(
                "app.services.drift_detection._send_notifications",
                new_callable=AsyncMock,
            ) as mock_notify,
        ):
            mock_calc_instance = MockCalc.return_value
            mock_calc_instance.get_latest_baseline.return_value = mock_baseline

            mock_detector_instance = MockDetector.return_value
            mock_detector_instance.detect_anomalies.return_value = mock_anomalies
            mock_detector_instance.analyze_root_causes.return_value = [
                {
                    "type": "input_complexity_increase",
                    "description": "Input complexity increased by 55.6%",
                    "confidence": 0.8,
                }
            ]

            from app.services.drift_detection import detect_drift_for_project

            result = await detect_drift_for_project("test-project")

            assert result["agents_checked"] == 1
            assert result["anomalies_found"] == 2
            assert result["alerts_created"] == 1

            # Verify ClickHouse inserts were called
            assert mock_client.insert.call_count == 2  # drift_alerts + analysis_results

            # Verify notifications were called
            mock_notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_alert_when_no_anomalies(self, mock_baseline):
        """Test that no alert is created when behavior is normal."""
        recent_alerts_result = MagicMock()
        recent_alerts_result.result_rows = []

        agents_result = MagicMock()
        agents_result.result_rows = [("researcher", "my-service")]

        mock_client = MagicMock()
        mock_client.query.side_effect = [
            recent_alerts_result,
            agents_result,
        ]

        with (
            patch(
                "shared.clickhouse.get_clickhouse_client",
                return_value=mock_client,
            ),
            patch(
                "app.services.drift_detection.BaselineCalculator"
            ) as MockCalc,
            patch(
                "app.services.drift_detection.AnomalyDetector"
            ) as MockDetector,
        ):
            mock_calc_instance = MockCalc.return_value
            mock_calc_instance.get_latest_baseline.return_value = mock_baseline

            mock_detector_instance = MockDetector.return_value
            mock_detector_instance.detect_anomalies.return_value = []

            from app.services.drift_detection import detect_drift_for_project

            result = await detect_drift_for_project("test-project")
            assert result["agents_checked"] == 1
            assert result["anomalies_found"] == 0
            assert result["alerts_created"] == 0
            assert mock_client.insert.call_count == 0

    @pytest.mark.asyncio
    async def test_continues_on_per_agent_error(self, mock_baseline):
        """Test that errors on one agent don't crash the whole project check."""
        recent_alerts_result = MagicMock()
        recent_alerts_result.result_rows = []

        agents_result = MagicMock()
        agents_result.result_rows = [
            ("agent-1", "svc-1"),
            ("agent-2", "svc-2"),
        ]

        mock_client = MagicMock()
        mock_client.query.side_effect = [
            recent_alerts_result,
            agents_result,
        ]

        with (
            patch(
                "shared.clickhouse.get_clickhouse_client",
                return_value=mock_client,
            ),
            patch(
                "app.services.drift_detection.BaselineCalculator"
            ) as MockCalc,
            patch(
                "app.services.drift_detection.AnomalyDetector"
            ) as MockDetector,
        ):
            mock_calc_instance = MockCalc.return_value
            # First agent raises error, second returns no baseline
            mock_calc_instance.get_latest_baseline.side_effect = [
                Exception("ClickHouse timeout"),
                None,
            ]

            from app.services.drift_detection import detect_drift_for_project

            # Should not raise
            result = await detect_drift_for_project("test-project")
            assert result["agents_checked"] == 2


class TestDetectDriftAllProjects:
    """Test multi-project drift detection."""

    @pytest.mark.asyncio
    async def test_iterates_all_projects(self):
        """Test that all active projects are checked."""
        with (
            patch(
                "app.services.drift_detection._get_active_projects",
                new_callable=AsyncMock,
                return_value=["proj-1", "proj-2"],
            ),
            patch(
                "app.services.drift_detection.detect_drift_for_project",
                new_callable=AsyncMock,
                return_value={
                    "agents_checked": 1,
                    "anomalies_found": 0,
                    "alerts_created": 0,
                },
            ) as mock_detect,
        ):
            from app.services.drift_detection import detect_drift_all_projects

            result = await detect_drift_all_projects()

            assert result["projects_checked"] == 2
            assert mock_detect.call_count == 2

    @pytest.mark.asyncio
    async def test_continues_on_project_error(self):
        """Test that errors on one project don't crash the loop."""
        call_count = 0

        async def detect_side_effect(project_id):
            nonlocal call_count
            call_count += 1
            if project_id == "proj-1":
                raise Exception("DB error")
            return {
                "agents_checked": 1,
                "anomalies_found": 0,
                "alerts_created": 0,
            }

        with (
            patch(
                "app.services.drift_detection._get_active_projects",
                new_callable=AsyncMock,
                return_value=["proj-1", "proj-2"],
            ),
            patch(
                "app.services.drift_detection.detect_drift_for_project",
                side_effect=detect_side_effect,
            ),
        ):
            from app.services.drift_detection import detect_drift_all_projects

            result = await detect_drift_all_projects()

            assert result["projects_checked"] == 2
            assert result["errors"] == 1
            assert call_count == 2


class TestRefreshBaselinesAllProjects:
    """Test baseline refresh across projects."""

    @pytest.mark.asyncio
    async def test_refreshes_all_projects(self):
        """Test that baselines are refreshed for all active projects."""
        mock_client = MagicMock()

        with (
            patch(
                "app.services.drift_detection._get_active_projects",
                new_callable=AsyncMock,
                return_value=["proj-1"],
            ),
            patch(
                "shared.clickhouse.get_clickhouse_client",
                return_value=mock_client,
            ),
            patch(
                "app.services.drift_detection.BaselineCalculator"
            ) as MockCalc,
        ):
            mock_calc_instance = MockCalc.return_value
            mock_calc_instance.calculate_all_baselines.return_value = 3

            from app.services.drift_detection import (
                refresh_baselines_all_projects,
            )

            result = await refresh_baselines_all_projects()

            assert result["projects_checked"] == 1
            assert result["baselines_calculated"] == 3


class TestSendNotifications:
    """Test notification sending."""

    @pytest.mark.asyncio
    async def test_sends_email_when_rule_matches(self, mock_clickhouse_client):
        """Test that email is sent when an alert rule matches."""
        mock_result = MagicMock()
        mock_result.result_rows = [
            (
                "rule-1",
                "My Rule",
                True,  # notify_email
                ["user@example.com"],  # email_addresses
                False,  # notify_slack
                None,  # slack_webhook_url
                None,  # slack_channel
            )
        ]
        mock_clickhouse_client.query.return_value = mock_result

        with (
            patch(
                "app.services.drift_detection.send_email_notification",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_send_email,
            patch(
                "app.services.drift_detection.format_drift_alert_email",
                return_value=("<html>alert</html>", "alert text"),
            ),
        ):
            from app.services.drift_detection import _send_notifications

            await _send_notifications(
                mock_clickhouse_client,
                "test-project",
                "researcher",
                "critical",
                [{"metric_name": "duration", "current_value": 8000, "baseline_mean": 5000, "change_percent": 60.0}],
            )

            mock_send_email.assert_called_once()

    @pytest.mark.asyncio
    async def test_sends_slack_when_rule_matches(self, mock_clickhouse_client):
        """Test that Slack notification is sent when a rule matches."""
        mock_result = MagicMock()
        mock_result.result_rows = [
            (
                "rule-1",
                "My Rule",
                False,  # notify_email
                [],  # email_addresses
                True,  # notify_slack
                "https://hooks.slack.com/test",  # slack_webhook_url
                "#alerts",  # slack_channel
            )
        ]
        mock_clickhouse_client.query.return_value = mock_result

        with (
            patch(
                "app.services.drift_detection.send_slack_notification",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_send_slack,
            patch(
                "app.services.drift_detection.format_drift_alert_slack",
                return_value={"blocks": []},
            ),
        ):
            from app.services.drift_detection import _send_notifications

            await _send_notifications(
                mock_clickhouse_client,
                "test-project",
                "researcher",
                "high",
                [{"metric_name": "cost", "current_value": 0.05, "baseline_mean": 0.02, "change_percent": 150.0}],
            )

            mock_send_slack.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_notification_when_no_rules_match(
        self, mock_clickhouse_client
    ):
        """Test that no notification is sent when no rules match."""
        mock_result = MagicMock()
        mock_result.result_rows = []
        mock_clickhouse_client.query.return_value = mock_result

        with (
            patch(
                "app.services.drift_detection.send_email_notification",
                new_callable=AsyncMock,
            ) as mock_email,
            patch(
                "app.services.drift_detection.send_slack_notification",
                new_callable=AsyncMock,
            ) as mock_slack,
        ):
            from app.services.drift_detection import _send_notifications

            await _send_notifications(
                mock_clickhouse_client,
                "test-project",
                "researcher",
                "low",
                [],
            )

            mock_email.assert_not_called()
            mock_slack.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_notification_error_gracefully(
        self, mock_clickhouse_client
    ):
        """Test that notification errors don't propagate."""
        mock_clickhouse_client.query.side_effect = Exception("DB error")

        from app.services.drift_detection import _send_notifications

        # Should not raise
        await _send_notifications(
            mock_clickhouse_client,
            "test-project",
            "researcher",
            "high",
            [],
        )
