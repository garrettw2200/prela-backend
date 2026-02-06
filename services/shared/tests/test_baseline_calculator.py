"""Tests for baseline calculator."""

import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from shared.baseline_calculator import BaselineCalculator


class TestBaselineCalculator:
    """Test baseline calculation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        self.calculator = BaselineCalculator(self.mock_client, window_days=7)

    def test_initialization(self):
        """Test calculator initialization."""
        assert self.calculator.client == self.mock_client
        assert self.calculator.window_days == 7

    def test_calculate_agent_baseline_success(self):
        """Test successful baseline calculation."""
        # Mock query result with sample data
        mock_result = MagicMock()
        mock_result.result_rows = [
            (
                100,  # sample_size
                5243.2,  # duration_mean
                823.5,  # duration_stddev
                4500.0,  # duration_p50
                7200.0,  # duration_p95
                8500.0,  # duration_p99
                2100.0,  # duration_min
                12000.0,  # duration_max
                1847.3,  # token_usage_mean
                245.8,  # token_usage_stddev
                1750.0,  # token_usage_p50
                2300.0,  # token_usage_p95
                3.2,  # tool_calls_mean
                1.1,  # tool_calls_stddev
                512.4,  # response_length_mean
                87.3,  # response_length_stddev
                0.982,  # success_rate
                3,  # error_count
                0.0234,  # cost_mean
                2.34,  # cost_total
            )
        ]
        self.mock_client.query.return_value = mock_result

        baseline = self.calculator.calculate_agent_baseline(
            "test-project", "researcher", "my-service"
        )

        assert baseline is not None
        assert baseline["sample_size"] == 100
        assert baseline["duration_mean"] == 5243.2
        assert baseline["success_rate"] == 0.982
        assert baseline["agent_name"] == "researcher"
        assert "baseline_id" in baseline

    def test_calculate_agent_baseline_insufficient_data(self):
        """Test baseline calculation with no data."""
        mock_result = MagicMock()
        mock_result.result_rows = [(0,) + (None,) * 19]  # sample_size = 0
        self.mock_client.query.return_value = mock_result

        baseline = self.calculator.calculate_agent_baseline(
            "test-project", "researcher", "my-service"
        )

        assert baseline is None

    def test_save_baseline(self):
        """Test baseline saving to ClickHouse."""
        baseline = {
            "baseline_id": str(uuid.uuid4()),
            "project_id": "test-project",
            "agent_name": "researcher",
            "service_name": "my-service",
            "window_start": datetime.utcnow() - timedelta(days=7),
            "window_end": datetime.utcnow(),
            "sample_size": 100,
            "duration_mean": 5243.2,
            "duration_stddev": 823.5,
            "duration_p50": 4500.0,
            "duration_p95": 7200.0,
            "duration_p99": 8500.0,
            "duration_min": 2100.0,
            "duration_max": 12000.0,
            "token_usage_mean": 1847.3,
            "token_usage_stddev": 245.8,
            "token_usage_p50": 1750.0,
            "token_usage_p95": 2300.0,
            "tool_calls_mean": 3.2,
            "tool_calls_stddev": 1.1,
            "response_length_mean": 512.4,
            "response_length_stddev": 87.3,
            "success_rate": 0.982,
            "error_count": 3,
            "cost_mean": 0.0234,
            "cost_total": 2.34,
        }

        self.calculator.save_baseline(baseline)

        # Verify insert was called
        self.mock_client.insert.assert_called_once()
        call_args = self.mock_client.insert.call_args
        assert call_args[0][0] == "agent_baselines"
        assert len(call_args[0][1]) == 1  # One row

    def test_get_latest_baseline(self):
        """Test retrieving latest baseline."""
        mock_result = MagicMock()
        now = datetime.utcnow()
        mock_result.result_rows = [
            (
                str(uuid.uuid4()),  # baseline_id
                "test-project",  # project_id
                "researcher",  # agent_name
                "my-service",  # service_name
                now - timedelta(days=7),  # window_start
                now,  # window_end
                100,  # sample_size
                5243.2,  # duration_mean
                823.5,  # duration_stddev
                4500.0,  # duration_p50
                7200.0,  # duration_p95
                8500.0,  # duration_p99
                2100.0,  # duration_min
                12000.0,  # duration_max
                1847.3,  # token_usage_mean
                245.8,  # token_usage_stddev
                1750.0,  # token_usage_p50
                2300.0,  # token_usage_p95
                3.2,  # tool_calls_mean
                1.1,  # tool_calls_stddev
                512.4,  # response_length_mean
                87.3,  # response_length_stddev
                0.982,  # success_rate
                3,  # error_count
                0.0234,  # cost_mean
                2.34,  # cost_total
                now,  # created_at
                now,  # updated_at
            )
        ]
        self.mock_client.query.return_value = mock_result

        baseline = self.calculator.get_latest_baseline(
            "test-project", "researcher", "my-service"
        )

        assert baseline is not None
        assert baseline["agent_name"] == "researcher"
        assert baseline["sample_size"] == 100
        assert baseline["duration_mean"] == 5243.2

    def test_get_latest_baseline_not_found(self):
        """Test retrieving baseline when none exists."""
        mock_result = MagicMock()
        mock_result.result_rows = []
        self.mock_client.query.return_value = mock_result

        baseline = self.calculator.get_latest_baseline(
            "test-project", "researcher", "my-service"
        )

        assert baseline is None

    def test_calculate_all_baselines(self):
        """Test calculating baselines for all agents."""
        # Mock agents query
        agents_result = MagicMock()
        agents_result.result_rows = [
            ("researcher", "my-service"),
            ("writer", "my-service"),
        ]

        # Mock baseline calculation query
        baseline_result = MagicMock()
        baseline_result.result_rows = [
            (100,) + (5243.2, 823.5, 4500.0, 7200.0, 8500.0, 2100.0, 12000.0)
            + (1847.3, 245.8, 1750.0, 2300.0)
            + (3.2, 1.1, 512.4, 87.3, 0.982, 3, 0.0234, 2.34)
        ]

        self.mock_client.query.side_effect = [agents_result, baseline_result, baseline_result]

        count = self.calculator.calculate_all_baselines("test-project")

        assert count == 2
        assert self.mock_client.insert.call_count == 2
