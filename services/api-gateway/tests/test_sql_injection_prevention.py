"""
Unit tests for SQL injection prevention and input validation.

Tests all validation functions and replay endpoints for SQL injection attempts.
"""

import sys
from pathlib import Path

# Add shared module to path
shared_path = Path(__file__).parent.parent.parent / "shared"
if str(shared_path) not in sys.path:
    sys.path.insert(0, str(shared_path))

import pytest
from fastapi import HTTPException

# Import directly from validation module to avoid loading settings
from validation import InputValidator


class TestInputValidation:
    """Test input validation functions."""

    def test_validate_uuid_valid(self):
        """Test UUID validation with valid UUIDs."""
        valid_uuids = [
            "550e8400-e29b-41d4-a716-446655440000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
            "00000000-0000-0000-0000-000000000000",
        ]
        for uuid in valid_uuids:
            result = InputValidator.validate_uuid(uuid, "test_id")
            assert result == uuid.lower()

    def test_validate_uuid_invalid(self):
        """Test UUID validation with invalid formats."""
        invalid_uuids = [
            "not-a-uuid",
            "550e8400-e29b-41d4-a716",  # Too short
            "550e8400-e29b-41d4-a716-446655440000-extra",  # Too long
            "'; DROP TABLE users; --",  # SQL injection attempt
            "550e8400e29b41d4a716446655440000",  # Missing hyphens
            "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",  # Invalid characters
        ]
        for uuid in invalid_uuids:
            with pytest.raises(HTTPException) as exc_info:
                InputValidator.validate_uuid(uuid, "test_id")
            assert exc_info.value.status_code == 400
            assert "Invalid test_id format" in exc_info.value.detail

    def test_validate_project_id_valid(self):
        """Test project ID validation with valid IDs."""
        valid_ids = [
            "my-project",
            "project_123",
            "PROJECT-ABC-123",
            "a",
            "a" * 100,  # Max length
        ]
        for project_id in valid_ids:
            result = InputValidator.validate_project_id(project_id)
            assert result == project_id

    def test_validate_project_id_invalid(self):
        """Test project ID validation with SQL injection attempts."""
        invalid_ids = [
            "'; DROP TABLE traces; --",
            "project'; DELETE FROM users WHERE '1'='1",
            "' OR 1=1 --",
            "' UNION SELECT * FROM secrets --",
            "project\"; DROP TABLE traces; --",
            "project/../../../etc/passwd",
            "project%00admin",
            "a" * 101,  # Too long
            "",  # Empty
            "project with spaces",
            "project@#$%",
            "project<script>",
        ]
        for project_id in invalid_ids:
            with pytest.raises(HTTPException) as exc_info:
                InputValidator.validate_project_id(project_id)
            assert exc_info.value.status_code == 400
            assert "Invalid project ID format" in exc_info.value.detail

    def test_validate_timestamp_valid(self):
        """Test timestamp validation with valid ISO 8601 formats."""
        valid_timestamps = [
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00+00:00",
            "2024-12-31T23:59:59.999999Z",
            "2024-06-15T12:30:45",
        ]
        for timestamp in valid_timestamps:
            result = InputValidator.validate_timestamp(timestamp, "test_time")
            assert result == timestamp

    def test_validate_timestamp_invalid(self):
        """Test timestamp validation with SQL injection attempts."""
        invalid_timestamps = [
            "'; DROP TABLE traces; --",
            "2024-13-01T00:00:00Z",  # Invalid month
            "2024-01-32T00:00:00Z",  # Invalid day
            "not-a-timestamp",
            "' OR 1=1 --",
            # Note: Python's fromisoformat accepts "2024-01-01" and "1234567890"
            # as valid formats, so we don't test those
        ]
        for timestamp in invalid_timestamps:
            with pytest.raises(HTTPException) as exc_info:
                InputValidator.validate_timestamp(timestamp, "test_time")
            assert exc_info.value.status_code == 400
            assert "Invalid test_time format" in exc_info.value.detail

    def test_validate_pagination_valid(self):
        """Test pagination validation with valid parameters."""
        valid_params = [
            (1, 10),
            (1, 100),
            (100, 50),
        ]
        for page, page_size in valid_params:
            result_page, result_size = InputValidator.validate_pagination(page, page_size)
            assert result_page == page
            assert result_size == page_size

    def test_validate_pagination_invalid(self):
        """Test pagination validation with invalid parameters."""
        invalid_params = [
            (0, 10),  # Page < 1
            (-1, 10),  # Negative page
            (1, 0),  # Page size < 1
            (1, 101),  # Page size > max (100)
            (1, -10),  # Negative page size
        ]
        for page, page_size in invalid_params:
            with pytest.raises(HTTPException) as exc_info:
                InputValidator.validate_pagination(page, page_size)
            assert exc_info.value.status_code == 400

    def test_validate_string_length_valid(self):
        """Test string length validation."""
        result = InputValidator.validate_string_length("test", 1, 10, "field")
        assert result == "test"

    def test_validate_string_length_invalid(self):
        """Test string length validation with invalid lengths."""
        # Too short
        with pytest.raises(HTTPException) as exc_info:
            InputValidator.validate_string_length("", 1, 10, "field")
        assert exc_info.value.status_code == 400

        # Too long
        with pytest.raises(HTTPException) as exc_info:
            InputValidator.validate_string_length("a" * 11, 1, 10, "field")
        assert exc_info.value.status_code == 400

    def test_sanitize_sql_identifier_valid(self):
        """Test SQL identifier sanitization with valid identifiers."""
        valid_identifiers = [
            "table_name",
            "column_name",
            "_private",
            "Table123",
        ]
        for identifier in valid_identifiers:
            result = InputValidator.sanitize_sql_identifier(identifier)
            assert result == identifier

    def test_sanitize_sql_identifier_invalid(self):
        """Test SQL identifier sanitization prevents SQL injection."""
        invalid_identifiers = [
            "table; DROP TABLE users; --",
            "table_name' OR '1'='1",
            "table-name",
            "table.name",
            "table name",
            "123table",  # Starts with number
            "table@name",
            "table#name",
        ]
        for identifier in invalid_identifiers:
            with pytest.raises(HTTPException) as exc_info:
                InputValidator.sanitize_sql_identifier(identifier)
            assert exc_info.value.status_code == 400


class TestSQLInjectionScenarios:
    """Test common SQL injection attack scenarios."""

    def test_classic_sql_injection(self):
        """Test classic SQL injection attempts are blocked."""
        injection_attempts = [
            "' OR '1'='1",
            "'; DROP TABLE traces; --",
            "' OR 1=1 --",
            "admin'--",
            "' OR 'x'='x",
        ]
        for attempt in injection_attempts:
            with pytest.raises(HTTPException):
                InputValidator.validate_project_id(attempt)

    def test_union_based_injection(self):
        """Test UNION-based SQL injection attempts are blocked."""
        injection_attempts = [
            "' UNION SELECT * FROM users --",
            "' UNION SELECT password FROM admin --",
            "' UNION ALL SELECT NULL, NULL, NULL --",
        ]
        for attempt in injection_attempts:
            with pytest.raises(HTTPException):
                InputValidator.validate_project_id(attempt)

    def test_blind_sql_injection(self):
        """Test blind SQL injection attempts are blocked."""
        injection_attempts = [
            "' AND SLEEP(5) --",
            "' AND (SELECT * FROM (SELECT(SLEEP(5)))a) --",
            "' WAITFOR DELAY '00:00:05' --",
        ]
        for attempt in injection_attempts:
            with pytest.raises(HTTPException):
                InputValidator.validate_project_id(attempt)

    def test_time_based_injection(self):
        """Test time-based SQL injection attempts via timestamps."""
        injection_attempts = [
            "2024-01-01' OR SLEEP(5) --",
            "'; WAITFOR DELAY '00:00:05' --",
            "' AND (SELECT SLEEP(5)) --",
        ]
        for attempt in injection_attempts:
            with pytest.raises(HTTPException):
                InputValidator.validate_timestamp(attempt)

    def test_stacked_queries(self):
        """Test stacked query injection attempts are blocked."""
        injection_attempts = [
            "project'; DELETE FROM users; --",
            "'; INSERT INTO admin VALUES('hacker', 'password'); --",
            "'; UPDATE users SET is_admin=1 WHERE username='hacker'; --",
        ]
        for attempt in injection_attempts:
            with pytest.raises(HTTPException):
                InputValidator.validate_project_id(attempt)

    def test_comment_based_injection(self):
        """Test comment-based injection attempts are blocked."""
        injection_attempts = [
            "project'/*",
            "project'-- ",
            "project'#",
            "project'; --",
        ]
        for attempt in injection_attempts:
            with pytest.raises(HTTPException):
                InputValidator.validate_project_id(attempt)


class TestExecutionIDValidation:
    """Test execution ID validation."""

    def test_validate_execution_id_valid(self):
        """Test execution ID validation with valid IDs."""
        valid_ids = [
            "exec-123-456",
            "execution_abc_xyz",
            "EXEC-ABC-123",
            "a",
            "a" * 100,
        ]
        for exec_id in valid_ids:
            result = InputValidator.validate_execution_id(exec_id)
            assert result == exec_id

    def test_validate_execution_id_invalid(self):
        """Test execution ID validation blocks SQL injection."""
        invalid_ids = [
            "'; DROP TABLE replay_executions; --",
            "exec'; DELETE FROM users --",
            "' OR 1=1 --",
            "exec with spaces",
            "exec@#$%",
            "a" * 101,
            "",
        ]
        for exec_id in invalid_ids:
            with pytest.raises(HTTPException) as exc_info:
                InputValidator.validate_execution_id(exec_id)
            assert exc_info.value.status_code == 400


def test_parameterized_query_pattern():
    """
    Verify that parameterized queries use %(param)s syntax.

    This is a documentation test to ensure developers understand
    the correct pattern for ClickHouse parameterized queries.
    """
    # Correct pattern (parameterized)
    correct_query = """
        SELECT * FROM traces
        WHERE project_id = %(project_id)s
          AND started_at >= %(since)s
        LIMIT %(limit)s
    """
    assert "%(project_id)s" in correct_query
    assert "%(since)s" in correct_query
    assert "%(limit)s" in correct_query

    # Incorrect patterns (vulnerable to SQL injection)
    # These should NEVER be used:
    # f"SELECT * FROM traces WHERE project_id = '{project_id}'"
    # f"SELECT * FROM traces WHERE started_at >= '{since}'"
    # f"LIMIT {limit} OFFSET {offset}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
