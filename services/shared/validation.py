"""
Input Validation Framework

Centralized validation for all user inputs to prevent injection attacks
and ensure data integrity.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from fastapi import HTTPException


class InputValidator:
    """Centralized input validation."""

    # Regex patterns
    UUID_PATTERN = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE,
    )
    API_KEY_PATTERN = re.compile(r'^(prela_sk_|sk_)[A-Za-z0-9_-]{32,}$')
    PROJECT_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,100}$')
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    EXECUTION_ID_PATTERN = re.compile(r'^[a-zA-Z0-9\-_]{1,100}$')

    @staticmethod
    def validate_uuid(value: str, field_name: str = "id") -> str:
        """Validate UUID format.

        Args:
            value: UUID string to validate
            field_name: Field name for error messages

        Returns:
            Validated and normalized UUID (lowercase)

        Raises:
            HTTPException: If UUID format is invalid
        """
        if not InputValidator.UUID_PATTERN.match(value):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid {field_name} format. Expected UUID format."
            )
        return value.lower()

    @staticmethod
    def validate_project_id(project_id: str) -> str:
        """Validate project ID format.

        Args:
            project_id: Project ID to validate

        Returns:
            Validated project ID

        Raises:
            HTTPException: If project ID format is invalid
        """
        if not InputValidator.PROJECT_ID_PATTERN.match(project_id):
            raise HTTPException(
                status_code=400,
                detail="Invalid project ID format. Only alphanumeric, hyphens, and underscores allowed."
            )
        return project_id

    @staticmethod
    def validate_timestamp(timestamp: str, field_name: str = "timestamp") -> str:
        """Validate ISO 8601 timestamp format.

        Args:
            timestamp: ISO 8601 timestamp string
            field_name: Field name for error messages

        Returns:
            Validated timestamp string

        Raises:
            HTTPException: If timestamp format is invalid
        """
        try:
            # Try parsing as ISO 8601 with optional timezone
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return timestamp
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid {field_name} format. Expected ISO 8601 format (e.g., 2024-01-01T00:00:00Z)."
            )

    @staticmethod
    def validate_pagination(
        page: int,
        page_size: int,
        max_page_size: int = 100
    ) -> tuple[int, int]:
        """Validate pagination parameters.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page
            max_page_size: Maximum allowed page size

        Returns:
            Tuple of (page, page_size)

        Raises:
            HTTPException: If pagination parameters are invalid
        """
        if page < 1:
            raise HTTPException(
                status_code=400,
                detail="Page number must be >= 1"
            )
        if page_size < 1 or page_size > max_page_size:
            raise HTTPException(
                status_code=400,
                detail=f"Page size must be between 1 and {max_page_size}"
            )
        return page, page_size

    @staticmethod
    def validate_string_length(
        value: str,
        min_len: int,
        max_len: int,
        field_name: str
    ) -> str:
        """Validate string length.

        Args:
            value: String to validate
            min_len: Minimum allowed length
            max_len: Maximum allowed length
            field_name: Field name for error messages

        Returns:
            Validated string

        Raises:
            HTTPException: If string length is invalid
        """
        if len(value) < min_len or len(value) > max_len:
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} must be between {min_len} and {max_len} characters"
            )
        return value

    @staticmethod
    def sanitize_sql_identifier(identifier: str) -> str:
        """Sanitize SQL identifier (table/column name).

        Only allows alphanumeric characters and underscores.
        Must start with letter or underscore.

        Args:
            identifier: SQL identifier to sanitize

        Returns:
            Validated identifier

        Raises:
            HTTPException: If identifier format is invalid
        """
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
            raise HTTPException(
                status_code=400,
                detail="Invalid identifier format. Only alphanumeric and underscores allowed."
            )
        return identifier

    @staticmethod
    def validate_execution_id(execution_id: str) -> str:
        """Validate execution ID format.

        Args:
            execution_id: Execution ID to validate

        Returns:
            Validated execution ID

        Raises:
            HTTPException: If execution ID format is invalid
        """
        if not InputValidator.EXECUTION_ID_PATTERN.match(execution_id):
            raise HTTPException(
                status_code=400,
                detail="Invalid execution ID format. Only alphanumeric, hyphens, and underscores allowed."
            )
        return execution_id
