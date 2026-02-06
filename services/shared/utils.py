"""
Shared utility functions for Prela backend services.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def safe_json_parse(
    json_str: str | None,
    default: Any = None,
    field_name: str = "field"
) -> Any:
    """Safely parse JSON with error handling.

    Prevents application crashes from malformed JSON data stored in database.

    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails
        field_name: Field name for logging

    Returns:
        Parsed JSON or default value
    """
    if not json_str:
        return default

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON for {field_name}: {e}")
        return default
    except Exception as e:
        logger.error(f"Unexpected error parsing {field_name}: {e}")
        return default


def safe_json_serialize(data: Any, field_name: str = "field") -> str:
    """Safely serialize data to JSON string.

    Args:
        data: Data to serialize
        field_name: Field name for logging

    Returns:
        JSON string or appropriate default (empty object/array)
    """
    try:
        return json.dumps(data)
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to serialize {field_name}: {e}, using default")
        # Return appropriate default based on type
        if isinstance(data, dict):
            return "{}"
        elif isinstance(data, list):
            return "[]"
        else:
            return json.dumps(str(data))
