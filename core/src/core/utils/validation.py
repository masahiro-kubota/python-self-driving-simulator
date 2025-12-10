"""Validation utilities for configuration parameters."""

from typing import Any


def validate_config(
    params: dict[str, Any],
    required_keys: list[str] | None = None,
    allow_none: bool = True,
) -> None:
    """Validate configuration parameters.

    Args:
        params: Parameters to validate
        required_keys: List of keys that must be present
        allow_none: If False, values cannot be None (for required keys)

    Raises:
        ValueError: If validation fails
    """
    if required_keys is None:
        return

    missing_keys = [key for key in required_keys if key not in params]
    if missing_keys:
        raise ValueError(f"Missing required parameters: {missing_keys}")

    if not allow_none:
        none_keys = [key for key in required_keys if params.get(key) is None]
        if none_keys:
            raise ValueError(f"Parameters cannot be None: {none_keys}")
