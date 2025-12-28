from typing import TypeVar

T = TypeVar("T")


class TopicSlot[T]:
    """Topic slot holding data and sequence number."""

    def __init__(self, initial_value: T | None = None):
        """Initialize slot.

        Args:
            initial_value: Initial value (optional)
        """
        self._data: T | None = initial_value
        self._seq: int = 0

    @property
    def data(self) -> T | None:
        """Get current data (read-only)."""
        return self._data

    @property
    def seq(self) -> int:
        """Get current sequence number (read-only)."""
        return self._seq

    def update(self, value: T) -> None:
        """Update data and increment sequence number.

        Args:
            value: New data value
        """
        self._data = value
        self._seq += 1

    def __repr__(self) -> str:
        return f"TopicSlot(seq={self._seq}, data={self._data})"
