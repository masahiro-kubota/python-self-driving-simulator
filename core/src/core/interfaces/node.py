"""Node interface."""

from abc import ABC, abstractmethod
from typing import Any, Protocol

from core.data import ComponentConfig, NodeExecutionResult
from core.data.frame_data import FrameData
from core.data.node_io import NodeIO


class FrameDataProtocol(Protocol):
    """Protocol for dynamic FrameData types."""


# Type variable for ComponentConfig
class Node[T: ComponentConfig](ABC):
    """Base class for schedulable nodes."""

    def __init__(
        self,
        name: str,
        rate_hz: float,
        config: T,
    ):
        """Initialize node.

        Args:
            name: Node name
            rate_hz: Execution frequency in Hz
            config: Validated configuration (Pydantic model instance)
        """
        self.name = name
        self.rate_hz = rate_hz
        self.period = 1.0 / rate_hz
        self.next_time = 0.0
        self.frame_data: FrameDataProtocol | None = None
        self.config: T = config

    @classmethod
    def from_dict(
        cls,
        rate_hz: float,
        config_class: type[T],
        config_dict: dict[str, Any],
        **kwargs: Any,
    ) -> "Node[T]":
        """Create node from configuration dictionary.

        This is a helper method for creating nodes from YAML/dict configs.

        Args:
            rate_hz: Execution frequency in Hz
            config_class: Pydantic model class for configuration validation
            config_dict: Configuration dictionary
            **kwargs: Additional arguments to pass to __init__

        Returns:
            Instantiated Node with validated configuration
        """
        config = config_class(**config_dict)
        return cls(rate_hz=rate_hz, config=config, **kwargs)

    @abstractmethod
    def get_node_io(self) -> NodeIO:
        """Get node I/O specification.

        Returns:
            NodeIO specification defining inputs and outputs
        """
        raise NotImplementedError

    def set_frame_data(self, frame_data: FrameData) -> None:
        """Set simulation frame data.

        Args:
            frame_data: Frame data to set
        """
        self.frame_data = frame_data

    def should_run(self, sim_time: float) -> bool:
        """Check if node should run at current time.

        Args:
            sim_time: Current simulation time

        Returns:
            True if node should run
        """
        return sim_time + 1e-9 >= self.next_time

    def update_next_time(self, current_time: float) -> None:
        """Update next execution time.

        Args:
            current_time: Current simulation time
        """
        self.next_time = current_time + self.period

    def on_init(self) -> None:
        """Initialize node resources.

        Called once before the first execution.
        Override to perform initialization tasks.
        """

    def on_shutdown(self) -> None:
        """Clean up node resources.

        Called once after the last execution.
        Override to perform cleanup tasks.
        """

    @abstractmethod
    def on_run(self, current_time: float) -> NodeExecutionResult:
        """Execute node logic.

        Args:
            current_time: Current simulation time

        Returns:
            NodeExecutionResult indicating execution status
        """
        raise NotImplementedError
