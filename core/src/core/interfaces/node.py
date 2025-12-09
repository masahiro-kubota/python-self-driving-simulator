"""Node interface."""

from abc import ABC, abstractmethod

from core.data.simulation_context import SimulationContext


class Node(ABC):
    """Base class for schedulable nodes."""

    def __init__(self, name: str, rate_hz: float):
        self.name = name
        self.rate_hz = rate_hz
        self.period = 1.0 / rate_hz
        self.next_time = 0.0
        self.context: SimulationContext | None = None

    def set_context(self, context: SimulationContext) -> None:
        """Set simulation context."""
        self.context = context

    def should_run(self, sim_time: float) -> bool:
        """Check if node should run at current time."""
        return sim_time + 1e-9 >= self.next_time

    @abstractmethod
    def on_run(self, current_time: float) -> bool:
        """Execute node logic.

        Args:
            current_time: Current simulation time

        Returns:
            bool: True if execution was successful
        """
        raise NotImplementedError
