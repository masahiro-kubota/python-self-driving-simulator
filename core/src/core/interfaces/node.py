"""Node interface and SimulationContext."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.data import Action, Observation, VehicleState
from core.data.ad_components import Trajectory


@dataclass
class SimulationContext:
    """Simulation execution context.

    Acts as a shared memory between nodes.
    """

    current_time: float = 0.0
    sim_state: VehicleState | None = None  # Ground truth state from simulator
    vehicle_state: VehicleState | None = None  # Perceived state for planner
    trajectory: Trajectory | None = None
    action: Action | None = None
    observation: Observation | None = None
    done: bool = False
    done_reason: str = "max_steps"
    success: bool = False


class Node(ABC):
    """Base class for schedulable nodes."""

    def __init__(self, name: str, rate_hz: float):
        self.name = name
        self.rate_hz = rate_hz
        self.period = 1.0 / rate_hz
        self.next_time = 0.0

    def should_run(self, sim_time: float) -> bool:
        """Check if node should run at current time."""
        return sim_time + 1e-9 >= self.next_time

    @abstractmethod
    def on_run(self, context: SimulationContext) -> None:
        """Execute node logic."""
        raise NotImplementedError
