"""Simulation execution context."""

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
