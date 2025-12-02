"""Simulation result data structure."""

from dataclasses import dataclass, field
from typing import Any

from core.data.simulation_log import SimulationLog
from core.data.state import VehicleState


@dataclass
class SimulationResult:
    """シミュレーション結果."""

    success: bool
    reason: str
    final_state: VehicleState
    log: SimulationLog
    metrics: dict[str, Any] = field(default_factory=dict)
