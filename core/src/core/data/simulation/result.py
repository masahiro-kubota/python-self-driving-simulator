"""Simulation result data structure."""

from dataclasses import dataclass, field
from typing import Any

from core.data.ad_components.state import VehicleState
from core.data.simulation.log import SimulationLog


@dataclass
class SimulationResult:
    """シミュレーション結果."""

    success: bool
    reason: str
    final_state: VehicleState
    log: SimulationLog
    metrics: dict[str, Any] = field(default_factory=dict)
