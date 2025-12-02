"""Simulation log data structures."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.data.ad_components.action import Action
from core.data.ad_components.log import ADComponentLog
from core.data.ad_components.state import VehicleState


@dataclass
class SimulationStep:
    """Single step in a simulation.

    Attributes:
        timestamp: タイムスタンプ [s]
        vehicle_state: 車両状態
        action: 実行されたアクション
        ad_component_log: ADコンポーネントのログ（任意）
        info: 追加情報（任意）
    """

    timestamp: float
    vehicle_state: VehicleState
    action: Action
    ad_component_log: ADComponentLog | None = None
    info: dict[str, Any] | None = None


class SimulationLog:
    """Log of a simulation run."""

    def __init__(
        self, steps: list[SimulationStep] | None = None, metadata: dict[str, Any] | None = None
    ) -> None:
        """Initialize simulation log.

        Args:
            steps: Optional list of simulation steps
            metadata: Optional metadata about the simulation (e.g. track name, config)
        """
        self.metadata = metadata or {}
        self.steps: list[SimulationStep] = steps or []

    def add_step(self, step: SimulationStep) -> None:
        """Add a step to the log.

        Args:
            step: Simulation step
        """
        self.steps.append(step)

    def save(self, file_path: str | Path) -> None:
        """Save log to file (JSON format).

        Args:
            file_path: Output file path
        """
        import json

        data = {
            "metadata": self.metadata,
            "steps": [
                {
                    "timestamp": s.timestamp,
                    "vehicle_state": asdict(s.vehicle_state),
                    "action": asdict(s.action),
                    # Handle observation and info if needed, skipping for simplicity in JSON
                }
                for s in self.steps
            ],
        }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, file_path: str | Path) -> "SimulationLog":
        """Load log from file.

        Args:
            file_path: Input file path

        Returns:
            SimulationLog object
        """
        import json

        with open(file_path) as f:
            data = json.load(f)

        log = cls(metadata=data.get("metadata"))

        for s in data.get("steps", []):
            step = SimulationStep(
                timestamp=s["timestamp"],
                vehicle_state=VehicleState(**s["vehicle_state"]),
                action=Action(**s["action"]),
                # Observation/info loading to be implemented if needed
            )
            log.add_step(step)

        return log
