"""Simulation log data structures."""

from dataclasses import dataclass
from typing import Any

from core.data.ad_components.state import VehicleState
from core.data.autoware import AckermannControlCommand


@dataclass
class SimulationStep:
    """Single step in a simulation.

    Attributes:
        timestamp: タイムスタンプ [s]
        vehicle_state: 車両状態
        action: 実行されたアクション
        info: 追加情報（任意）
    """

    timestamp: float
    vehicle_state: VehicleState
    action: AckermannControlCommand
    info: dict[str, Any] | None = None


@dataclass
class SimulationLog:
    """Log of a simulation run.

    Attributes:
        steps: List of simulation steps
        metadata: Metadata about the simulation (e.g. track name, config)
    """

    steps: list[SimulationStep]
    metadata: dict[str, Any]

    def save(self, path: str | Any) -> None:
        """Save simulation log to a JSON file.

        Args:
            path: Path to save the JSON file
        """
        import dataclasses
        import json
        from pathlib import Path

        import numpy as np
        from pydantic import BaseModel

        class NormalizedEncoder(json.JSONEncoder):
            def default(self, obj: Any) -> Any:
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, BaseModel):
                    return obj.model_dump()
                return super().default(obj)

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Manually construct dict since asdict() fails on Pydantic models
        data = {
            "metadata": self.metadata,
            "steps": [
                {
                    "timestamp": s.timestamp,
                    "vehicle_state": dataclasses.asdict(s.vehicle_state),
                    "action": s.action.model_dump(),
                    "info": s.info,
                }
                for s in self.steps
            ],
        }
        with path_obj.open("w") as f:
            json.dump(data, f, cls=NormalizedEncoder, indent=2)
