"""Simulation log data structures."""

from dataclasses import dataclass
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

        class NormalizedEncoder(json.JSONEncoder):
            def default(self, obj: Any) -> Any:
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        data = dataclasses.asdict(self)
        with path_obj.open("w") as f:
            json.dump(data, f, cls=NormalizedEncoder, indent=2)
