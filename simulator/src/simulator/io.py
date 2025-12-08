"""JSON-based implementation of SimulationLogRepository."""

import json
from pathlib import Path

from core.data.ad_components.action import Action
from core.data.ad_components.state import VehicleState
from core.data.simulator.log import SimulationLog, SimulationStep
from core.interfaces.simulator import SimulationLogRepository


class JsonSimulationLogRepository(SimulationLogRepository):
    """JSON-based implementation of simulation log repository.

    This implementation saves and loads simulation logs in JSON format,
    providing human-readable storage for debugging and analysis.
    """

    def save(self, log: SimulationLog, file_path: Path) -> bool:
        """Save simulation log to JSON file.

        Args:
            log: Simulation log to save
            file_path: Output file path

        Returns:
            bool: 保存が成功した場合True
        """
        from dataclasses import asdict

        data = {
            "metadata": log.metadata,
            "steps": [
                {
                    "timestamp": s.timestamp,
                    "vehicle_state": asdict(s.vehicle_state),
                    "action": asdict(s.action),
                    # Handle observation and info if needed, skipping for simplicity in JSON
                }
                for s in log.steps
            ],
        }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        return True

    def load(self, file_path: Path) -> SimulationLog:
        """Load simulation log from JSON file.

        Args:
            file_path: Input file path

        Returns:
            SimulationLog object
        """
        with open(file_path) as f:
            data = json.load(f)

        steps = []
        for s in data.get("steps", []):
            step = SimulationStep(
                timestamp=s["timestamp"],
                vehicle_state=VehicleState(**s["vehicle_state"]),
                action=Action(**s["action"]),
                # Observation/info loading to be implemented if needed
            )
            steps.append(step)

        return SimulationLog(steps=steps, metadata=data.get("metadata", {}))
