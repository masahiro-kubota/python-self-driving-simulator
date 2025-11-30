"""HTML dashboard generator implementation."""

import logging
from pathlib import Path
from typing import Any

from core.data import SimulationLog
from dashboard.injector import inject_simulation_data

logger = logging.getLogger(__name__)


class HTMLDashboardGenerator:
    """HTML dashboard generator using React template.

    This class implements the DashboardGenerator protocol and generates
    interactive HTML dashboards from simulation logs using a pre-built
    React template.
    """

    def generate(
        self,
        log: SimulationLog,
        output_path: Path,
        osm_path: Path | None = None,
    ) -> None:
        """Generate interactive HTML dashboard.

        Args:
            log: Simulation log containing trajectory and metadata
            output_path: Path where the generated HTML dashboard will be saved
            osm_path: Optional path to OSM map file for map visualization

        Raises:
            FileNotFoundError: If template file not found
            ValueError: If log data is invalid
        """
        # Prepare data in the format expected by the dashboard
        data: dict[str, Any] = {
            "metadata": {
                "controller": log.metadata.get("controller", "Unknown Controller"),
                "execution_time": log.metadata.get("execution_time", "Unknown Time"),
                **log.metadata,
            },
            "steps": [],
        }

        for step in log.steps:
            data["steps"].append(
                {
                    "timestamp": step.timestamp,
                    "x": step.vehicle_state.x,
                    "y": step.vehicle_state.y,
                    "z": getattr(step.vehicle_state, "z", 0.0),
                    "yaw": step.vehicle_state.yaw,
                    "velocity": step.vehicle_state.velocity,
                    "acceleration": step.action.acceleration,
                    "steering": step.action.steering,
                }
            )

        # Find template path
        # __file__ is dashboard/src/dashboard/generator.py
        # Go up to dashboard/ root
        package_root = Path(__file__).parent.parent.parent
        template_path = package_root / "frontend" / "dist" / "index.html"

        if not template_path.exists():
            msg = (
                f"Dashboard template not found at {template_path}\n"
                "Please build the dashboard first: cd dashboard/frontend && npm run build"
            )
            logger.error(msg)
            raise FileNotFoundError(msg)

        # Inject data into template
        inject_simulation_data(template_path, data, output_path, osm_path)
        logger.info("Dashboard saved to %s", output_path)


__all__ = ["HTMLDashboardGenerator"]
