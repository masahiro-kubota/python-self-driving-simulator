"""HTML dashboard generator implementation."""

import logging
from pathlib import Path
from typing import Any

from core.data.dashboard import DashboardData
from core.interfaces import DashboardGenerator

from dashboard.injector import inject_simulation_data

logger = logging.getLogger(__name__)


def downsample_indices(total_count: int, max_steps: int = 1000) -> list[int]:
    """Calculate indices for uniform downsampling.

    Args:
        total_count: Total number of data points
        max_steps: Maximum number of steps to keep (default: 1000)

    Returns:
        List of indices to selected
    """
    if total_count <= max_steps:
        return list(range(total_count))

    step_size = total_count / max_steps
    indices = [int(i * step_size) for i in range(max_steps)]

    # Always include the last step
    if indices[-1] != total_count - 1:
        indices[-1] = total_count - 1

    return indices


class HTMLDashboardGenerator(DashboardGenerator):
    """HTML dashboard generator using React template.

    This class implements the DashboardGenerator interface and generates
    interactive HTML dashboards from experiment results using a pre-built
    React template.
    """

    def generate(
        self,
        data: "DashboardData",
        output_path: Path,
        osm_path: Path,
    ) -> Path:
        """Generate interactive HTML dashboard.

        Args:
            data: Structured simulation data (column-oriented)
            output_path: Path where the generated HTML dashboard will be saved
            osm_path: Optional path to OSM map file for map visualization

        Returns:
            Path: Path to the generated dashboard file

        Raises:
            FileNotFoundError: If template file not found
        """
        vehicle_params = data["vehicle_params"]

        # Prepare vehicle params for injection (ensure clean dict)
        if hasattr(vehicle_params, "model_dump"):
            v_params = vehicle_params.model_dump()
        elif isinstance(vehicle_params, dict):
            v_params = vehicle_params
        else:
            # Fallback wrapper
            v_params = {
                k: getattr(vehicle_params, k)
                for k in ["width", "wheelbase", "front_overhang", "rear_overhang"]
            }

        # Calculate logical length if missing
        if "length" not in v_params:
            v_params["length"] = (
                v_params.get("wheelbase", 0)
                + v_params.get("front_overhang", 0)
                + v_params.get("rear_overhang", 0)
            )

        # Prepare injection data
        injection_data: dict[str, Any] = {
            "metadata": {
                "controller": "Unknown",
            },
            "vehicle_params": v_params,
            "steps": [],
            "obstacles": [],
        }

        # Merge metadata
        if "metadata" in data:
            injection_data["metadata"].update(data["metadata"])
            for k, v in injection_data["metadata"].items():
                if isinstance(v, (dict, list)) and k != "obstacles":
                    injection_data["metadata"][k] = str(v)

        # Merge obstacles
        if data.get("obstacles"):
            injection_data["obstacles"] = data["obstacles"]

        # Downsample and Convert to Legacy Row-Oriented format for Frontend
        timestamps = data["timestamps"]
        vehicle = data["vehicle"]
        action = data.get("action", {})

        total_steps = len(timestamps)
        logger.info("Original steps count: %d", total_steps)

        indices = downsample_indices(total_steps, max_steps=1000)
        logger.info("Downsampled steps count: %d", len(indices))

        steps = []
        for i in indices:
            step = {
                "timestamp": timestamps[i],
                # Vehicle State
                "x": vehicle["x"][i],
                "y": vehicle["y"][i],
                "z": vehicle["z"][i] if "z" in vehicle and i < len(vehicle["z"]) else 0.0,
                "yaw": vehicle["yaw"][i],
                "velocity": vehicle["velocity"][i],
                # Action (ROS message compatible)
                "acceleration": action["acceleration"][i] if "acceleration" in action else 0.0,
                "steering": action["steering"][i] if "steering" in action else 0.0,
                # placeholders for legacy frontend expectations if any
                "lidar_scan": None,
            }
            steps.append(step)

        injection_data["steps"] = steps

        # Find template path
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
        inject_simulation_data(template_path, injection_data, output_path, osm_path)
        logger.info("Dashboard saved to %s", output_path)
        return output_path
