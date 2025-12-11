"""HTML dashboard generator implementation."""

import logging
from pathlib import Path
from typing import Any

from core.data.experiment import ExperimentResult
from core.interfaces import DashboardGenerator
from dashboard.injector import inject_simulation_data

logger = logging.getLogger(__name__)


class HTMLDashboardGenerator(DashboardGenerator):
    """HTML dashboard generator using React template.

    This class implements the DashboardGenerator interface and generates
    interactive HTML dashboards from experiment results using a pre-built
    React template.
    """

    def generate(
        self,
        result: ExperimentResult,
        output_path: Path,
        osm_path: Path | None = None,
        vehicle_params: dict[str, Any] | None = None,
    ) -> Path:
        """Generate interactive HTML dashboard.

        Args:
            result: Experiment result containing simulation results and metadata
            output_path: Path where the generated HTML dashboard will be saved
            osm_path: Optional path to OSM map file for map visualization
            vehicle_params: Optional vehicle parameters dict with keys: width, wheelbase,
                          front_overhang, rear_overhang. If provided, these will be used
                          instead of extracting from metadata.

        Returns:
            Path: Path to the generated dashboard file

        Raises:
            FileNotFoundError: If template file not found
            ValueError: If result data is invalid or contains no simulation results
        """
        # For now, use the first simulation result
        # TODO: Support multiple simulation results display
        if not result.simulation_results:
            raise ValueError("ExperimentResult contains no simulation results")

        log = result.simulation_results[0].log
        print(f"DEBUG generator.py: log.metadata has obstacles: {'obstacles' in log.metadata}")
        print(f"DEBUG generator.py: log.metadata keys: {list(log.metadata.keys())}")

        # Prepare data in the format expected by the dashboard
        # Sanitize metadata to avoid React rendering errors with nested objects
        metadata = log.metadata.copy()

        # Special handling for controller if it is a config dict
        if (
            "controller" in metadata
            and isinstance(metadata["controller"], dict)
            and "type" in metadata["controller"]
        ):
            metadata["controller"] = metadata["controller"]["type"]

        # Generic sanitization: convert dicts/lists to strings
        # Exclude obstacles from sanitized_metadata (will be added separately to data)
        sanitized_metadata = {}
        for k, v in metadata.items():
            if k == "obstacles":
                # Skip obstacles - will be added directly to data["obstacles"]
                continue
            elif isinstance(v, dict | list):
                sanitized_metadata[k] = str(v)
            else:
                sanitized_metadata[k] = v

        # Extract vehicle parameters for dashboard visualization
        # Use provided vehicle_params if available, otherwise extract from metadata
        if vehicle_params:
            # Use provided parameters, with fallback to defaults
            dashboard_vehicle_params = {
                "width": vehicle_params.get("width", 1.8),
                "wheelbase": vehicle_params.get("wheelbase", 2.5),
                "front_overhang": vehicle_params.get("front_overhang", 0.9),
                "rear_overhang": vehicle_params.get("rear_overhang", 1.1),
            }
            # Calculate length from components
            length = (
                dashboard_vehicle_params["wheelbase"]
                + dashboard_vehicle_params["front_overhang"]
                + dashboard_vehicle_params["rear_overhang"]
            )
            dashboard_vehicle_params["length"] = length
        else:
            # Fallback to metadata extraction
            dashboard_vehicle_params = {
                "width": sanitized_metadata.get("width", 1.8),
                "length": sanitized_metadata.get("length", 4.5),
                "wheelbase": sanitized_metadata.get("wheelbase", 2.5),
                "front_overhang": sanitized_metadata.get("front_overhang", 0.9),
                "rear_overhang": sanitized_metadata.get("rear_overhang", 1.1),
            }

        data: dict[str, Any] = {
            "metadata": {
                "experiment_name": result.experiment_name,
                "experiment_type": result.experiment_type,
                "execution_time": result.execution_time.isoformat(),
                "controller": sanitized_metadata.get("controller", "Unknown Controller"),
                **sanitized_metadata,
            },
            "vehicle_params": dashboard_vehicle_params,
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

        # Extract obstacles from metadata if available
        if "obstacles" in metadata and isinstance(metadata["obstacles"], list):
            data["obstacles"] = metadata["obstacles"]
            logger.info("Added %d obstacles to dashboard data", len(metadata["obstacles"]))
        else:
            logger.warning("No obstacles found in metadata")

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
        return output_path


__all__ = ["HTMLDashboardGenerator"]
