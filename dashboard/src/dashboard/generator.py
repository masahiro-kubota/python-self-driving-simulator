"""HTML dashboard generator implementation."""

import logging
from pathlib import Path
from typing import Any

from core.data.experiment import ExperimentResult
from core.interfaces import DashboardGenerator

from dashboard.injector import inject_simulation_data

logger = logging.getLogger(__name__)


def downsample_steps(steps_data: list[dict], max_steps: int = 1000) -> list[dict]:
    """Downsample step data to reduce dashboard file size.

    Args:
        steps_data: Full step data
        max_steps: Maximum number of steps to keep (default: 1000)

    Returns:
        Downsampled step data with uniform sampling
    """
    if len(steps_data) <= max_steps:
        return steps_data

    # Uniform sampling
    step_size = len(steps_data) / max_steps
    indices = [int(i * step_size) for i in range(max_steps)]

    # Always include the last step
    if indices[-1] != len(steps_data) - 1:
        indices[-1] = len(steps_data) - 1

    return [steps_data[i] for i in indices]


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
        osm_path: Path,
        vehicle_params: dict[str, Any] | Any,  # Any to support VehicleParameters
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
            if isinstance(v, dict | list):
                sanitized_metadata[k] = str(v)
            else:
                sanitized_metadata[k] = v

        # Extract vehicle parameters for dashboard visualization
        if hasattr(vehicle_params, "model_dump"):
            # Pydantic model
            dashboard_vehicle_params = {
                "width": vehicle_params.width,
                "wheelbase": vehicle_params.wheelbase,
                "front_overhang": vehicle_params.front_overhang,
                "rear_overhang": vehicle_params.rear_overhang,
            }
        else:
            # Strictly expect dict with all keys if no model provided
            required_keys = ["width", "wheelbase", "front_overhang", "rear_overhang"]
            missing = [k for k in required_keys if k not in vehicle_params]
            if missing:
                raise ValueError(f"Missing required vehicle parameters: {missing}")

            # Create explicit dict to ensure JSON serializability (handle DictConfig)
            dashboard_vehicle_params = {
                "width": float(vehicle_params["width"]),
                "wheelbase": float(vehicle_params["wheelbase"]),
                "front_overhang": float(vehicle_params["front_overhang"]),
                "rear_overhang": float(vehicle_params["rear_overhang"]),
            }

        # Calculate length from components
        length = (
            dashboard_vehicle_params["wheelbase"]
            + dashboard_vehicle_params["front_overhang"]
            + dashboard_vehicle_params["rear_overhang"]
        )
        dashboard_vehicle_params["length"] = length

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

        # Try to read steps from MCAP artifact first
        mcap_artifact = next(
            (a for a in result.artifacts if a.local_path and a.local_path.suffix == ".mcap"), None
        )

        steps_data = []
        if mcap_artifact and mcap_artifact.local_path and mcap_artifact.local_path.exists():
            logger.info("Reading steps from MCAP artifact: %s", mcap_artifact.local_path)
            try:
                import json

                from mcap.reader import make_reader

                with open(mcap_artifact.local_path, "rb") as f:
                    reader = make_reader(f)
                    for schema, channel, message in reader.iter_messages(
                        topics=["/simulation/step"]
                    ):
                        # The message is a String(data=JSON) from LoggerNode
                        wrapper = json.loads(message.data)
                        if "data" not in wrapper:
                            continue

                        step_dict = json.loads(wrapper["data"])

                        # LiDAR data is not needed for dashboard visualization
                        steps_data.append(
                            {
                                "timestamp": step_dict["timestamp"],
                                "x": step_dict["vehicle_state"]["x"]
                                if "vehicle_state" in step_dict
                                else 0.0,
                                "y": step_dict["vehicle_state"]["y"]
                                if "vehicle_state" in step_dict
                                else 0.0,
                                "z": step_dict["vehicle_state"].get("z", 0.0)
                                if "vehicle_state" in step_dict
                                else 0.0,
                                "yaw": step_dict["vehicle_state"]["yaw"]
                                if "vehicle_state" in step_dict
                                else 0.0,
                                "velocity": step_dict["vehicle_state"]["velocity"]
                                if "vehicle_state" in step_dict
                                else 0.0,
                                "acceleration": step_dict["action"]["acceleration"]
                                if "action" in step_dict
                                else 0.0,
                                "steering": step_dict["action"]["steering"]
                                if "action" in step_dict
                                else 0.0,
                                "lidar_scan": None,
                                "ad_component_log": step_dict.get("ad_component_log"),
                            }
                        )
                logger.info("Loaded %d steps from MCAP", len(steps_data))
            except Exception as e:
                logger.warning("Failed to read MCAP file: %s. Fallback to memory.", e)
                steps_data = []

        # Fallback to in-memory log if MCAP failed or was not found
        if not steps_data:
            logger.info("Using in-memory simulation log for steps")
            for step in log.steps:
                steps_data.append(
                    {
                        "timestamp": step.timestamp,
                        "x": step.vehicle_state.x,
                        "y": step.vehicle_state.y,
                        "z": getattr(step.vehicle_state, "z", 0.0),
                        "yaw": step.vehicle_state.yaw,
                        "velocity": step.vehicle_state.velocity,
                        "acceleration": step.action.acceleration,
                        "steering": step.action.steering,
                        "ad_component_log": {
                            "component_type": step.ad_component_log.component_type,
                            "data": step.ad_component_log.data,
                        }
                        if step.ad_component_log
                        else None,
                        "lidar_scan": None,  # LiDAR data not needed for dashboard
                    }
                )

        # Downsample step data to reduce file size
        logger.info("Original steps count: %d", len(steps_data))
        steps_data = downsample_steps(steps_data, max_steps=1000)
        logger.info("Downsampled steps count: %d", len(steps_data))

        data["steps"] = steps_data

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
