import json
import logging
from pathlib import Path

from core.data import (
    ComponentConfig,
    SimulationLog,
    TopicSlot,
    VehicleParameters,
)
from core.data.node_io import NodeIO
from core.data.ros import MarkerArray, String
from core.interfaces.node import Node, NodeExecutionResult
from core.utils.lanelet2_parser import Lanelet2Parser
from core.utils.ros_message_builder import to_ros_time
from core.visualization.map_visualizer import MapVisualizer
from pydantic import Field

from logger.mcap_logger import MCAPLogger
from logger.track_loader import load_track_csv_simple

logger = logging.getLogger(__name__)


class LoggerConfig(ComponentConfig):
    """Configuration for LoggerNode."""

    output_mcap_path: str = Field(..., description="Path to output MCAP file")
    map_path: str = Field(..., description="Path to map file")
    track_path: str = Field(..., description="Path to track file")
    vehicle_params: VehicleParameters = Field(..., description="Vehicle parameters")

    # Visualization colors (Hex strings #RRGGBB or #RRGGBBAA)
    path_color: str = Field(..., description="Vehicle path color")
    map_left_color: str = Field(..., description="Map left boundary color")
    map_right_color: str = Field(..., description="Map right boundary color")
    global_track_color: str = Field(..., description="Global track color")


class LoggerNode(Node[LoggerConfig]):
    """Node responsible for recording FrameData to simulation log."""

    def __init__(self, config: LoggerConfig, rate_hz: float, priority: int):
        """Initialize LoggerNode."""
        super().__init__("Logger", rate_hz, config, priority)
        self.current_time = 0.0
        self.mcap_logger: MCAPLogger | None = None
        self.log = SimulationLog(steps=[], metadata={})
        self.map_published = False
        self.track_published = False
        self.vehicle_positions: list[tuple[float, float]] = []

        self.vehicle_positions: list[tuple[float, float]] = []
        self._last_logged_seq: dict[str, int] = {}

        self.map_visualizer: MapVisualizer | None = None

    def on_init(self) -> None:
        """Initialize resources."""
        mcap_path = Path(self.config.output_mcap_path)
        # Ensure parent directory exists
        mcap_path.parent.mkdir(parents=True, exist_ok=True)

        self.mcap_logger = MCAPLogger(mcap_path)
        self.mcap_logger.__enter__()

        # Initialize map visualizer and publish map once
        parser = Lanelet2Parser(self.config.map_path)
        from core.data.ros import ColorRGBA

        self.map_visualizer = MapVisualizer(
            parser,
            left_color=ColorRGBA.from_hex(self.config.map_left_color),
            right_color=ColorRGBA.from_hex(self.config.map_right_color),
        )
        self._publish_map()
        self.map_published = True

        # Initialize track visualization and publish once
        self._publish_track()
        self.track_published = True

    def on_shutdown(self) -> None:
        """Cleanup resources and generate path visualization."""
        if self.mcap_logger:
            # Generate and log final path marker before closing
            self._log_final_path()
            self.mcap_logger.__exit__(None, None, None)

    def get_node_io(self) -> NodeIO:
        """Define node IO."""
        return NodeIO(inputs={}, outputs={})

    def on_run(self, current_time: float) -> NodeExecutionResult:
        """Record current FrameData to log."""
        if self.frame_data is None:
            return NodeExecutionResult.SUCCESS

        self.current_time = current_time

        if self.mcap_logger is None:
            return NodeExecutionResult.SUCCESS

        # Generically log all Pydantic models (BaseModel) and MarkerArrays found in FrameData
        from pydantic import BaseModel

        simulation_info = {}

        for key, slot in vars(self.frame_data).items():
            if not isinstance(slot, TopicSlot):
                continue

            # Check sequence number for updates
            current_seq = slot.seq
            last_seq = self._last_logged_seq.get(key, -1)

            if current_seq == last_seq:
                continue

            # Update last logged sequence
            self._last_logged_seq[key] = current_seq

            value = slot.data
            if value is None:
                continue

            # Log to dedicated topic: /key (with mapping for special cases)
            topic_map = {
                "sim_state": "/localization/kinematic_state",
                "localization_kinematic_state": "/localization/kinematic_state",
                "control_cmd": "/control/command/control_cmd",
                "perception_lidar_scan": "/sensing/lidar/scan",
                "trajectory": "/planning/trajectory",
                "lookahead_marker": "/planning/lookahead_marker",
                "obstacle_markers": "/perception/obstacle_markers",
                "tf_kinematic": "/tf",
                "tf_lidar": "/tf",
                "vehicle_marker": "/vehicle/marker",
                "planning_marker": "/planning/marker",
                "mppi_candidates": "/planning/mppi/candidates",
                "mppi_optimal": "/planning/mppi/optimal",
            }
            topic = topic_map.get(key, f"/{key}")

            if isinstance(value, (BaseModel, MarkerArray)):
                self.mcap_logger.log(topic, value, current_time)
            elif key == "obstacles" and isinstance(value, list):
                # Log obstacles list as JSON string
                try:
                    obs_data = [
                        obs.model_dump() if hasattr(obs, "model_dump") else obs for obs in value
                    ]
                    self.mcap_logger.log(topic, String(data=json.dumps(obs_data)), current_time)
                except Exception:
                    pass
            elif isinstance(value, int | float | str | bool):
                # Gather primitive fields for /simulation/info
                simulation_info[key] = value

        # Log simulation info for metadata
        try:
            if simulation_info:
                self.mcap_logger.log(
                    "/simulation/info", String(data=json.dumps(simulation_info)), current_time
                )
        except Exception as e:
            logger.warning("Failed to log /simulation/info: %s", e)

        # Track vehicle position for final path generation
        from core.utils.mcap_utils import extract_dashboard_state

        for key, slot in vars(self.frame_data).items():
            if not isinstance(slot, TopicSlot):
                continue

            value = slot.data
            if value is None:
                continue

            state = extract_dashboard_state(value)
            if "x" in state and "y" in state:
                self.vehicle_positions.append((state["x"], state["y"]))
                break

        return NodeExecutionResult.SUCCESS

    def _publish_map(self) -> None:
        """Publish map markers."""
        if not self.map_visualizer:
            return

        try:
            marker_array = self.map_visualizer.create_marker_array(self.current_time)
            if marker_array.markers:
                self.mcap_logger.log("/map/vector", marker_array, self.current_time)
        except Exception as e:
            print(f"Failed to load/publish map: {e}")

    def _publish_track(self) -> None:
        """Publish global track centerline markers."""
        if not self.config.track_path:
            return

        try:
            track = load_track_csv_simple(Path(self.config.track_path))

            from core.data.ros import ColorRGBA, Header, Marker, MarkerArray, Point, Vector3

            points = [Point(x=p.x, y=p.y, z=0.0) for p in track.points]
            marker = Marker(
                header=Header(stamp=to_ros_time(self.current_time), frame_id="map"),
                ns="global_track",
                id=999,
                type=4,  # LINE_STRIP
                action=0,
                scale=Vector3(x=0.2, y=0.0, z=0.0),
                color=ColorRGBA.from_hex(self.config.global_track_color),
                points=points,
                frame_locked=True,
            )
            self.mcap_logger.log("/map/track", MarkerArray(markers=[marker]), self.current_time)
        except Exception as e:
            print(f"Failed to load/publish track: {e}")

    def _log_final_path(self) -> None:
        """Generate and log vehicle path visualization."""
        if not self.vehicle_positions:
            return

        try:
            from core.data.ros import ColorRGBA
            from core.visualization.path_visualizer import PathVisualizer

            path_visualizer = PathVisualizer(
                max_history=0, color=ColorRGBA.from_hex(self.config.path_color)
            )
            path_visualizer.set_positions(self.vehicle_positions)

            path_marker = path_visualizer.create_marker(0.0)
            if path_marker:
                from core.data.ros import MarkerArray

                self.mcap_logger.log("/vehicle_path", MarkerArray(markers=[path_marker]), 0.0)
                logger.info(
                    f"Logged final path visualization with {len(self.vehicle_positions)} points"
                )
            else:
                logger.warning("Failed to create path marker")

        except Exception as e:
            logger.error(f"Failed to log final path: {e}")

    def get_log(self) -> SimulationLog:
        """Get simulation log."""
        return self.log
