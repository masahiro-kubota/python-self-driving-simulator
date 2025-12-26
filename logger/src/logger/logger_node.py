import json
import logging
from pathlib import Path
from typing import Any

from core.data import (
    Action,
    ComponentConfig,
    SimulationLog,
    VehicleParameters,
    VehicleState,
)
from core.data.node_io import NodeIO
from core.data.ros import MarkerArray, String
from core.interfaces.node import Node, NodeExecutionResult
from pydantic import Field

from logger.mcap_logger import MCAPLogger
from logger.parsers.lanelet2_parser import Lanelet2Parser
from logger.ros_message_builder import (
    build_ackermann_drive_message,
    build_laser_scan_message,
    build_lidar_tf_message,
    build_odometry_message,
    build_tf_message,
)
from logger.track_loader import load_track_csv_simple
from logger.visualization.map_visualizer import MapVisualizer
from logger.visualization.obstacle_visualizer import ObstacleVisualizer
from logger.visualization.path_visualizer import PathVisualizer
from logger.visualization.trajectory_visualizer import TrajectoryVisualizer
from logger.visualization.vehicle_visualizer import VehicleVisualizer

logger = logging.getLogger(__name__)


class LoggerConfig(ComponentConfig):
    """Configuration for LoggerNode."""

    output_mcap_path: str = Field(..., description="Path to output MCAP file")
    map_path: str = Field(..., description="Path to map file")
    track_path: str = Field(..., description="Path to track file")
    vehicle_params: VehicleParameters = Field(..., description="Vehicle parameters")

    # Visualization colors (Hex strings #RRGGBB or #RRGGBBAA)
    vehicle_color: str = Field("#0080FFCC", description="Vehicle color")
    obstacle_color: str = Field("#FF0000B2", description="Obstacle color")
    trajectory_color: str = Field("#00FF00CC", description="Trajectory color")
    lookahead_point_color: str = Field("#FF00FFCC", description="Lookahead point color")
    path_color: str = Field("#00CCFFCC", description="Vehicle path history color")
    map_left_color: str = Field("#FFFFFFCC", description="Map left boundary color")
    map_right_color: str = Field("#CCCCCCB2", description="Map right boundary color")
    global_track_color: str = Field("#FFFF00FF", description="Global track color")


class LoggerNode(Node[LoggerConfig]):
    """Node responsible for recording FrameData to simulation log."""

    def __init__(self, config: LoggerConfig, rate_hz: float = 10.0):
        """Initialize LoggerNode."""
        super().__init__("Logger", rate_hz, config)
        self.current_time = 0.0
        self.mcap_logger: MCAPLogger | None = None
        self.log = SimulationLog(steps=[], metadata={})
        self.map_published = False
        self.track_published = False

        from core.data.ros import ColorRGBA

        # Initialize visualizers with configured colors
        self.vehicle_visualizer = VehicleVisualizer(
            config.vehicle_params, color=ColorRGBA.from_hex(config.vehicle_color)
        )
        self.obstacle_visualizer = ObstacleVisualizer(
            color=ColorRGBA.from_hex(config.obstacle_color)
        )
        self.trajectory_visualizer = TrajectoryVisualizer(
            lookahead_color=ColorRGBA.from_hex(config.lookahead_point_color),
            trajectory_color=ColorRGBA.from_hex(config.trajectory_color),
        )
        self.path_visualizer = PathVisualizer(
            max_history=0, color=ColorRGBA.from_hex(config.path_color)
        )  # Unlimited for batch processing
        self.map_visualizer: MapVisualizer | None = None

        # Data accumulation for path visualization only
        self.vehicle_positions: list[tuple[float, float]] = []
        self.first_timestamp: float | None = None

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
        """Cleanup resources."""
        # Finalize visualizations before closing MCAP
        self._finalize_visualizations()

        if self.mcap_logger:
            self.mcap_logger.__exit__(None, None, None)

    def get_node_io(self) -> NodeIO:
        """Define node IO."""
        return NodeIO(inputs={}, outputs={})

    def on_run(self, current_time: float) -> NodeExecutionResult:
        """Record current FrameData to log."""
        if self.frame_data is None:
            return NodeExecutionResult.SUCCESS

        self.current_time = current_time
        if self.first_timestamp is None:
            self.first_timestamp = current_time

        if self.mcap_logger is None:
            return NodeExecutionResult.SUCCESS

        # Dictionary to store a snapshot for the /simulation/step topic (legacy support/dashboard)
        step_snapshot: dict[str, Any] = {"timestamp": current_time}

        # 1. Generically log all Pydantic models (BaseModel) and MarkerArrays found in FrameData
        from pydantic import BaseModel

        # Prepare AD log structure and simulation info for compatibility
        ad_log_data = {}
        simulation_info = {}

        for key, value in vars(self.frame_data).items():
            if value is None:
                continue

            # Log to dedicated topic: /key
            topic = f"/{key}"

            if isinstance(value, BaseModel):
                self.mcap_logger.log(topic, value, current_time)
                # Add to snapshot for dashboard (exclude large/internal data if necessary)
                if key in ["vehicle_state", "action", "sim_state"]:
                    step_snapshot[key] = value.model_dump()
                elif isinstance(value, MarkerArray) and key.startswith("mppi_"):
                    # Map MPPI markers to the format dashboard expects
                    ad_log_data[key.replace("mppi_", "")] = value.model_dump()
            elif isinstance(value, MarkerArray):
                self.mcap_logger.log(topic, value, current_time)
                if key.startswith("mppi_"):
                    ad_log_data[key.replace("mppi_", "")] = value.model_dump()
            elif key == "obstacles" and isinstance(value, list):
                # Special handling for obstacles list
                step_snapshot[key] = [
                    obs.model_dump() if hasattr(obs, "model_dump") else obs for obs in value
                ]
            elif isinstance(value, int | float | str | bool):
                # Gather primitive fields for /simulation/info
                simulation_info[key] = value

        if ad_log_data:
            step_snapshot["ad_component_log"] = {
                "component_type": "mppi_controller",
                "data": ad_log_data,
            }

        # 2. Log consolidated simulation step and info for compatibility
        try:
            self.mcap_logger.log(
                "/simulation/step", String(data=json.dumps(step_snapshot)), current_time
            )
            if simulation_info:
                self.mcap_logger.log(
                    "/simulation/info", String(data=json.dumps(simulation_info)), current_time
                )
        except Exception as e:
            logger.warning("Failed to log /simulation/step or info: %s", e)

        # 3. Maintain ROS-standard specific messages for Foxglove compatibility
        sim_state = getattr(self.frame_data, "vehicle_state", None) or getattr(
            self.frame_data, "sim_state", None
        )
        if sim_state:
            self._log_vehicle_state(sim_state, current_time)
            # Accumulate vehicle positions for final path visualization
            self.vehicle_positions.append((sim_state.x, sim_state.y))

        lidar_scan = getattr(self.frame_data, "lidar_scan", None)
        if lidar_scan:
            self._log_lidar_scan(current_time)

        action = getattr(self.frame_data, "action", None)
        if action:
            self._log_control_command(action, current_time)

        trajectory = getattr(self.frame_data, "trajectory", None)
        if trajectory:
            self._log_trajectory(trajectory, current_time)

        return NodeExecutionResult.SUCCESS

    def _log_trajectory(self, trajectory: Any, timestamp: float) -> None:
        """Log trajectory (lookahead point) markers."""
        marker = self.trajectory_visualizer.create_marker(trajectory, timestamp)
        if marker:
            self.mcap_logger.log("/planning/marker", MarkerArray(markers=[marker]), timestamp)

    def _log_vehicle_state(self, vehicle_state: VehicleState, timestamp: float) -> None:
        """Log vehicle state messages."""
        # TF: map -> base_link
        tf_msg = build_tf_message(vehicle_state, timestamp)
        self.mcap_logger.log("/tf", tf_msg, timestamp)

        # Odometry
        odom_msg = build_odometry_message(vehicle_state, timestamp)
        self.mcap_logger.log("/localization/kinematic_state", odom_msg, timestamp)

        # Vehicle marker (real-time)
        vehicle_marker = self.vehicle_visualizer.create_marker(vehicle_state, timestamp)
        vehicle_marker_array = MarkerArray(markers=[vehicle_marker])
        self.mcap_logger.log("/vehicle/marker", vehicle_marker_array, timestamp)

        # Obstacle markers (real-time)
        obstacles = getattr(self.frame_data, "obstacles", None)
        if obstacles:
            obstacle_marker_array = self.obstacle_visualizer.create_marker_array(
                obstacles, timestamp
            )
            if obstacle_marker_array.markers:
                self.mcap_logger.log("/obstacles/marker", obstacle_marker_array, timestamp)

    def _log_lidar_scan(self, timestamp: float) -> None:
        """Log LiDAR scan messages."""
        lidar_scan = getattr(self.frame_data, "lidar_scan", None)
        if not lidar_scan:
            return

        # TF: base_link -> lidar_link
        tf_lidar = build_lidar_tf_message(lidar_scan, timestamp)
        self.mcap_logger.log("/tf", tf_lidar, timestamp)

        # LaserScan
        scan_msg = build_laser_scan_message(lidar_scan)
        self.mcap_logger.log("/perception/lidar/scan", scan_msg, lidar_scan.timestamp)

    def _log_control_command(self, action: Action, timestamp: float) -> None:
        """Log control command messages."""
        cmd_msg = build_ackermann_drive_message(action, timestamp)
        self.mcap_logger.log("/control/command/control_cmd", cmd_msg, timestamp)

    def _finalize_visualizations(self) -> None:
        """Finalize path visualization at the end of simulation."""
        if not self.mcap_logger or self.first_timestamp is None:
            return

        # Use the first timestamp for static visualizations (like map)
        first_timestamp = self.first_timestamp

        # Vehicle path (entire trajectory as a single marker)
        if self.vehicle_positions:
            # Set all positions at once
            for x, y in self.vehicle_positions:
                self.path_visualizer.add_position(x, y)

            path_marker = self.path_visualizer.create_marker(first_timestamp)
            if path_marker:
                self.mcap_logger.log(
                    "/vehicle/path", MarkerArray(markers=[path_marker]), first_timestamp
                )

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
            marker = self.trajectory_visualizer.create_marker(track, self.current_time)
            if marker:
                # Override namespace/color for global track
                marker.ns = "global_track"
                marker.id = 999
                from core.data.ros import ColorRGBA

                marker.color = ColorRGBA.from_hex(self.config.global_track_color)
                self.mcap_logger.log("/map/track", MarkerArray(markers=[marker]), self.current_time)
        except Exception as e:
            print(f"Failed to load/publish track: {e}")

    def get_log(self) -> SimulationLog:
        """Get simulation log."""
        return self.log
