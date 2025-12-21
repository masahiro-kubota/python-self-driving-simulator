"""Logger node for recording FrameData."""

import json
from pathlib import Path
from typing import Any

from core.data import Action, ComponentConfig, SimulationLog, SimulationStep, VehicleState
from core.data.node_io import NodeIO
from core.data.ros import MarkerArray, String
from core.interfaces.node import Node, NodeExecutionResult
from logger.mcap_logger import MCAPLogger
from logger.parsers.lanelet2_parser import Lanelet2Parser
from logger.ros_message_builder import (
    build_ackermann_drive_message,
    build_laser_scan_message,
    build_lidar_tf_message,
    build_odometry_message,
    build_tf_message,
)
from logger.visualization.map_visualizer import MapVisualizer
from logger.visualization.obstacle_visualizer import ObstacleVisualizer
from logger.visualization.vehicle_visualizer import VehicleVisualizer


class LoggerConfig(ComponentConfig):
    """Configuration for LoggerNode."""

    output_mcap_path: str | None = None
    map_path: str | None = None
    vehicle_params: Any = None


class LoggerNode(Node[LoggerConfig]):
    """Node responsible for recording FrameData to simulation log."""

    def __init__(self, config: LoggerConfig = LoggerConfig(), rate_hz: float = 10.0):
        """Initialize LoggerNode."""
        super().__init__("Logger", rate_hz, config)
        self.current_time = 0.0
        self.mcap_logger: MCAPLogger | None = None
        self.log = SimulationLog(steps=[], metadata={})
        self.map_published = False

        # Initialize visualizers
        self.vehicle_visualizer = VehicleVisualizer(config.vehicle_params)
        self.obstacle_visualizer = ObstacleVisualizer()
        self.map_visualizer: MapVisualizer | None = None

    def on_init(self) -> None:
        """Initialize resources."""
        if self.config.output_mcap_path:
            mcap_path = Path(self.config.output_mcap_path)

            # Handle directory path or file path
            if mcap_path.is_dir() or (not mcap_path.exists() and not mcap_path.suffix):
                mcap_path.mkdir(parents=True, exist_ok=True)
                # Use fixed filename to avoid accumulation
                mcap_path = mcap_path / "simulation.mcap"

            self.mcap_logger = MCAPLogger(mcap_path)
            self.mcap_logger.__enter__()

            # Initialize map visualizer and publish map once
            if self.config.map_path:
                parser = Lanelet2Parser(self.config.map_path)
                self.map_visualizer = MapVisualizer(parser)
                self._publish_map()
                self.map_published = True

    def on_shutdown(self) -> None:
        """Cleanup resources."""
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

        # Reconstruct SimulationStep for legacy compatibility
        sim_state = getattr(self.frame_data, "sim_state", None)
        if sim_state is None:
            sim_state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0, timestamp=current_time)

        action = getattr(self.frame_data, "action", None)
        if action is None:
            action = Action(steering=0.0, acceleration=0.0)

        simulation_info = {
            "goal_count": getattr(self.frame_data, "goal_count", 0),
        }

        step = SimulationStep(
            timestamp=current_time,
            vehicle_state=sim_state,
            action=action,
            ad_component_log=None,
            info=simulation_info,
        )
        self.log.steps.append(step)

        if self.mcap_logger is None:
            return NodeExecutionResult.SUCCESS

        # Log ROS 2 messages to MCAP
        self._log_vehicle_state(sim_state, current_time)
        self._log_lidar_scan(current_time)
        self._log_control_command(action, current_time)
        self._log_simulation_info(simulation_info, current_time)

        return NodeExecutionResult.SUCCESS

    def _log_vehicle_state(self, vehicle_state: VehicleState, timestamp: float) -> None:
        """Log vehicle state messages."""
        # TF: map -> base_link
        tf_msg = build_tf_message(vehicle_state, timestamp)
        self.mcap_logger.log("/tf", tf_msg, timestamp)

        # Odometry
        odom_msg = build_odometry_message(vehicle_state, timestamp)
        self.mcap_logger.log("/localization/kinematic_state", odom_msg, timestamp)

        # Vehicle marker
        vehicle_marker = self.vehicle_visualizer.create_marker(vehicle_state, timestamp)
        vehicle_marker_array = MarkerArray(markers=[vehicle_marker])
        self.mcap_logger.log("/vehicle/marker", vehicle_marker_array, timestamp)

        # Obstacle markers
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

    def _log_simulation_info(self, simulation_info: dict, timestamp: float) -> None:
        """Log simulation info as JSON."""
        if simulation_info:
            self.mcap_logger.log(
                "/simulation/info", String(data=json.dumps(simulation_info)), timestamp
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

    def get_log(self) -> SimulationLog:
        """Get simulation log."""
        return self.log
