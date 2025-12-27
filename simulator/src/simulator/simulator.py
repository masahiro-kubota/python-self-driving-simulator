"""Simulator implementation."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.data import (
    ADComponentLog,
    ComponentConfig,  # Added ComponentConfig
    SimulationLog,
    SimulationStep,
    SimulatorObstacle,
    VehicleParameters,
    VehicleState,
)
from core.data.node_io import NodeIO
from core.interfaces.node import Node, NodeExecutionResult  # Removed NodeConfig
from pydantic import Field

from simulator.state import SimulationVehicleState

if TYPE_CHECKING:
    from core.data import ADComponentLog
    from shapely.geometry import Polygon


class SimulatorConfig(ComponentConfig):
    """Configuration for Simulator."""

    vehicle_params: VehicleParameters = Field(..., description="Vehicle parameters")
    initial_state: VehicleState = Field(..., description="Initial vehicle state")
    map_path: Path = Field(..., description="Path to Lanelet2 map file")
    obstacles: list = Field(default_factory=list, description="List of obstacles")
    obstacle_color: str = Field(..., description="Obstacle marker color")


class Simulator(Node[SimulatorConfig]):
    """Simulator node using bicycle kinematic model."""

    def __init__(
        self,
        config: SimulatorConfig,
        rate_hz: float,
    ) -> None:
        """Initialize Simulator.

        Args:
            config: Validated configuration
            rate_hz: Physics update rate [Hz]
        """
        super().__init__("Simulator", rate_hz, config)

        self.dt = 1.0 / rate_hz
        self._current_state: SimulationVehicleState | None = None
        self.current_time = 0.0
        self.log = SimulationLog(steps=[], metadata={})
        self.map: Any = None
        self.obstacle_manager: Any = None
        self.lidar_sensor: Any = None

    def get_node_io(self) -> NodeIO:
        """Define node IO."""
        # Use lazy import for LidarScan because it might be circular if imported at top-level
        # (Though currently it's safe as it's in core.data)
        from core.data.ros import AckermannDriveStamped, LaserScan, MarkerArray, TFMessage

        return NodeIO(
            inputs={
                "control_cmd": AckermannDriveStamped,
            },
            outputs={
                "sim_state": VehicleState,
                "obstacles": list[SimulatorObstacle],
                "obstacle_markers": MarkerArray,
                "perception_lidar_scan": LaserScan,
                "tf_lidar": TFMessage,
            },
        )

    def on_init(self) -> None:
        """Initialize simulation state and load map."""
        # Reset state
        self._current_state = SimulationVehicleState.from_vehicle_state(self.config.initial_state)
        self.current_time = 0.0

        # Initialize metadata with vehicle params and obstacles
        metadata = {}

        # Add vehicle parameters to metadata
        # Pydantic ensures vehicle_params is not None and is a VehicleParameters object
        vp_dict = self.config.vehicle_params.model_dump()
        metadata.update(vp_dict)

        # Add obstacles to metadata
        if self.config.obstacles:
            # Convert obstacles to dict for metadata
            obstacles_data = []
            for obs in self.config.obstacles:
                obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs
                obstacles_data.append(obs_dict)
            metadata["obstacles"] = obstacles_data

        self.log = SimulationLog(steps=[], metadata=metadata)

        # Load map
        from pathlib import Path

        from simulator.map import LaneletMap

        self.map = LaneletMap(Path(self.config.map_path))

        # Initialize obstacle manager
        if self.config.obstacles:
            from core.data import SimulatorObstacle

            from simulator.obstacle import ObstacleManager

            # Convert dict obstacles to SimulatorObstacle instances
            obstacles = [
                SimulatorObstacle(**obs) if isinstance(obs, dict) else obs
                for obs in self.config.obstacles
            ]
            obstacles = [self._prepare_obstacle(obs) for obs in obstacles]
            self.obstacle_manager = ObstacleManager(obstacles)

        # Initialize Lidar config
        if self.config.vehicle_params.lidar:
            lidar_config = self.config.vehicle_params.lidar
            from simulator.sensor import LidarSensor

            self.lidar_sensor = LidarSensor(
                config=lidar_config, map_instance=self.map, obstacle_manager=self.obstacle_manager
            )

        # Initialize ObstacleVisualizer
        from core.data.ros import ColorRGBA
        from core.visualization.obstacle_visualizer import ObstacleVisualizer

        self.obstacle_visualizer = ObstacleVisualizer(
            color=ColorRGBA.from_hex(self.config.obstacle_color)
        )

    def on_run(self, _current_time: float) -> NodeExecutionResult:
        """Execute physics simulation step.

        Args:
            _current_time: Current simulation time

        Returns:
            NodeExecutionResult indicating execution status
        """
        if self.frame_data is None or self._current_state is None:
            return NodeExecutionResult.FAILED

        # Skip if simulation is terminated
        if hasattr(self.frame_data, "termination_signal") and self.frame_data.termination_signal:
            return NodeExecutionResult.SUCCESS

        # Expose Lidar data to frame_data (NodeIO) if needed
        # Currently NodeIO doesn't explicitly define 'scan' output, but we can add it to info or frame_data dynamic
        from core.data.ros import AckermannDrive, MarkerArray

        # Get control command from frame_data
        control_cmd = getattr(self.frame_data, "control_cmd", None)
        if control_cmd is None:
            # Default to zero control
            steering = 0.0
            acceleration = 0.0
        else:
            steering = control_cmd.drive.steering_angle
            acceleration = control_cmd.drive.acceleration

        # Update state using bicycle model
        from simulator.dynamics import update_bicycle_model

        self._current_state = update_bicycle_model(
            self._current_state,
            steering,
            acceleration,
            self.dt,
            self.config.vehicle_params.wheelbase,
        )
        self.current_time += self.dt
        self._current_state.timestamp = self.current_time

        # Convert to VehicleState
        vehicle_state = self._current_state.to_vehicle_state(
            steering=steering, acceleration=acceleration
        )

        # Map validation (if map is loaded)
        if self.map is not None:
            try:
                poly = self._get_vehicle_polygon(vehicle_state)
                if not self.map.is_drivable_polygon(poly):
                    vehicle_state.off_track = True
            except Exception:
                # Fallback to point check if polygon check fails
                if not self.map.is_drivable(vehicle_state.x, vehicle_state.y):
                    vehicle_state.off_track = True

        obstacle_states = []
        if self.obstacle_manager is not None:
            from simulator.obstacle import check_collision, get_obstacle_polygon, get_obstacle_state

            # Precompute obstacle states for logging and collision
            for obstacle in self.obstacle_manager.obstacles:
                try:
                    obstacle_states.append(get_obstacle_state(obstacle, self.current_time))
                except Exception:
                    # Skip malformed obstacle state
                    continue

            # Obstacle collision detection
            try:
                poly = self._get_vehicle_polygon(vehicle_state)
                for obstacle, obs_state in zip(self.obstacle_manager.obstacles, obstacle_states):
                    obstacle_polygon = get_obstacle_polygon(obstacle, obs_state)
                    if check_collision(poly, obstacle_polygon):
                        vehicle_state.collision = True
                        break
            except Exception:
                # If polygon check fails, skip collision detection
                pass

        # Scan Lidar if available
        ranges = None
        if self.lidar_sensor:
            ranges = self.lidar_sensor.scan(vehicle_state)

        # Logging
        drive_action = AckermannDrive(steering_angle=steering, acceleration=acceleration)
        step_log = SimulationStep(
            timestamp=self.current_time,
            vehicle_state=vehicle_state,
            action=drive_action,
            ad_component_log=self._create_ad_component_log(),
            info={"lidar_ranges": ranges.tolist()} if ranges is not None else {},
        )
        self.log.steps.append(step_log)

        # Update frame_data with new state
        self.frame_data.sim_state = vehicle_state
        if self.obstacle_manager:
            self.frame_data.obstacles = self.obstacle_manager.obstacles

            # Generate obstacle markers
            obstacle_marker_array = self.obstacle_visualizer.create_marker_array(
                self.obstacle_manager.obstacles, self.current_time
            )
            self.frame_data.obstacle_markers = obstacle_marker_array
        else:
            self.frame_data.obstacles = []
            self.frame_data.obstacle_markers = MarkerArray(markers=[])
        self.frame_data.obstacle_states = obstacle_states

        if ranges is not None:
            from core.utils.ros_message_builder import (
                build_laser_scan_message,
                build_lidar_tf_message,
            )

            # LaserScan message
            # Create a dict that looks like LidarScan for the builder if needed,
            # but let's check build_laser_scan_message first
            scan_msg = build_laser_scan_message(
                self.config.vehicle_params.lidar, ranges, self.current_time
            )
            self.frame_data.perception_lidar_scan = scan_msg

            # TF (base_link -> lidar_link)
            tf_msg = build_lidar_tf_message(self.config.vehicle_params.lidar, self.current_time)
            self.frame_data.tf_lidar = tf_msg

        return NodeExecutionResult.SUCCESS

    def get_log(self) -> SimulationLog:
        """Get simulation log.

        Returns:
            SimulationLog
        """
        return self.log

    def _get_vehicle_polygon(self, state: VehicleState) -> "Polygon":
        """Get vehicle polygon for collision detection."""
        from simulator.dynamics import get_bicycle_model_polygon

        return get_bicycle_model_polygon(state, self.config.vehicle_params)

    def _create_ad_component_log(self) -> "ADComponentLog":
        """Create AD component log."""
        from core.data import ADComponentLog

        return ADComponentLog(component_type="simulator", data={})

    def _prepare_obstacle(self, obstacle: SimulatorObstacle) -> SimulatorObstacle:
        """Convert CSV-based trajectories into waypoint trajectories."""
        if obstacle.type != "dynamic" or obstacle.trajectory is None:
            return obstacle

        trajectory = obstacle.trajectory
        if getattr(trajectory, "type", None) != "csv_path":
            return obstacle

        from simulator.obstacle import load_csv_trajectory

        new_trajectory = load_csv_trajectory(trajectory)
        return obstacle.model_copy(update={"trajectory": new_trajectory})
