"""Simulator implementation."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.data import (
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
    from shapely.geometry import Polygon


class SimulatorConfig(ComponentConfig):
    """Configuration for Simulator."""

    vehicle_params: VehicleParameters = Field(..., description="Vehicle parameters")
    initial_state: VehicleState = Field(..., description="Initial vehicle state")
    map_path: Path = Field(..., description="Path to Lanelet2 map file")
    obstacles: list = Field(default_factory=list, description="List of obstacles")
    obstacle_color: str = Field(..., description="Obstacle marker color")
    topic_rates: dict[str, float] = Field(
        default_factory=dict,
        description="Publish rates for specific topics [Hz]",
    )


class SimulatorNode(Node[SimulatorConfig]):
    """Simulator node using bicycle kinematic model."""

    def __init__(
        self,
        config: SimulatorConfig,
        rate_hz: float,
        priority: int,
    ) -> None:
        """Initialize Simulator.

        Args:
            config: Validated configuration
            rate_hz: Physics update rate [Hz]
            priority: Execution priority
        """
        super().__init__("SimulatorNode", rate_hz, config, priority)

        self.dt = 1.0 / rate_hz
        self._current_state: SimulationVehicleState | None = None
        self.current_time = 0.0
        self.log = SimulationLog(steps=[], metadata={})
        self.map: Any = None
        self.obstacle_manager: Any = None
        self.lidar_sensor: Any = None

        # Topic publish control
        self._topic_intervals: dict[str, int] = {}
        self._topic_counters: dict[str, int] = {}

        # Steering response model
        from collections import deque

        self._steer_delay_buffer: deque[float] = deque()

    def get_node_io(self) -> NodeIO:
        """Define node IO."""
        # Use lazy import for LidarScan because it might be circular if imported at top-level
        # (Though currently it's safe as it's in core.data)
        from core.data.autoware import SteeringReport
        from core.data.ros import LaserScan, MarkerArray

        return NodeIO(
            inputs={
                "control_cmd": Any,  # AckermannControlCommand
            },
            outputs={
                "sim_state": VehicleState,
                "obstacles": list[SimulatorObstacle],
                "obstacle_states": list,
                "obstacle_markers": MarkerArray,
                "perception_lidar_scan": LaserScan,
                "steering_status": SteeringReport,
            },
        )

    def on_init(self) -> None:
        """Initialize simulation state and load map."""
        # Reset state
        self._current_state = SimulationVehicleState.from_vehicle_state(self.config.initial_state)
        self.current_time = 0.0

        # Reset steering delay buffer and pre-fill with initial steering
        self._steer_delay_buffer.clear()
        initial_steering = self._current_state.actual_steering
        # Calculate needed steps using same logic as dynamics.py
        delay_steps = max(1, int(self.config.vehicle_params.steer_delay_time / self.dt))
        # Pre-fill with initial value to avoid "jump" at t=0
        for _ in range(delay_steps):
            self._steer_delay_buffer.append(initial_steering)

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

        # Initialize topic intervals
        self._topic_intervals = {}
        self._topic_counters = {}
        for topic, rate in self.config.topic_rates.items():
            self._topic_intervals[topic] = max(1, round(self.rate_hz / rate))
            self._topic_counters[topic] = 0

        # LiDAR special case
        if self.lidar_sensor and self.config.vehicle_params.lidar:
            lidar_rate = self.config.vehicle_params.lidar.publish_rate_hz
            self._topic_intervals["perception_lidar_scan"] = max(
                1, round(self.rate_hz / lidar_rate)
            )
            self._topic_counters["perception_lidar_scan"] = 0

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
        if self.subscribe("termination_signal"):
            return NodeExecutionResult.SUCCESS

        # Expose Lidar data to frame_data (NodeIO) if needed
        # Currently NodeIO doesn't explicitly define 'scan' output, but we can add it to info or frame_data dynamic
        from core.data.autoware import AckermannControlCommand

        # Get control command from frame_data
        control_cmd = self.subscribe("control_cmd")
        if control_cmd is None:
            # Default to zero control
            steering = 0.0
            acceleration = 0.0
        else:
            if hasattr(control_cmd, "lateral") and hasattr(control_cmd, "longitudinal"):
                # AckermannControlCommand
                steering = control_cmd.lateral.steering_tire_angle
                acceleration = control_cmd.longitudinal.acceleration
            else:
                self.get_log().warning(f"Unknown control command type: {type(control_cmd)}")
                steering = 0.0
                acceleration = 0.0

        # Apply steering response model
        from simulator.dynamics import apply_steering_response_model

        actual_steering, steer_rate_internal, self._steer_delay_buffer = (
            apply_steering_response_model(
                self._current_state,
                steering,
                self.dt,
                self.config.vehicle_params,
                self._steer_delay_buffer,
            )
        )

        # Update internal state for next iteration
        self._current_state.actual_steering = actual_steering
        self._current_state.steer_rate_internal = steer_rate_internal

        # Update state using bicycle model with actual steering
        from simulator.dynamics import update_bicycle_model

        self._current_state = update_bicycle_model(
            self._current_state,
            actual_steering,
            acceleration,
            self.dt,
            self.config.vehicle_params,
        )
        self.current_time += self.dt
        self._current_state.timestamp = self.current_time

        # Convert to VehicleState
        vehicle_state = self._current_state.to_vehicle_state(
            steering=actual_steering, acceleration=acceleration
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
        from core.data.autoware import AckermannLateralCommand, LongitudinalCommand
        from core.data.ros import MarkerArray
        from core.utils.ros_message_builder import to_ros_time

        stamp = to_ros_time(self.current_time)
        drive_action = AckermannControlCommand(
            stamp=stamp,
            lateral=AckermannLateralCommand(stamp=stamp, steering_tire_angle=steering),
            longitudinal=LongitudinalCommand(stamp=stamp, acceleration=acceleration, speed=0.0),
        )
        step_log = SimulationStep(
            timestamp=self.current_time,
            vehicle_state=vehicle_state,
            action=drive_action,
            info={"lidar_ranges": ranges.tolist()} if ranges is not None else {},
        )
        self.log.steps.append(step_log)

        # Update frame_data with new state
        if self._should_publish("sim_state"):
            self.publish("sim_state", vehicle_state)

        if self.obstacle_manager:
            if self._should_publish("obstacles"):
                self.publish("obstacles", self.obstacle_manager.obstacles)

            # Generate obstacle markers
            if self._should_publish("obstacle_markers"):
                obstacle_marker_array = self.obstacle_visualizer.create_marker_array(
                    self.obstacle_manager.obstacles, self.current_time
                )
                self.publish("obstacle_markers", obstacle_marker_array)
        else:
            if self._should_publish("obstacles"):
                self.publish("obstacles", [])
            if self._should_publish("obstacle_markers"):
                self.publish("obstacle_markers", MarkerArray(markers=[]))

        if self._should_publish("obstacle_states"):
            self.publish("obstacle_states", obstacle_states)

        if ranges is not None:
            from core.utils.ros_message_builder import build_laser_scan_message

            # LaserScan message
            if self._should_publish("perception_lidar_scan"):
                scan_msg = build_laser_scan_message(
                    self.config.vehicle_params.lidar, ranges, self.current_time
                )
                self.publish("perception_lidar_scan", scan_msg)

        # Publish steering status (actual steering angle after response model)
        if self._should_publish("steering_status"):
            from core.utils.ros_message_builder import build_steering_report

            steering_report = build_steering_report(actual_steering, self.current_time)
            self.publish("steering_status", steering_report)

        return NodeExecutionResult.SUCCESS

    def _should_publish(self, topic: str) -> bool:
        """Check if topic should be published based on rate."""
        interval = self._topic_intervals.get(topic, 1)
        counter = self._topic_counters.get(topic, 0)
        should = counter % interval == 0
        self._topic_counters[topic] = counter + 1
        return should

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
