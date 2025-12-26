"""Simulator implementation."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.data import (
    Action,
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
        from core.data import LidarScan

        return NodeIO(
            inputs={
                "action": Action,
            },
            outputs={
                "sim_state": VehicleState,
                "obstacles": list[SimulatorObstacle],
                "lidar_scan": LidarScan,  # Declare output
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

        # Get action from frame_data
        action = getattr(self.frame_data, "action", None)
        if action is None:
            action = Action(steering=0.0, acceleration=0.0)

        # Update state using bicycle model
        from simulator.dynamics import update_bicycle_model

        self._current_state = update_bicycle_model(
            self._current_state,
            action.steering,
            action.acceleration,
            self.dt,
            self.config.vehicle_params.wheelbase,
        )
        self.current_time += self.dt
        self._current_state.timestamp = self.current_time

        # Convert to VehicleState
        vehicle_state = self._current_state.to_vehicle_state(action)

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

        # Logging
        lidar_scan = None
        if self.lidar_sensor:
            lidar_scan = self.lidar_sensor.scan(vehicle_state)

        step_log = SimulationStep(
            timestamp=self.current_time,
            vehicle_state=vehicle_state,
            action=action,
            ad_component_log=self._create_ad_component_log(),
            info={"lidar_scan": lidar_scan} if lidar_scan else {},
        )
        self.log.steps.append(step_log)

        # Update frame_data with new state
        self.frame_data.sim_state = vehicle_state
        if self.obstacle_manager:
            self.frame_data.obstacles = self.obstacle_manager.obstacles
        else:
            self.frame_data.obstacles = []
        self.frame_data.obstacle_states = obstacle_states

        # Expose Lidar data to frame_data (NodeIO) if needed
        # Currently NodeIO doesn't explicitly define 'scan' output, but we can add it to info or frame_data dynamic
        if lidar_scan:
            self.frame_data.lidar_scan = lidar_scan

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
