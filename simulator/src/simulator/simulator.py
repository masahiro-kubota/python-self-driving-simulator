"""Simulator implementation."""

from typing import TYPE_CHECKING, Any

from pydantic import Field

from core.data import (
    Action,
    SimulationLog,
    SimulationStep,
    VehicleParameters,
    VehicleState,
)
from core.data.node_io import NodeIO
from core.interfaces.node import Node, NodeConfig, NodeExecutionResult
from simulator.state import SimulationVehicleState

if TYPE_CHECKING:
    from shapely.geometry import Polygon

    from core.data import ADComponentLog


class SimulatorConfig(NodeConfig):
    """Configuration for Simulator."""

    vehicle_params: VehicleParameters = Field(
        default_factory=VehicleParameters, description="Vehicle parameters"
    )
    initial_state: VehicleState = Field(
        default_factory=lambda: VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0, timestamp=0.0),
        description="Initial vehicle state",
    )
    map_path: str | None = Field(None, description="Path to Lanelet2 map file")
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

    def get_node_io(self) -> NodeIO:
        """Define node IO."""
        return NodeIO(
            inputs={
                "action": Action,
            },
            outputs={
                "sim_state": VehicleState,
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
        if self.config.vehicle_params:
            if hasattr(self.config.vehicle_params, "model_dump"):
                vp_dict = self.config.vehicle_params.model_dump()
            elif isinstance(self.config.vehicle_params, dict):
                vp_dict = self.config.vehicle_params
            else:
                # Fallback: convert to dict using vars()
                vp_dict = vars(self.config.vehicle_params)
            metadata.update(vp_dict)

        # Add obstacles to metadata
        if self.config.obstacles:
            # Convert obstacles to dict for metadata
            obstacles_data = []
            for obs in self.config.obstacles:
                obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs
                obstacles_data.append(obs_dict)
            metadata["obstacles"] = obstacles_data
            print(f"DEBUG Simulator.on_init: Added {len(obstacles_data)} obstacles to metadata")
        else:
            print("DEBUG Simulator.on_init: No obstacles in config")

        self.log = SimulationLog(steps=[], metadata=metadata)
        print(
            f"DEBUG Simulator.on_init: log.metadata has obstacles: {'obstacles' in self.log.metadata}"
        )

        # Load map if specified
        if self.config.map_path:
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
            self.obstacle_manager = ObstacleManager(obstacles)

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

        # Obstacle collision detection
        if self.obstacle_manager is not None:
            try:
                poly = self._get_vehicle_polygon(vehicle_state)
                if self.obstacle_manager.check_vehicle_collision(poly, self.current_time):
                    vehicle_state.collision = True
            except Exception:
                # If polygon check fails, skip collision detection
                pass

        # Logging
        step_log = SimulationStep(
            timestamp=self.current_time,
            vehicle_state=vehicle_state,
            action=action,
            ad_component_log=self._create_ad_component_log(),
            info={},
        )
        self.log.steps.append(step_log)

        # Update frame_data with new state
        self.frame_data.sim_state = vehicle_state

        return NodeExecutionResult.SUCCESS

    def get_log(self) -> SimulationLog:
        """Get simulation log.

        Returns:
            SimulationLog
        """
        print(
            f"DEBUG Simulator.get_log: log.metadata has obstacles: {'obstacles' in self.log.metadata}"
        )
        print(f"DEBUG Simulator.get_log: log.metadata keys: {list(self.log.metadata.keys())}")
        return self.log

    def _get_vehicle_polygon(self, state: VehicleState) -> "Polygon":
        """Get vehicle polygon for collision detection."""
        from simulator.dynamics import get_bicycle_model_polygon

        return get_bicycle_model_polygon(state, self.config.vehicle_params)

    def _create_ad_component_log(self) -> "ADComponentLog":
        """Create AD component log."""
        from core.data import ADComponentLog

        return ADComponentLog(component_type="simulator", data={})
