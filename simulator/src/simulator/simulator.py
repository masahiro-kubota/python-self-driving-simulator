"""Simulator implementation."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from core.data import (
    Action,
    SimulationLog,
    SimulationResult,
    SimulationStep,
    VehicleParameters,
    VehicleState,
)
from core.interfaces import Simulator as SimulatorInterface
from simulator.state import SimulationVehicleState

if TYPE_CHECKING:
    from shapely.geometry import Polygon

    from core.data import ADComponentLog
    from core.interfaces import ADComponent


class Simulator(SimulatorInterface):
    """Generic Simulator class using composition for dynamics."""

    def __init__(
        self,
        step_update_func: Callable[[SimulationVehicleState, Action, float], SimulationVehicleState],
        get_vehicle_polygon_func: Callable[[VehicleState], "Polygon"],
        vehicle_params: "VehicleParameters | None" = None,
        initial_state: VehicleState | None = None,
        dt: float = 0.1,
        map_path: str | None = None,
        goal_x: float | None = None,
        goal_y: float | None = None,
    ) -> None:
        """Initialize Simulator.

        Args:
            step_update_func: Function to update state (state, action, dt) -> next_state
            get_vehicle_polygon_func: Function to get vehicle polygon (state) -> Polygon
            vehicle_params: Vehicle parameters (None will use default)
            initial_state: Initial vehicle state
            dt: Time step [s]
            map_path: Path to Lanelet2 map file
            goal_x: Goal X coordinate [m]
            goal_y: Goal Y coordinate [m]
        """
        self.step_update_func = step_update_func
        self.get_vehicle_polygon_func = get_vehicle_polygon_func

        # For backward compatibility / default values
        if vehicle_params is None:
            vehicle_params = VehicleParameters()
        elif isinstance(vehicle_params, dict):
            vehicle_params = VehicleParameters(**vehicle_params)

        self.vehicle_params = vehicle_params
        self.dt = dt
        self.goal_x = goal_x
        self.goal_y = goal_y

        if initial_state is None:
            self.initial_state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0, timestamp=0.0)
        elif isinstance(initial_state, dict):
            self.initial_state = VehicleState(**initial_state)
        else:
            self.initial_state = initial_state

        # Manage internal state with SimulationVehicleState
        self._current_state = SimulationVehicleState.from_vehicle_state(self.initial_state)
        self.current_time = 0.0
        self.log = SimulationLog(steps=[], metadata={})

        # Load map
        self.map: Any = None
        if map_path:
            from pathlib import Path

            from simulator.map import LaneletMap

            self.map = LaneletMap(Path(map_path))

    def reset(self) -> VehicleState:
        """Reset simulation.

        Returns:
            Initial vehicle state
        """
        self._current_state = SimulationVehicleState.from_vehicle_state(self.initial_state)
        self.current_time = 0.0
        self.log = SimulationLog(steps=[], metadata={})
        return self.initial_state

    def step(self, action: Action) -> tuple[VehicleState, bool, dict[str, Any]]:
        """Advance simulation by one step.

        Args:
            action: Action to execute

        Returns:
            tuple containing:
                - next_state: Updated vehicle state
                - done: Episode termination flag
                - info: Additional information
        """
        # 1. Update state using injected function
        self._current_state = self.step_update_func(self._current_state, action, self.dt)
        self.current_time += self.dt
        self._current_state.timestamp = self.current_time

        # 2. Convert to VehicleState for external interface
        vehicle_state = self._current_state.to_vehicle_state(action)

        # 3. Map validation (if map is loaded)
        if self.map is not None:
            try:
                poly = self.get_vehicle_polygon_func(vehicle_state)
                if not self.map.is_drivable_polygon(poly):
                    vehicle_state.off_track = True
            except Exception:  # Catching broadly to prevent crash during validation
                # Fallback to point check if polygon check fails (e.g. geometry error)
                if not self.map.is_drivable(vehicle_state.x, vehicle_state.y):
                    vehicle_state.off_track = True

        # 4. Logging
        step_log = SimulationStep(
            timestamp=self.current_time,
            vehicle_state=vehicle_state,
            action=action,
            ad_component_log=self._create_ad_component_log(),
            info=self._create_info(),
        )
        self.log.steps.append(step_log)

        # 5. Check done
        done = self._is_done()
        info = self._create_info()

        return vehicle_state, done, info

    def run(
        self,
        ad_component: "ADComponent",
        max_steps: int = 1000,
        goal_threshold: float = 5.0,
        min_elapsed_time: float = 20.0,
    ) -> SimulationResult:
        """Run simulation.

        Args:
            ad_component: AD component instance (planner + controller)
            max_steps: Maximum steps
            goal_threshold: Goal threshold [m]
            min_elapsed_time: Minimum elapsed time [s] before goal check

        Returns:
            SimulationResult
        """
        # Reset simulator
        current_state = self.reset()

        # Run simulation loop
        for step in range(max_steps):
            # Plan
            target_trajectory = ad_component.planner.plan(None, current_state)

            # Control
            action = ad_component.controller.control(target_trajectory, current_state)

            # Simulate
            next_state, done, _ = self.step(action)

            # Check goal
            if self.goal_x is not None and self.goal_y is not None:
                goal_reached = self._check_goal(
                    next_state,
                    step,
                    goal_threshold,
                    min_elapsed_time,
                )
                if goal_reached:
                    return SimulationResult(
                        success=True,
                        reason="goal_reached",
                        final_state=next_state,
                        log=self.get_log(),
                    )

            # Update state
            current_state = next_state

            # Check done flag
            if done:
                return SimulationResult(
                    success=False,
                    reason="done_flag",
                    final_state=current_state,
                    log=self.get_log(),
                )

        # Max steps reached
        return SimulationResult(
            success=False,
            reason="max_steps",
            final_state=current_state,
            log=self.get_log(),
        )

    def get_log(self) -> SimulationLog:
        """Get simulation log.

        Returns:
            SimulationLog
        """
        return self.log

    def close(self) -> None:
        """Close simulator."""

    def _create_ad_component_log(self) -> "ADComponentLog":
        """Create AD component log."""
        from core.data import ADComponentLog

        return ADComponentLog(component_type="simulator", data={})

    def _create_info(self) -> dict[str, Any]:
        """Create additional info."""
        return {}

    def _is_done(self) -> bool:
        """Check episode termination."""
        return False

    def _check_goal(
        self,
        state: VehicleState,
        step: int,
        goal_threshold: float,
        min_elapsed_time: float,
    ) -> bool:
        """Check goal reached."""
        # Check distance to goal
        assert self.goal_x is not None and self.goal_y is not None
        dist_to_end = ((state.x - self.goal_x) ** 2 + (state.y - self.goal_y) ** 2) ** 0.5

        # Use time threshold to avoid early goal detection
        elapsed_time = step * self.dt
        return dist_to_end < goal_threshold and elapsed_time > min_elapsed_time


class KinematicSimulator(Simulator):
    """Kinematic Simulator preset."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize KinematicSimulator."""
        super().__init__(
            step_update_func=self._kinematic_step,
            get_vehicle_polygon_func=self._kinematic_polygon,
            **kwargs,
        )

    def _kinematic_step(
        self, state: SimulationVehicleState, action: Action, dt: float
    ) -> SimulationVehicleState:
        from simulator.dynamics import update_bicycle_model

        return update_bicycle_model(
            state,
            action.steering,
            action.acceleration,
            dt,
            self.vehicle_params.wheelbase,
        )

    def _kinematic_polygon(self, state: VehicleState) -> "Polygon":
        from simulator.dynamics import get_bicycle_model_polygon

        return get_bicycle_model_polygon(state, self.vehicle_params)
