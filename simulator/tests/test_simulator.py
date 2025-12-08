"""Tests for Simulator."""

from dataclasses import asdict
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from core.data import Action, SimulationResult, VehicleParameters, VehicleState
from simulator.simulator import Simulator
from simulator.state import SimulationVehicleState

if TYPE_CHECKING:
    from shapely.geometry import Polygon


def mock_update_state(
    state: SimulationVehicleState, action: Action, dt: float
) -> SimulationVehicleState:
    """Mock update function."""
    d = asdict(state)
    d["x"] += 1.0
    d["timestamp"] = (d["timestamp"] or 0.0) + dt
    d["ax"] = action.acceleration
    d["steering"] = action.steering
    return SimulationVehicleState(**d)


def mock_get_polygon(state: VehicleState) -> "Polygon":
    """Mock polygon function."""
    from shapely.geometry import Polygon

    return Polygon(
        [
            (state.x - 1, state.y - 1),
            (state.x + 1, state.y - 1),
            (state.x + 1, state.y + 1),
            (state.x - 1, state.y + 1),
        ]
    )


class TestSimulator:
    """Tests for Simulator class."""

    def test_initialization(self) -> None:
        """Test initialization with various parameters."""
        mock_update = MagicMock(side_effect=mock_update_state)

        # Default initialization
        sim = Simulator(step_update_func=mock_update, get_vehicle_polygon_func=mock_get_polygon)
        assert isinstance(sim.initial_state, VehicleState)
        assert sim.dt == 0.1
        assert isinstance(sim.vehicle_params, VehicleParameters)

        # Custom initialization
        custom_state = VehicleState(x=10.0, y=5.0, yaw=1.0, velocity=2.0)
        sim = Simulator(
            step_update_func=mock_update,
            get_vehicle_polygon_func=mock_get_polygon,
            initial_state=custom_state,
            dt=0.05,
        )

        assert sim.initial_state.x == 10.0
        assert sim.dt == 0.05
        # Internal state should be converted correctly
        assert sim._current_state.x == 10.0
        assert sim._current_state.y == 5.0

    def test_step_logic(self) -> None:
        """Test step method logic."""
        mock_update = MagicMock(side_effect=mock_update_state)
        sim = Simulator(step_update_func=mock_update, get_vehicle_polygon_func=mock_get_polygon)
        sim.reset()

        action = Action(steering=0.1, acceleration=0.5)

        # Step execution
        next_state, done, info = sim.step(action)

        # Check if update_state was called with correct action
        # mock_update calls: (state, action, dt)
        assert mock_update.call_count == 1
        call_args = mock_update.call_args
        assert call_args[0][1] == action
        assert call_args[0][2] == 0.1

        # Check conversion back to VehicleState
        assert isinstance(next_state, VehicleState)
        assert next_state.x == 1.0  # Mock update moves +1.0 in x
        assert next_state.acceleration == 0.5  # From action

        # Check logging
        log = sim.get_log()
        assert len(log.steps) == 1
        assert log.steps[0].action == action
        assert log.steps[0].vehicle_state == next_state
        assert not done

    def test_run_loop(self) -> None:
        """Test run method loop."""
        mock_update = MagicMock(side_effect=mock_update_state)
        sim = Simulator(step_update_func=mock_update, get_vehicle_polygon_func=mock_get_polygon)

        # Mock ADComponent
        from core.interfaces import ADComponent

        ad_component = MagicMock(spec=ADComponent)
        ad_component.planner = MagicMock()
        ad_component.planner.plan.return_value = []
        ad_component.controller = MagicMock()
        ad_component.controller.control.return_value = Action(steering=0.0, acceleration=0.0)

        # Run for 5 steps
        result = sim.run(ad_component, max_steps=5)

        assert isinstance(result, SimulationResult)
        assert not result.success  # Not reached goal
        assert result.reason == "max_steps"
        assert len(result.log.steps) == 5

        # Check interaction counts
        assert ad_component.planner.plan.call_count == 5
        assert ad_component.controller.control.call_count == 5

    def test_goal_check(self) -> None:
        """Test goal checking logic in run method."""
        mock_update = MagicMock(side_effect=mock_update_state)
        sim = Simulator(step_update_func=mock_update, get_vehicle_polygon_func=mock_get_polygon)

        # Mock ADComponent
        from core.interfaces import ADComponent

        ad_component = MagicMock(spec=ADComponent)
        ad_component.planner = MagicMock()
        ad_component.controller = MagicMock()
        ad_component.controller.control.return_value = Action(steering=0.0, acceleration=0.0)

        # Set goal in simulator
        sim.goal_x = 3.0
        sim.goal_y = 0.0

        result = sim.run(
            ad_component,
            max_steps=10,
            goal_threshold=0.1,
            min_elapsed_time=0.0,  # Immediate check
        )

        assert result.success
        assert result.reason == "goal_reached"
        # Should finish after step 3 (x=3.0)
        assert len(result.log.steps) == 3
