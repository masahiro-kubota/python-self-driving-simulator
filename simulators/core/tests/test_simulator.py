"""Tests for BaseSimulator."""

from unittest.mock import MagicMock

from simulator_core.data import SimulationVehicleState
from simulator_core.simulator import BaseSimulator

from core.data import Action, SimulationResult, VehicleParameters, VehicleState


class MockSimulator(BaseSimulator):
    """Mock simulator for testing BaseSimulator."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_called_with = None

    def _update_state(self, action: Action) -> SimulationVehicleState:
        self.update_called_with = action
        # Simple update: move 1.0m in x direction
        current = self._current_state
        return SimulationVehicleState(
            x=current.x + 1.0,
            y=current.y,
            z=current.z,
            roll=current.roll,
            pitch=current.pitch,
            yaw=current.yaw,
            vx=current.vx,
            vy=current.vy,
            vz=current.vz,
            roll_rate=current.roll_rate,
            pitch_rate=current.pitch_rate,
            yaw_rate=current.yaw_rate,
            ax=action.acceleration,
            ay=current.ay,
            az=current.az,
            steering=action.steering,
            throttle=current.throttle,
            timestamp=(current.timestamp or 0.0) + self.dt,
        )


class TestBaseSimulator:
    """Tests for BaseSimulator."""

    def test_initialization(self) -> None:
        """Test initialization with various parameters."""
        # Default initialization
        sim = MockSimulator()
        assert isinstance(sim.initial_state, VehicleState)
        assert sim.dt == 0.1
        assert isinstance(sim.vehicle_params, VehicleParameters)

        # Custom initialization
        custom_state = VehicleState(x=10.0, y=5.0, yaw=1.0, velocity=2.0)
        sim = MockSimulator(initial_state=custom_state, dt=0.05)

        assert sim.initial_state.x == 10.0
        assert sim.dt == 0.05
        # Internal state should be converted correctly
        assert sim._current_state.x == 10.0
        assert sim._current_state.y == 5.0

    def test_step_logic(self) -> None:
        """Test step method logic."""
        sim = MockSimulator()
        sim.reset()

        action = Action(steering=0.1, acceleration=0.5)

        # Step execution
        next_state, done, info = sim.step(action)

        # Check if _update_state was called with correct action
        assert sim.update_called_with == action

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
        sim = MockSimulator()

        # Mock planner and controller
        planner = MagicMock()
        planner.plan.return_value = []

        controller = MagicMock()
        controller.control.return_value = Action(steering=0.0, acceleration=0.0)

        # Run for 5 steps
        result = sim.run(planner, controller, max_steps=5)

        assert isinstance(result, SimulationResult)
        assert not result.success  # Not reached goal
        assert result.reason == "max_steps"
        assert len(result.log.steps) == 5

        # Check interaction counts
        assert planner.plan.call_count == 5
        assert controller.control.call_count == 5

    def test_goal_check(self) -> None:
        """Test goal checking logic in run method."""
        sim = MockSimulator()

        # Mock planner/controller to do nothing
        planner = MagicMock()
        controller = MagicMock()
        controller.control.return_value = Action(steering=0.0, acceleration=0.0)

        # Define a goal trajectory near the starting point + movement
        from core.data import TrajectoryPoint

        # Goal at x=3.0 (reached after 3 steps)
        reference_trajectory = [TrajectoryPoint(x=3.0, y=0.0, yaw=0.0, velocity=0.0)]

        # Run with sufficient steps but short validation time
        # MockSimulator moves +1.0 x per step.
        # Step 1: x=1.0
        # Step 2: x=2.0
        # Step 3: x=3.0 (Goal reached)

        result = sim.run(
            planner,
            controller,
            max_steps=10,
            reference_trajectory=reference_trajectory,
            goal_threshold=0.1,
            min_elapsed_time=0.0,  # Immediate check
        )

        assert result.success
        assert result.reason == "goal_reached"
        # Should finish after step 3 (x=3.0)
        assert len(result.log.steps) == 3
