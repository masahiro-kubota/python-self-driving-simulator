"""Tests for abstract interfaces."""

from core.data import (
    Action,
    Observation,
    SimulationLog,
    Trajectory,
    TrajectoryPoint,
    VehicleState,
)
from core.interfaces import (
    Controller,
    Perception,
    Planner,
    Simulator,
)


class DummyPerception(Perception):
    """Dummy perception implementation for testing."""

    def perceive(self, sensor_data: object) -> Observation:
        """Return dummy observation."""
        return Observation(lateral_error=0.0, heading_error=0.0, velocity=5.0, target_velocity=5.0)

    def reset(self) -> None:
        """Reset perception."""


class DummyPlanner(Planner):
    """Dummy planner implementation for testing."""

    def plan(self, observation: Observation, vehicle_state: VehicleState) -> Trajectory:
        """Return dummy trajectory."""
        points = [
            TrajectoryPoint(x=0.0, y=0.0, yaw=0.0, velocity=5.0),
            TrajectoryPoint(x=1.0, y=0.0, yaw=0.0, velocity=5.0),
        ]
        return Trajectory(points=points)

    def reset(self) -> None:
        """Reset planner."""


class DummyController(Controller):
    """Dummy controller implementation for testing."""

    def control(
        self,
        trajectory: Trajectory,
        vehicle_state: VehicleState,
        observation: Observation | None = None,
    ) -> Action:
        """Return dummy action."""
        return Action(steering=0.0, acceleration=0.0)

    def reset(self) -> None:
        """Reset controller."""


class DummySimulator(Simulator):
    """Dummy simulator implementation for testing."""

    def __init__(self) -> None:
        """Initialize simulator."""
        self.state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0)

    def reset(self) -> VehicleState:
        """Reset simulator."""
        self.state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0)
        return self.state

    def step(self, action: Action) -> tuple[VehicleState, Observation, bool, dict[str, object]]:
        """Step simulator."""
        # Simple update
        self.state.x += 0.1
        obs = Observation(lateral_error=0.0, heading_error=0.0, velocity=5.0, target_velocity=5.0)
        done = False
        info: dict[str, object] = {}
        return self.state, obs, done, info

    def close(self) -> None:
        """Close simulator."""

    def render(self) -> None:
        """Render simulator."""

    def run(
        self,
        planner: Planner,
        controller: Controller,
        max_steps: int = 1000,
        reference_trajectory: Trajectory | None = None,
    ) -> object:
        """Run simulation (dummy implementation for testing)."""
        from core.data import SimulationResult

        return SimulationResult(
            success=True,
            total_steps=10,
            final_state=self.state,
            metrics={},
        )

    def get_log(self) -> SimulationLog:
        """Get simulation log."""
        return SimulationLog(steps=[], metadata={})


class TestPerceptionInterface:
    """Tests for Perception interface."""

    def test_implementation(self) -> None:
        """Test that implementation works."""
        perception = DummyPerception()
        obs = perception.perceive(None)
        assert isinstance(obs, Observation)

    def test_reset(self) -> None:
        """Test reset method."""
        perception = DummyPerception()
        perception.reset()  # Should not raise


class TestPlanningInterface:
    """Tests for Planner interface."""

    def test_implementation(self) -> None:
        """Test that implementation works."""
        planner = DummyPlanner()
        obs = Observation(lateral_error=0.0, heading_error=0.0, velocity=5.0, target_velocity=5.0)
        state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=5.0)
        traj = planner.plan(obs, state)
        assert isinstance(traj, Trajectory)
        assert len(traj) > 0

    def test_reset(self) -> None:
        """Test reset method."""
        planner = DummyPlanner()
        planner.reset()  # Should not raise


class TestControlInterface:
    """Tests for Controller interface."""

    def test_implementation(self) -> None:
        """Test that implementation works."""
        controller = DummyController()
        traj = Trajectory(points=[TrajectoryPoint(x=0.0, y=0.0, yaw=0.0, velocity=5.0)])
        state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=5.0)
        action = controller.control(traj, state)
        assert isinstance(action, Action)

    def test_with_observation(self) -> None:
        """Test control with observation."""
        controller = DummyController()
        traj = Trajectory(points=[TrajectoryPoint(x=0.0, y=0.0, yaw=0.0, velocity=5.0)])
        state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=5.0)
        obs = Observation(lateral_error=0.0, heading_error=0.0, velocity=5.0, target_velocity=5.0)
        action = controller.control(traj, state, obs)
        assert isinstance(action, Action)

    def test_reset(self) -> None:
        """Test reset method."""
        controller = DummyController()
        controller.reset()  # Should not raise


class TestSimulatorInterface:
    """Tests for Simulator interface."""

    def test_reset(self) -> None:
        """Test simulator reset."""
        sim = DummySimulator()
        state = sim.reset()
        assert isinstance(state, VehicleState)
        assert state.x == 0.0

    def test_step(self) -> None:
        """Test simulator step."""
        sim = DummySimulator()
        sim.reset()
        action = Action(steering=0.0, acceleration=1.0)
        state, obs, done, info = sim.step(action)

        assert isinstance(state, VehicleState)
        assert isinstance(obs, Observation)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_multiple_steps(self) -> None:
        """Test multiple simulator steps."""
        sim = DummySimulator()
        sim.reset()

        for _ in range(10):
            action = Action(steering=0.0, acceleration=1.0)
            state, obs, done, info = sim.step(action)
            assert isinstance(state, VehicleState)

    def test_close(self) -> None:
        """Test simulator close."""
        sim = DummySimulator()
        sim.close()  # Should not raise

    def test_render(self) -> None:
        """Test simulator render."""
        sim = DummySimulator()
        sim.render()  # Should not raise


class TestPipelineIntegration:
    """Integration tests for component pipeline."""

    def test_full_pipeline(self) -> None:
        """Test full pipeline from perception to control."""
        # Create components
        perception = DummyPerception()
        planner = DummyPlanner()
        controller = DummyController()
        simulator = DummySimulator()

        # Reset
        state = simulator.reset()

        # Run one step
        sensor_data = None  # Dummy sensor data
        obs = perception.perceive(sensor_data)
        traj = planner.plan(obs, state)
        action = controller.control(traj, state, obs)
        new_state, new_obs, done, info = simulator.step(action)

        # Verify types
        assert isinstance(new_state, VehicleState)
        assert isinstance(new_obs, Observation)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_multi_step_pipeline(self) -> None:
        """Test pipeline over multiple steps."""
        perception = DummyPerception()
        planner = DummyPlanner()
        controller = DummyController()
        simulator = DummySimulator()

        state = simulator.reset()

        for _ in range(5):
            obs = perception.perceive(None)
            traj = planner.plan(obs, state)
            action = controller.control(traj, state, obs)
            state, obs, done, info = simulator.step(action)

            if done:
                break

        # Should complete without errors
        assert isinstance(state, VehicleState)
