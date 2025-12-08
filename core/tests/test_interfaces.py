"""Tests for abstract interfaces."""

from core.data import (
    Action,
    Observation,
    SimulationLog,
    VehicleState,
)
from core.interfaces import Simulator


class DummySimulator(Simulator):
    """Dummy simulator implementation for testing."""

    def __init__(self) -> None:
        """Initialize simulator."""
        self.state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0)

    def reset(self) -> VehicleState:
        """Reset simulator."""
        self.state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0)
        return self.state

    def step(self, _action: Action) -> tuple[VehicleState, Observation, bool, dict[str, object]]:
        """Step simulator."""
        # Simple update
        self.state.x += 0.1
        obs = Observation(lateral_error=0.0, heading_error=0.0, velocity=5.0, target_velocity=5.0)
        done = False
        info: dict[str, object] = {}
        return self.state, obs, done, info

    def close(self) -> bool:
        """Close simulator."""
        return True

    def render(self) -> None:
        """Render simulator."""

    def run(
        self,
        _planner: object,
        _controller: object,
        _max_steps: int = 1000,
        _reference_trajectory: object | None = None,
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
