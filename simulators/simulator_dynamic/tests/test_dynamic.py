"""Tests for Dynamic Simulator."""

from simulator_core.data import SimulationVehicleState
from simulator_dynamic import DynamicSimulator
from simulator_dynamic.vehicle import DynamicVehicleModel

from core.data import Action, VehicleParameters, VehicleState


class TestVehicleParameters:
    """Tests for VehicleParameters."""

    def test_default_parameters(self) -> None:
        """Test default parameters."""
        params = VehicleParameters()
        assert params.mass == 1500.0
        assert params.wheelbase == 2.5
        assert abs(params.lf + params.lr - params.wheelbase) < 1e-6

    def test_custom_parameters(self) -> None:
        """Test custom parameters."""
        params = VehicleParameters(lf=1.5, lr=1.0, wheelbase=2.5)
        assert params.lf == 1.5
        assert params.lr == 1.0


class TestDynamicVehicleModel:
    """Tests for DynamicVehicleModel."""

    def test_straight_line_low_speed(self) -> None:
        """Test straight line motion derivative."""
        model = DynamicVehicleModel()
        state = SimulationVehicleState(
            x=0.0,
            y=0.0,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            vx=5.0,
            vy=0.0,
            vz=0.0,
            roll_rate=0.0,
            pitch_rate=0.0,
            yaw_rate=0.0,
        )

        # Get derivatives for straight motion
        derivative = model.calculate_derivative(state, steering=0.0, throttle=0.0)

        # x_dot should be approx vx (5.0) in global frame if yaw is 0
        assert abs(derivative.x - 5.0) < 1e-1
        assert abs(derivative.y) < 1e-1
        # Should simulate drag, so vx should decrease (negative derivative)
        # unless throttle compensates. Here throttle is 0.
        assert derivative.vx < 0.0

    def test_acceleration(self) -> None:
        """Test acceleration derivative."""
        model = DynamicVehicleModel()
        state = SimulationVehicleState(
            x=0.0,
            y=0.0,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            vx=0.0,
            vy=0.0,
            vz=0.0,
            roll_rate=0.0,
            pitch_rate=0.0,
            yaw_rate=0.0,
        )

        # Apply throttle
        derivative = model.calculate_derivative(state, steering=0.0, throttle=0.5)

        # vx_dot should be positive
        assert derivative.vx > 0.0

    def test_lateral_slip(self) -> None:
        """Test lateral slip generation during turn."""
        model = DynamicVehicleModel()
        state = SimulationVehicleState(
            x=0.0,
            y=0.0,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            vx=10.0,  # 高速
            vy=0.0,
            vz=0.0,
            roll_rate=0.0,
            pitch_rate=0.0,
            yaw_rate=0.0,
        )

        # 急ハンドル
        derivative = model.calculate_derivative(state, steering=0.5, throttle=0.0)

        # 横滑り成分(vyの変化率)が発生するはず
        # 直進状態(vy=0)から操舵すると、タイヤ横力によりayが発生し、vyが変化する
        assert abs(derivative.vy) > 0.01

    def test_braking(self) -> None:
        """Test braking (negative throttle)."""
        model = DynamicVehicleModel()
        state = SimulationVehicleState(
            x=0.0,
            y=0.0,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            vx=10.0,
            vy=0.0,
            vz=0.0,
            roll_rate=0.0,
            pitch_rate=0.0,
            yaw_rate=0.0,
        )

        # Full brake
        derivative = model.calculate_derivative(state, steering=0.0, throttle=-1.0)

        # 減速するはず(vx_dot < 0)
        # 抗力もあるので、アクセルオフよりも強く減速するはず
        derivative_coast = model.calculate_derivative(state, steering=0.0, throttle=0.0)

        assert derivative.vx < 0.0
        assert derivative.vx < derivative_coast.vx


class TestDynamicSimulator:
    """Tests for DynamicSimulator."""

    def test_initialization(self) -> None:
        """Test initialization."""
        sim = DynamicSimulator()
        state = sim.reset()
        assert state.x == 0.0
        assert state.y == 0.0
        assert state.velocity == 0.0

    def test_step(self) -> None:
        """Test step execution."""
        sim = DynamicSimulator(dt=0.01)
        sim.reset()

        action = Action(steering=0.0, acceleration=1.0)
        state, done, info = sim.step(action)

        assert isinstance(state, VehicleState)
        assert not done
        assert isinstance(info, dict)

    def test_custom_initial_state(self) -> None:
        """Test with custom initial state."""
        initial_state = VehicleState(x=10.0, y=5.0, yaw=1.0, velocity=5.0)
        sim = DynamicSimulator(initial_state=initial_state)
        state = sim.reset()

        assert state.x == 10.0
        assert state.y == 5.0
        assert state.yaw == 1.0
        assert state.velocity == 5.0
