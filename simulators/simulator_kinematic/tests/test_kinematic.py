"""Tests for Kinematic Simulator."""

import math

from simulator_core.data import DynamicVehicleState
from simulator_kinematic import KinematicSimulator
from simulator_kinematic.vehicle import KinematicVehicleModel

from core.data import Action, VehicleState


class TestKinematicVehicleModel:
    """Tests for KinematicVehicleModel."""

    def test_straight_line(self) -> None:
        """Test straight line motion."""
        model = KinematicVehicleModel(wheelbase=2.5)
        state = DynamicVehicleState(
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

        # Move straight for 1 second
        new_state = model.step(state, steering=0.0, acceleration=0.0, dt=1.0)

        assert abs(new_state.x - 10.0) < 1e-10
        assert abs(new_state.y - 0.0) < 1e-10
        assert abs(new_state.yaw - 0.0) < 1e-10
        assert abs(new_state.vx - 10.0) < 1e-10
        # Kinematic model maintains vy = 0
        assert abs(new_state.vy) < 1e-10

    def test_acceleration(self) -> None:
        """Test acceleration."""
        model = KinematicVehicleModel(wheelbase=2.5)
        state = DynamicVehicleState(
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

        # Accelerate at 2.0 m/s^2 for 1 second
        new_state = model.step(state, steering=0.0, acceleration=2.0, dt=1.0)

        assert abs(new_state.vx - 2.0) < 1e-10
        # Euler: x_next = x + vx * dt = 0 + 0 * 1 = 0
        # vx_next = vx + a * dt = 0 + 2 * 1 = 2
        assert abs(new_state.x - 0.0) < 1e-10
        assert abs(new_state.vy) < 1e-10

    def test_turning(self) -> None:
        """Test turning motion."""
        model = KinematicVehicleModel(wheelbase=2.5)
        state = DynamicVehicleState(
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

        # Turn left with steering angle
        steering = math.atan(2.5 / 10.0)  # Radius = L / tan(delta) = 10.0
        # yaw_rate = vx / R = 5.0 / 10.0 = 0.5 rad/s

        new_state = model.step(state, steering=steering, acceleration=0.0, dt=1.0)

        # Expected yaw change = 0.5 * 1.0 = 0.5 rad
        assert abs(new_state.yaw - 0.5) < 1e-10
        # yaw_rate should be calculated
        assert abs(new_state.yaw_rate - 0.5) < 1e-10

    def test_backward_motion(self) -> None:
        """Test backward motion."""
        model = KinematicVehicleModel(wheelbase=2.5)
        # 後退速度
        state = DynamicVehicleState(
            x=0.0,
            y=0.0,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            vx=-5.0,
            vy=0.0,
            vz=0.0,
            roll_rate=0.0,
            pitch_rate=0.0,
            yaw_rate=0.0,
        )

        # ステアリングを切って後退
        # 後退時の旋回中心は変わらないが、yaw_rateの符号は速度に依存するはず
        # yaw_rate = v * tan(delta) / L
        # v < 0 なので yaw_rate も符号が反転するはず
        steering = 0.5
        new_state = model.step(state, steering=steering, acceleration=0.0, dt=1.0)

        # X座標は減少
        assert new_state.x < 0.0
        # 速度は維持(加速度0)
        assert abs(new_state.vx - (-5.0)) < 1e-10
        # Yawレートは負になるはず(左ハンドルで後退=車体後部は右へ=Yawは右回転=負?)
        # 数式: yaw_rate = v * tan(delta) / L
        # -5 * tan(0.5) / 2.5 ~= -1.09
        expected_yaw_rate = -5.0 * math.tan(0.5) / 2.5
        assert abs(new_state.yaw_rate - expected_yaw_rate) < 1e-10


class TestKinematicSimulator:
    """Tests for KinematicSimulator."""

    def test_initialization(self) -> None:
        """Test initialization."""
        sim = KinematicSimulator()
        state = sim.reset()
        assert state.x == 0.0
        assert state.y == 0.0
        assert state.velocity == 0.0

    def test_step(self) -> None:
        """Test step execution."""
        sim = KinematicSimulator(dt=0.1)
        sim.reset()

        action = Action(steering=0.0, acceleration=1.0)
        state, done, info = sim.step(action)

        assert isinstance(state, VehicleState)
        assert state.velocity > 0.0
        assert not done
        assert isinstance(info, dict)

    def test_custom_initial_state(self) -> None:
        """Test with custom initial state."""
        initial_state = VehicleState(x=10.0, y=5.0, yaw=1.0, velocity=5.0)
        sim = KinematicSimulator(initial_state=initial_state)
        state = sim.reset()

        assert state.x == 10.0
        assert state.y == 5.0
        assert state.yaw == 1.0
        assert state.velocity == 5.0
