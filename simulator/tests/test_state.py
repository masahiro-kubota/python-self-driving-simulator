"""Tests for SimulationVehicleState."""

from core.data import Action, VehicleState
from simulator.state import SimulationVehicleState


class TestSimulationVehicleState:
    """Tests for SimulationVehicleState."""

    def test_from_vehicle_state(self) -> None:
        """Test conversion from VehicleState."""
        vehicle_state = VehicleState(
            x=10.0,
            y=20.0,
            yaw=1.57,
            velocity=5.0,
            acceleration=1.0,
            steering=0.5,
            timestamp=100.0,
        )

        dynamic_state = SimulationVehicleState.from_vehicle_state(vehicle_state)

        assert dynamic_state.x == 10.0
        assert dynamic_state.y == 20.0
        assert dynamic_state.z == 0.0
        assert dynamic_state.yaw == 1.57
        assert dynamic_state.roll == 0.0
        assert dynamic_state.pitch == 0.0

        # Velocity decomposition (yaw is ignored in conversion, vx = velocity, vy = 0)
        assert abs(dynamic_state.vx - 5.0) < 1e-10
        assert dynamic_state.vy == 0.0
        assert dynamic_state.vz == 0.0

        assert dynamic_state.ax == 1.0
        assert dynamic_state.steering == 0.5
        assert dynamic_state.timestamp == 100.0

    def test_to_vehicle_state(self) -> None:
        """Test conversion to VehicleState."""
        dynamic_state = SimulationVehicleState(
            x=10.0,
            y=20.0,
            z=5.0,
            roll=0.1,
            pitch=0.2,
            yaw=1.57,
            vx=3.0,
            vy=4.0,  # Velocity = 5.0
            vz=0.0,
            ax=0.5,
            steering=0.1,
            timestamp=100.0,
        )

        # Test without action
        vehicle_state = dynamic_state.to_vehicle_state()

        assert vehicle_state.x == 10.0
        assert vehicle_state.y == 20.0
        assert vehicle_state.yaw == 1.57
        assert abs(vehicle_state.velocity - 5.0) < 1e-10  # sqrt(3^2 + 4^2)
        assert vehicle_state.acceleration == 0.5
        assert vehicle_state.steering == 0.1
        assert vehicle_state.timestamp == 100.0

        # Test with action
        action = Action(steering=0.2, acceleration=1.0)
        vehicle_state_with_action = dynamic_state.to_vehicle_state(action)

        assert vehicle_state_with_action.acceleration == 1.0
        assert vehicle_state_with_action.steering == 0.2
        # Other fields should remain same
        assert vehicle_state_with_action.x == 10.0
