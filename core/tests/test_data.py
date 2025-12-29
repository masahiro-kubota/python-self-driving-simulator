"""Tests for core data structures."""

import numpy as np
from core.data import Observation, VehicleState


class TestVehicleState:
    """Tests for VehicleState data class."""

    def test_creation(self) -> None:
        """Test basic creation of VehicleState."""
        state = VehicleState(x=1.0, y=2.0, yaw=0.5, velocity=5.0)
        assert state.x == 1.0
        assert state.y == 2.0
        assert state.yaw == 0.5
        assert state.velocity == 5.0
        assert state.acceleration == 0.0
        assert state.steering == 0.0

    def test_to_array(self) -> None:
        """Test conversion to numpy array."""
        state = VehicleState(x=1.0, y=2.0, yaw=0.5, velocity=5.0)
        arr = state.to_array()
        expected = np.array([1.0, 2.0, 0.5, 5.0])
        np.testing.assert_array_equal(arr, expected)

    def test_from_array(self) -> None:
        """Test creation from numpy array."""
        arr = np.array([1.0, 2.0, 0.5, 5.0])
        state = VehicleState.from_array(arr)
        assert state.x == 1.0
        assert state.y == 2.0
        assert state.yaw == 0.5
        assert state.velocity == 5.0

    def test_roundtrip(self) -> None:
        """Test array conversion roundtrip."""
        original = VehicleState(x=1.5, y=2.5, yaw=0.7, velocity=10.0)
        arr = original.to_array()
        restored = VehicleState.from_array(arr)
        assert restored.x == original.x
        assert restored.y == original.y
        assert restored.yaw == original.yaw
        assert restored.velocity == original.velocity


class TestObservation:
    """Tests for Observation data class."""

    def test_creation(self) -> None:
        """Test basic creation of Observation."""
        obs = Observation(lateral_error=0.5, heading_error=0.1, velocity=5.0, target_velocity=6.0)
        assert obs.lateral_error == 0.5
        assert obs.heading_error == 0.1
        assert obs.velocity == 5.0
        assert obs.target_velocity == 6.0

    def test_to_array(self) -> None:
        """Test conversion to numpy array."""
        obs = Observation(lateral_error=0.5, heading_error=0.1, velocity=5.0, target_velocity=6.0)
        arr = obs.to_array()
        expected = np.array([0.5, 0.1, 5.0, 6.0])
        np.testing.assert_array_equal(arr, expected)

    def test_from_array(self) -> None:
        """Test creation from numpy array."""
        arr = np.array([0.5, 0.1, 5.0, 6.0])
        obs = Observation.from_array(arr)
        assert obs.lateral_error == 0.5
        assert obs.heading_error == 0.1
        assert obs.velocity == 5.0
        assert obs.target_velocity == 6.0
