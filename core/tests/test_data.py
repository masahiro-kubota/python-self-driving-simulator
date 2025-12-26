"""Tests for core data structures."""

import numpy as np
from core.data import Action, Observation, Trajectory, TrajectoryPoint, VehicleState


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


class TestAction:
    """Tests for Action data class."""

    def test_creation(self) -> None:
        """Test basic creation of Action."""
        action = Action(steering=0.1, acceleration=1.0)
        assert action.steering == 0.1
        assert action.acceleration == 1.0

    def test_to_array(self) -> None:
        """Test conversion to numpy array."""
        action = Action(steering=0.1, acceleration=1.0)
        arr = action.to_array()
        expected = np.array([0.1, 1.0])
        np.testing.assert_array_equal(arr, expected)

    def test_from_array(self) -> None:
        """Test creation from numpy array."""
        arr = np.array([0.1, 1.0])
        action = Action.from_array(arr)
        assert action.steering == 0.1
        assert action.acceleration == 1.0


class TestTrajectory:
    """Tests for Trajectory data class."""

    def test_creation(self) -> None:
        """Test basic creation of Trajectory."""
        points = [
            TrajectoryPoint(x=0.0, y=0.0, yaw=0.0, velocity=5.0),
            TrajectoryPoint(x=1.0, y=0.0, yaw=0.0, velocity=5.0),
            TrajectoryPoint(x=2.0, y=0.0, yaw=0.0, velocity=5.0),
        ]
        traj = Trajectory(points=points)
        assert len(traj) == 3

    def test_indexing(self) -> None:
        """Test trajectory indexing."""
        points = [
            TrajectoryPoint(x=0.0, y=0.0, yaw=0.0, velocity=5.0),
            TrajectoryPoint(x=1.0, y=0.0, yaw=0.0, velocity=5.0),
        ]
        traj = Trajectory(points=points)
        assert traj[0].x == 0.0
        assert traj[1].x == 1.0

    def test_to_arrays(self) -> None:
        """Test conversion to numpy arrays."""
        points = [
            TrajectoryPoint(x=0.0, y=0.0, yaw=0.0, velocity=5.0),
            TrajectoryPoint(x=1.0, y=1.0, yaw=0.5, velocity=6.0),
            TrajectoryPoint(x=2.0, y=2.0, yaw=1.0, velocity=7.0),
        ]
        traj = Trajectory(points=points)
        x, y, yaw, velocity = traj.to_arrays()

        np.testing.assert_array_equal(x, np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(y, np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(yaw, np.array([0.0, 0.5, 1.0]))
        np.testing.assert_array_equal(velocity, np.array([5.0, 6.0, 7.0]))

    def test_empty_trajectory(self) -> None:
        """Test empty trajectory."""
        traj = Trajectory(points=[])
        assert len(traj) == 0
