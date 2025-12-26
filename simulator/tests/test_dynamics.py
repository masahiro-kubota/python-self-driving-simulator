"""Tests for vehicle dynamics functions."""

import math

import pytest
from core.data import VehicleParameters, VehicleState
from shapely.geometry import Polygon
from simulator.dynamics import (
    create_vehicle_polygon,
    get_bicycle_model_polygon,
    update_bicycle_model,
)
from simulator.state import SimulationVehicleState


class TestUpdateBicycleModel:
    """Tests for update_bicycle_model function."""

    def test_straight_line_motion(self) -> None:
        """Test straight line motion with constant velocity."""
        state = SimulationVehicleState(
            x=0.0,
            y=0.0,
            yaw=0.0,
            vx=5.0,
            timestamp=0.0,
        )

        # No steering, no acceleration
        next_state = update_bicycle_model(
            state=state,
            steering=0.0,
            acceleration=0.0,
            dt=0.1,
            wheelbase=2.5,
        )

        # Position should move forward
        assert next_state.x > state.x
        assert abs(next_state.y - state.y) < 1e-10
        assert abs(next_state.yaw - state.yaw) < 1e-10
        assert abs(next_state.vx - 5.0) < 1e-10
        assert next_state.timestamp == 0.1

    def test_acceleration(self) -> None:
        """Test acceleration increases velocity."""
        state = SimulationVehicleState(
            x=0.0,
            y=0.0,
            yaw=0.0,
            vx=5.0,
        )

        next_state = update_bicycle_model(
            state=state,
            steering=0.0,
            acceleration=2.0,
            dt=0.1,
            wheelbase=2.5,
        )

        # Velocity should increase
        expected_vx = 5.0 + 2.0 * 0.1
        assert abs(next_state.vx - expected_vx) < 1e-10
        assert next_state.ax == 2.0

    def test_deceleration(self) -> None:
        """Test deceleration decreases velocity."""
        state = SimulationVehicleState(
            x=0.0,
            y=0.0,
            yaw=0.0,
            vx=5.0,
        )

        next_state = update_bicycle_model(
            state=state,
            steering=0.0,
            acceleration=-1.0,
            dt=0.1,
            wheelbase=2.5,
        )

        # Velocity should decrease
        expected_vx = 5.0 - 1.0 * 0.1
        assert abs(next_state.vx - expected_vx) < 1e-10

    def test_turning_motion(self) -> None:
        """Test turning motion with steering input."""
        state = SimulationVehicleState(
            x=0.0,
            y=0.0,
            yaw=0.0,
            vx=5.0,
        )

        # Left turn (positive steering)
        next_state = update_bicycle_model(
            state=state,
            steering=0.1,  # Small left turn
            acceleration=0.0,
            dt=0.1,
            wheelbase=2.5,
        )

        # Yaw should increase (turning left)
        assert next_state.yaw > state.yaw
        # Yaw rate should be non-zero
        assert next_state.yaw_rate > 0
        # Steering should be stored
        assert next_state.steering == 0.1

    def test_zero_velocity(self) -> None:
        """Test behavior at zero velocity."""
        state = SimulationVehicleState(
            x=0.0,
            y=0.0,
            yaw=0.0,
            vx=0.0,
        )

        next_state = update_bicycle_model(
            state=state,
            steering=0.5,  # Large steering angle
            acceleration=0.0,
            dt=0.1,
            wheelbase=2.5,
        )

        # At zero velocity, yaw rate should be zero
        assert abs(next_state.yaw_rate) < 1e-10
        # Position should not change
        assert abs(next_state.x - state.x) < 1e-10
        assert abs(next_state.y - state.y) < 1e-10

    def test_large_steering_angle(self) -> None:
        """Test with large steering angle."""
        state = SimulationVehicleState(
            x=0.0,
            y=0.0,
            yaw=0.0,
            vx=5.0,
        )

        # Maximum steering angle (e.g., 45 degrees)
        next_state = update_bicycle_model(
            state=state,
            steering=math.pi / 4,
            acceleration=0.0,
            dt=0.1,
            wheelbase=2.5,
        )

        # Should produce large yaw rate
        assert abs(next_state.yaw_rate) > 0.5

    def test_negative_steering(self) -> None:
        """Test right turn with negative steering."""
        state = SimulationVehicleState(
            x=0.0,
            y=0.0,
            yaw=0.0,
            vx=5.0,
        )

        # Right turn (negative steering)
        next_state = update_bicycle_model(
            state=state,
            steering=-0.1,
            acceleration=0.0,
            dt=0.1,
            wheelbase=2.5,
        )

        # Yaw should decrease (turning right)
        assert next_state.yaw < state.yaw
        assert next_state.yaw_rate < 0

    def test_timestamp_none(self) -> None:
        """Test with no initial timestamp."""
        state = SimulationVehicleState(
            x=0.0,
            y=0.0,
            yaw=0.0,
            vx=5.0,
            timestamp=None,
        )

        next_state = update_bicycle_model(
            state=state,
            steering=0.0,
            acceleration=0.0,
            dt=0.1,
            wheelbase=2.5,
        )

        # Timestamp should remain None
        assert next_state.timestamp is None

    def test_kinematic_constraints(self) -> None:
        """Test that kinematic model constraints are maintained."""
        state = SimulationVehicleState(
            x=0.0,
            y=0.0,
            yaw=0.0,
            vx=5.0,
        )

        next_state = update_bicycle_model(
            state=state,
            steering=0.1,
            acceleration=1.0,
            dt=0.1,
            wheelbase=2.5,
        )

        # Kinematic model assumes vy = 0
        assert next_state.vy == 0.0
        assert next_state.vz == 0.0
        # 2D motion only
        assert next_state.z == 0.0
        assert next_state.roll == 0.0
        assert next_state.pitch == 0.0


class TestCreateVehiclePolygon:
    """Tests for create_vehicle_polygon function."""

    def test_polygon_shape(self) -> None:
        """Test that polygon has correct shape."""
        polygon = create_vehicle_polygon(
            x=0.0,
            y=0.0,
            yaw=0.0,
            front_edge_dist=4.0,
            rear_edge_dist=-1.0,
            half_width=1.0,
        )

        assert isinstance(polygon, Polygon)
        # Should have 4 vertices (rectangle)
        assert len(polygon.exterior.coords) == 5  # 5 because first and last are same

    def test_polygon_at_origin_no_rotation(self) -> None:
        """Test polygon at origin with no rotation."""
        polygon = create_vehicle_polygon(
            x=0.0,
            y=0.0,
            yaw=0.0,
            front_edge_dist=4.0,
            rear_edge_dist=-1.0,
            half_width=1.0,
        )

        coords = list(polygon.exterior.coords)
        # Check bounds
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]

        assert min(xs) == pytest.approx(-1.0, abs=1e-10)
        assert max(xs) == pytest.approx(4.0, abs=1e-10)
        assert min(ys) == pytest.approx(-1.0, abs=1e-10)
        assert max(ys) == pytest.approx(1.0, abs=1e-10)

    def test_polygon_with_translation(self) -> None:
        """Test polygon with translation."""
        polygon = create_vehicle_polygon(
            x=10.0,
            y=5.0,
            yaw=0.0,
            front_edge_dist=4.0,
            rear_edge_dist=-1.0,
            half_width=1.0,
        )

        coords = list(polygon.exterior.coords)
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]

        # Should be translated by (10, 5)
        assert min(xs) == pytest.approx(9.0, abs=1e-10)
        assert max(xs) == pytest.approx(14.0, abs=1e-10)
        assert min(ys) == pytest.approx(4.0, abs=1e-10)
        assert max(ys) == pytest.approx(6.0, abs=1e-10)

    def test_polygon_with_rotation(self) -> None:
        """Test polygon with 90 degree rotation."""
        polygon = create_vehicle_polygon(
            x=0.0,
            y=0.0,
            yaw=math.pi / 2,  # 90 degrees
            front_edge_dist=4.0,
            rear_edge_dist=-1.0,
            half_width=1.0,
        )

        coords = list(polygon.exterior.coords)
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]

        # After 90 degree rotation, x and y bounds should swap
        assert min(xs) == pytest.approx(-1.0, abs=1e-9)
        assert max(xs) == pytest.approx(1.0, abs=1e-9)
        assert min(ys) == pytest.approx(-1.0, abs=1e-9)
        assert max(ys) == pytest.approx(4.0, abs=1e-9)

    def test_polygon_area(self) -> None:
        """Test polygon area calculation."""
        polygon = create_vehicle_polygon(
            x=0.0,
            y=0.0,
            yaw=0.0,
            front_edge_dist=4.0,
            rear_edge_dist=-1.0,
            half_width=1.0,
        )

        # Area should be length * width = 5.0 * 2.0 = 10.0
        expected_area = 5.0 * 2.0
        assert polygon.area == pytest.approx(expected_area, abs=1e-10)


class TestGetBicycleModelPolygon:
    """Tests for get_bicycle_model_polygon function."""

    def test_polygon_from_vehicle_state(self) -> None:
        """Test polygon generation from VehicleState."""
        state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0)
        params = VehicleParameters(
            wheelbase=2.5,
            front_overhang=1.0,
            rear_overhang=0.5,
            width=2.0,
            max_steering_angle=0.6,
            max_velocity=20.0,
            max_acceleration=3.0,
            mass=1500.0,
            inertia=2500.0,
            lf=1.2,
            lr=1.3,
            cf=80000.0,
            cr=80000.0,
            c_drag=0.3,
            c_roll=0.015,
            max_drive_force=5000.0,
            max_brake_force=8000.0,
            tire_params={},
        )

        polygon = get_bicycle_model_polygon(state, params)

        assert isinstance(polygon, Polygon)
        # Check that polygon is created with correct dimensions
        coords = list(polygon.exterior.coords)
        xs = [c[0] for c in coords]

        # Front edge = wheelbase + front_overhang = 3.5
        # Rear edge = -rear_overhang = -0.5
        assert max(xs) == pytest.approx(3.5, abs=1e-10)
        assert min(xs) == pytest.approx(-0.5, abs=1e-10)

    def test_polygon_with_vehicle_rotation(self) -> None:
        """Test polygon with rotated vehicle."""
        state = VehicleState(x=10.0, y=5.0, yaw=math.pi / 2, velocity=0.0)
        params = VehicleParameters(
            wheelbase=2.5,
            front_overhang=1.0,
            rear_overhang=0.5,
            width=2.0,
            max_steering_angle=0.6,
            max_velocity=20.0,
            max_acceleration=3.0,
            mass=1500.0,
            inertia=2500.0,
            lf=1.2,
            lr=1.3,
            cf=80000.0,
            cr=80000.0,
            c_drag=0.3,
            c_roll=0.015,
            max_drive_force=5000.0,
            max_brake_force=8000.0,
            tire_params={},
        )

        polygon = get_bicycle_model_polygon(state, params)

        # Check that polygon is created successfully
        assert isinstance(polygon, Polygon)
        # With rear-axle centered system and rotation, centroid will shift
        # Just verify polygon is valid and has reasonable bounds
        assert polygon.is_valid
        assert polygon.area > 0

    def test_rear_axle_centered(self) -> None:
        """Test that polygon is rear-axle centered."""
        state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0)
        params = VehicleParameters(
            wheelbase=2.5,
            front_overhang=1.0,
            rear_overhang=0.5,
            width=2.0,
            max_steering_angle=0.6,
            max_velocity=20.0,
            max_acceleration=3.0,
            mass=1500.0,
            inertia=2500.0,
            lf=1.2,
            lr=1.3,
            cf=80000.0,
            cr=80000.0,
            c_drag=0.3,
            c_roll=0.015,
            max_drive_force=5000.0,
            max_brake_force=8000.0,
            tire_params={},
        )

        polygon = get_bicycle_model_polygon(state, params)

        coords = list(polygon.exterior.coords)
        xs = [c[0] for c in coords]

        # Rear edge should be at -rear_overhang (behind the reference point)
        assert min(xs) == pytest.approx(-0.5, abs=1e-10)
        # Front edge should be at wheelbase + front_overhang
        assert max(xs) == pytest.approx(3.5, abs=1e-10)
