"""Tests for control components."""

import math

import pytest
from components.control.pid import PIDController
from core.data import Action, Trajectory, TrajectoryPoint, VehicleState


class TestPIDController:
    """Tests for PIDController."""

    def test_steering_control(self) -> None:
        """Test steering control logic (Pure Pursuit)."""
        controller = PIDController(wheelbase=2.0)
        state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=5.0)
        
        # Target at (2, 2) -> 45 degrees
        target = TrajectoryPoint(x=2.0, y=2.0, yaw=0.0, velocity=5.0)
        traj = Trajectory(points=[target])
        
        action = controller.control(traj, state)
        
        # alpha = 45 deg = pi/4
        # Ld = sqrt(8) = 2.828
        # delta = atan(2 * L * sin(alpha) / Ld)
        # delta = atan(2 * 2 * 0.707 / 2.828) = atan(1) = pi/4
        assert abs(action.steering - math.pi/4) < 0.1

    def test_velocity_control(self) -> None:
        """Test velocity control logic (PID)."""
        controller = PIDController(kp=1.0, ki=0.0, kd=0.0)
        state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0)
        
        # Target velocity 10.0
        target = TrajectoryPoint(x=10.0, y=0.0, yaw=0.0, velocity=10.0)
        traj = Trajectory(points=[target])
        
        action = controller.control(traj, state)
        
        # Error = 10 - 0 = 10
        # Accel = Kp * Error = 1.0 * 10 = 10.0
        assert abs(action.acceleration - 10.0) < 1e-10
