"""Tests for planning components."""

import math

import pytest
from components.planning.pure_pursuit import PurePursuitPlanner
from core.data import Observation, Trajectory, TrajectoryPoint, VehicleState


class TestPurePursuitPlanner:
    """Tests for PurePursuitPlanner."""

    def test_plan_without_reference(self) -> None:
        """Test planning without reference trajectory."""
        planner = PurePursuitPlanner()
        state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0)
        traj = planner.plan(Observation(0, 0, 0, 0), state)
        assert len(traj) == 0

    def test_plan_straight_line(self) -> None:
        """Test planning on a straight line."""
        planner = PurePursuitPlanner(lookahead_distance=5.0)
        
        # Create straight reference trajectory
        points = [
            TrajectoryPoint(x=i, y=0.0, yaw=0.0, velocity=10.0)
            for i in range(20)
        ]
        planner.set_reference_trajectory(Trajectory(points=points))
        
        # Vehicle at start
        state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0)
        traj = planner.plan(Observation(0, 0, 0, 0), state)
        
        assert len(traj) == 1
        target = traj[0]
        # Should look ahead 5m
        assert abs(target.x - 5.0) < 0.5  # Approximation due to discrete points
        assert abs(target.y - 0.0) < 1e-10

    def test_plan_curve(self) -> None:
        """Test planning on a curve."""
        planner = PurePursuitPlanner(lookahead_distance=2.0)
        
        # Create 90 degree turn trajectory
        points = [
            TrajectoryPoint(x=0.0, y=i, yaw=math.pi/2, velocity=5.0)
            for i in range(10)
        ]
        planner.set_reference_trajectory(Trajectory(points=points))
        
        # Vehicle at (1, 0) facing North, target is on y-axis
        state = VehicleState(x=1.0, y=0.0, yaw=math.pi/2, velocity=5.0)
        traj = planner.plan(Observation(0, 0, 0, 0), state)
        
        assert len(traj) == 1
        target = traj[0]
        # Should find point on y-axis approx 2m away
        # Nearest point is (0,0), lookahead 2m -> (0, 2)
        assert abs(target.x - 0.0) < 1e-10
        assert abs(target.y - 2.0) < 0.5
