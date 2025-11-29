"""Tests for Simple 2D Simulator."""

import math

import numpy as np
import pytest
from core.data import Action, VehicleState
from simulators.simple_2d import Simple2DSimulator, VehicleDynamics


class TestVehicleDynamics:
    """Tests for VehicleDynamics."""

    def test_straight_line(self) -> None:
        """Test straight line motion."""
        dynamics = VehicleDynamics(wheelbase=2.5)
        state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=10.0)
        
        # Move straight for 1 second
        new_state = dynamics.step(state, steering=0.0, acceleration=0.0, dt=1.0)
        
        assert abs(new_state.x - 10.0) < 1e-10
        assert abs(new_state.y - 0.0) < 1e-10
        assert abs(new_state.yaw - 0.0) < 1e-10
        assert abs(new_state.velocity - 10.0) < 1e-10

    def test_acceleration(self) -> None:
        """Test acceleration."""
        dynamics = VehicleDynamics(wheelbase=2.5)
        state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0)
        
        # Accelerate at 2.0 m/s^2 for 1 second
        new_state = dynamics.step(state, steering=0.0, acceleration=2.0, dt=1.0)
        
        assert abs(new_state.velocity - 2.0) < 1e-10
        # Distance = 0.5 * a * t^2 = 0.5 * 2.0 * 1.0^2 = 1.0 (approx, Euler integration might differ slightly)
        # Euler: x_next = x + v * dt = 0 + 0 * 1 = 0
        # v_next = v + a * dt = 0 + 2 * 1 = 2
        # Note: Simple Euler integration uses velocity at t, not t+1 or average.
        assert abs(new_state.x - 0.0) < 1e-10 

    def test_turning(self) -> None:
        """Test turning motion."""
        dynamics = VehicleDynamics(wheelbase=2.5)
        state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=5.0)
        
        # Turn left with steering angle
        steering = math.atan(2.5 / 10.0) # Radius = L / tan(delta) = 10.0
        # yaw_rate = v / R = 5.0 / 10.0 = 0.5 rad/s
        
        new_state = dynamics.step(state, steering=steering, acceleration=0.0, dt=1.0)
        
        # Expected yaw change = 0.5 * 1.0 = 0.5 rad
        assert abs(new_state.yaw - 0.5) < 1e-10


class TestSimple2DSimulator:
    """Tests for Simple2DSimulator."""

    def test_initialization(self) -> None:
        """Test initialization."""
        sim = Simple2DSimulator()
        state = sim.reset()
        assert state.x == 0.0
        assert state.y == 0.0
        assert state.velocity == 0.0

    def test_step(self) -> None:
        """Test step execution."""
        sim = Simple2DSimulator(dt=0.1)
        sim.reset()
        
        action = Action(steering=0.0, acceleration=1.0)
        state, obs, done, info = sim.step(action)
        
        assert isinstance(state, VehicleState)
        assert state.velocity > 0.0
        assert not done
        assert isinstance(info, dict)

    def test_custom_initial_state(self) -> None:
        """Test with custom initial state."""
        initial_state = VehicleState(x=10.0, y=5.0, yaw=1.0, velocity=5.0)
        sim = Simple2DSimulator(initial_state=initial_state)
        state = sim.reset()
        
        assert state.x == 10.0
        assert state.y == 5.0
        assert state.yaw == 1.0
        assert state.velocity == 5.0
