"""Dummy data generator for dashboard testing.

This module provides utilities to generate test simulation data.
"""

import math

from core.data import Action, SimulationLog, SimulationStep, VehicleState


def generate_circular_trajectory(
    num_steps: int = 100,
    radius: float = 50.0,
    duration: float = 10.0,
) -> SimulationLog:
    """Generate a circular trajectory for testing.

    Args:
        num_steps: Number of simulation steps
        radius: Radius of the circular trajectory in meters
        duration: Total duration of the simulation in seconds

    Returns:
        SimulationLog with circular trajectory
    """
    log = SimulationLog(
        metadata={
            "controller": "Test Controller (Circular Trajectory)",
            "track": "Test Track",
            "num_steps": num_steps,
            "radius": radius,
        }
    )

    for i in range(num_steps):
        t = (i / num_steps) * duration
        angle = (i / num_steps) * 2 * math.pi

        vehicle_state = VehicleState(
            x=radius * math.cos(angle),
            y=radius * math.sin(angle),
            yaw=angle + math.pi / 2,
            velocity=10.0 + 2 * math.sin(angle * 3),
        )

        action = Action(
            acceleration=0.5 * math.sin(angle * 2),
            steering=0.2 * math.cos(angle),
        )

        step = SimulationStep(
            timestamp=t,
            vehicle_state=vehicle_state,
            action=action,
        )
        log.add_step(step)

    return log


def generate_figure_eight_trajectory(
    num_steps: int = 200,
    radius: float = 30.0,
    duration: float = 20.0,
) -> SimulationLog:
    """Generate a figure-eight trajectory for testing.

    Args:
        num_steps: Number of simulation steps
        radius: Radius of each loop in meters
        duration: Total duration of the simulation in seconds

    Returns:
        SimulationLog with figure-eight trajectory
    """
    log = SimulationLog(
        metadata={
            "controller": "Test Controller (Figure-Eight Trajectory)",
            "track": "Test Track",
            "num_steps": num_steps,
            "radius": radius,
        }
    )

    for i in range(num_steps):
        t = (i / num_steps) * duration
        angle = (i / num_steps) * 4 * math.pi  # Two full circles

        # Figure-eight parametric equations
        x = radius * math.sin(angle)
        y = radius * math.sin(angle) * math.cos(angle)
        yaw = math.atan2(
            radius * math.cos(angle) * math.cos(angle) - radius * math.sin(angle) * math.sin(angle),
            radius * math.cos(angle),
        )

        vehicle_state = VehicleState(
            x=x,
            y=y,
            yaw=yaw,
            velocity=8.0 + 3 * math.cos(angle * 2),
        )

        action = Action(
            acceleration=0.3 * math.cos(angle),
            steering=0.3 * math.sin(angle),
        )

        step = SimulationStep(
            timestamp=t,
            vehicle_state=vehicle_state,
            action=action,
        )
        log.add_step(step)

    return log


__all__ = ["generate_circular_trajectory", "generate_figure_eight_trajectory"]
