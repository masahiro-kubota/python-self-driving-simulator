"""Dummy data generator for dashboard testing.

This module provides utilities to generate test simulation data.
"""

import math
from datetime import datetime

from core.data import SimulationLog, SimulationStep, VehicleState
from core.data.autoware import AckermannControlCommand, AckermannLateralCommand, LongitudinalCommand
from core.data.experiment import ExperimentResult
from core.data.simulator import SimulationResult


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
        steps=[],
        metadata={
            "controller": "Test Controller (Circular Trajectory)",
            "track": "Test Track",
            "num_steps": num_steps,
            "radius": radius,
            "width": 1.8,
            "length": 4.5,
            "wheelbase": 2.5,
            "front_overhang": 0.9,
            "rear_overhang": 1.1,
        },
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

        action = AckermannControlCommand(
            lateral=AckermannLateralCommand(steering_tire_angle=0.2 * math.cos(angle)),
            longitudinal=LongitudinalCommand(acceleration=0.5 * math.sin(angle * 2)),
        )

        step = SimulationStep(
            timestamp=t,
            vehicle_state=vehicle_state,
            action=action,
        )
        log.steps.append(step)

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
        steps=[],
        metadata={
            "controller": "Test Controller (Figure-Eight Trajectory)",
            "track": "Test Track",
            "num_steps": num_steps,
            "radius": radius,
            "width": 1.8,
            "length": 4.5,
            "wheelbase": 2.5,
            "front_overhang": 0.9,
            "rear_overhang": 1.1,
        },
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

        action = AckermannControlCommand(
            lateral=AckermannLateralCommand(steering_tire_angle=0.3 * math.sin(angle)),
            longitudinal=LongitudinalCommand(acceleration=0.3 * math.cos(angle)),
        )

        step = SimulationStep(
            timestamp=t,
            vehicle_state=vehicle_state,
            action=action,
        )
        log.steps.append(step)

    return log


def create_experiment_result_from_log(
    log: SimulationLog,
    experiment_name: str = "Test Experiment",
    experiment_type: str = "evaluation",
) -> ExperimentResult:
    """Create an ExperimentResult from a SimulationLog for testing.

    Args:
        log: Simulation log to wrap
        experiment_name: Name of the experiment
        experiment_type: Type of the experiment

    Returns:
        ExperimentResult containing the simulation log
    """
    # Get final state from last step
    final_state = (
        log.steps[-1].vehicle_state
        if log.steps
        else VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0)
    )

    sim_result = SimulationResult(
        log=log,
        success=True,
        reason="Test completed successfully",
        final_state=final_state,
    )

    return ExperimentResult(
        experiment_name=experiment_name,
        experiment_type=experiment_type,
        execution_time=datetime.now(),
        simulation_results=[sim_result],
    )


__all__ = [
    "create_experiment_result_from_log",
    "generate_circular_trajectory",
    "generate_figure_eight_trajectory",
]
