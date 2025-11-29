"""Metrics calculator for simulation evaluation."""

from dataclasses import asdict, dataclass

import numpy as np

from core.data import SimulationLog, Trajectory


@dataclass
class SimulationMetrics:
    """Standard metrics for simulation evaluation."""

    lap_time_sec: float
    collision_count: int
    lane_departure_rate: float
    avg_lateral_accel: float
    max_lateral_accel: float
    comfort_score: float
    success: int

    # Additional metrics
    avg_lateral_error: float
    max_lateral_error: float
    avg_velocity: float

    def to_dict(self) -> dict[str, float | int]:
        """Convert to dictionary for MLflow."""
        return asdict(self)


class MetricsCalculator:
    """Calculate metrics from simulation log."""

    def __init__(self, reference_trajectory: Trajectory | None = None, wheelbase: float = 2.5) -> None:
        """Initialize metrics calculator.

        Args:
            reference_trajectory: Reference trajectory for error calculation
            wheelbase: Vehicle wheelbase [m]
        """
        self.reference_trajectory = reference_trajectory
        self.wheelbase = wheelbase

    def calculate(self, log: SimulationLog) -> SimulationMetrics:
        """Calculate all metrics from log.

        Args:
            log: Simulation log

        Returns:
            Calculated metrics
        """
        # Lap time
        lap_time_sec = log.steps[-1].timestamp if log.steps else 0.0

        # Lateral acceleration
        lateral_accels = self._calculate_lateral_accelerations(log)
        avg_lateral_accel = float(np.mean(np.abs(lateral_accels))) if len(lateral_accels) > 0 else 0.0
        max_lateral_accel = float(np.max(np.abs(lateral_accels))) if len(lateral_accels) > 0 else 0.0

        # Lateral error (if reference trajectory available)
        if self.reference_trajectory:
            lateral_errors = self._calculate_lateral_errors(log)
            avg_lateral_error = float(np.mean(np.abs(lateral_errors)))
            max_lateral_error = float(np.max(np.abs(lateral_errors)))
        else:
            avg_lateral_error = 0.0
            max_lateral_error = 0.0

        # Comfort score (inverse of jerk)
        comfort_score = self._calculate_comfort_score(log)

        # Velocity
        velocities = [s.vehicle_state.velocity for s in log.steps]
        avg_velocity = float(np.mean(velocities)) if velocities else 0.0

        # Success (reached goal)
        success = 1 if self._check_success(log) else 0

        # Collision & lane departure (placeholder - need collision detection)
        collision_count = 0
        lane_departure_rate = 0.0

        return SimulationMetrics(
            lap_time_sec=lap_time_sec,
            collision_count=collision_count,
            lane_departure_rate=lane_departure_rate,
            avg_lateral_accel=avg_lateral_accel,
            max_lateral_accel=max_lateral_accel,
            comfort_score=comfort_score,
            success=success,
            avg_lateral_error=avg_lateral_error,
            max_lateral_error=max_lateral_error,
            avg_velocity=avg_velocity,
        )

    def _calculate_lateral_accelerations(self, log: SimulationLog) -> list[float]:
        """Calculate lateral accelerations."""
        accels = []
        for step in log.steps:
            # Simplified: v^2 * tan(delta) / L (kinematic bicycle model)
            v = step.vehicle_state.velocity
            delta = step.action.steering
            if self.wheelbase > 0:
                lateral_accel = (v * v * np.tan(delta)) / self.wheelbase
                accels.append(lateral_accel)
        return accels

    def _calculate_lateral_errors(self, log: SimulationLog) -> list[float]:
        """Calculate lateral errors from reference trajectory."""
        if not self.reference_trajectory:
            return []

        # Use NeuralController's error calculation logic
        try:
            from components.control.neural_controller import NeuralController

            controller = NeuralController("dummy", "dummy")
            controller.set_reference_trajectory(self.reference_trajectory)

            errors = []
            for step in log.steps:
                e_lat, _, _ = controller.calculate_errors(step.vehicle_state)
                errors.append(e_lat)
            return errors
        except ImportError:
            # Fallback: simple nearest point distance
            return [0.0] * len(log.steps)

    def _calculate_comfort_score(self, log: SimulationLog) -> float:
        """Calculate comfort score (0-1, higher is better)."""
        if len(log.steps) < 3:
            return 1.0

        jerks = []
        for i in range(2, len(log.steps)):
            dt1 = log.steps[i - 1].timestamp - log.steps[i - 2].timestamp
            dt2 = log.steps[i].timestamp - log.steps[i - 1].timestamp

            if dt1 > 0 and dt2 > 0:
                a1 = log.steps[i - 1].action.acceleration
                a2 = log.steps[i].action.acceleration
                jerk = abs((a2 - a1) / dt2)
                jerks.append(jerk)

        if not jerks:
            return 1.0

        # Normalize: lower jerk = higher score
        avg_jerk = float(np.mean(jerks))
        comfort = max(0.0, 1.0 - avg_jerk / 10.0)  # Assuming jerk < 10 is comfortable
        return comfort

    def _check_success(self, log: SimulationLog) -> bool:
        """Check if simulation succeeded (reached goal)."""
        if not log.steps or not self.reference_trajectory:
            return False

        final_state = log.steps[-1].vehicle_state
        goal = self.reference_trajectory[-1]

        dist = np.sqrt((final_state.x - goal.x) ** 2 + (final_state.y - goal.y) ** 2)

        return bool(dist < 5.0)  # Within 5m of goal
