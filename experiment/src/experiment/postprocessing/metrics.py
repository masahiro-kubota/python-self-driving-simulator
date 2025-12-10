import numpy as np

from core.data import SimulationLog
from core.data.experiment import EvaluationMetrics


class MetricsCalculator:
    """Calculate metrics from simulation log."""

    def __init__(self, wheelbase: float = 2.5) -> None:
        """Initialize metrics calculator.

        Args:
            wheelbase: Vehicle wheelbase [m]
        """
        self.wheelbase = wheelbase

    def calculate(self, log: SimulationLog) -> EvaluationMetrics:
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
        avg_lateral_accel = (
            float(np.mean(np.abs(lateral_accels))) if len(lateral_accels) > 0 else 0.0
        )
        max_lateral_accel = (
            float(np.max(np.abs(lateral_accels))) if len(lateral_accels) > 0 else 0.0
        )

        # Comfort score (inverse of jerk)
        comfort_score = self._calculate_comfort_score(log)

        # Velocity
        velocities = [s.vehicle_state.velocity for s in log.steps]
        avg_velocity = float(np.mean(velocities)) if velocities else 0.0

        # Success (Placeholder, to be set by caller)
        success = 0

        # Collision & lane departure (collision is placeholder)
        collision_count = 0
        lane_departure_rate = self._calculate_lane_departure_rate(log)

        return EvaluationMetrics(
            lap_time_sec=lap_time_sec,
            collision_count=collision_count,
            lane_departure_rate=lane_departure_rate,
            avg_lateral_accel=avg_lateral_accel,
            max_lateral_accel=max_lateral_accel,
            comfort_score=comfort_score,
            success=success,
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
        return max(0.0, 1.0 - avg_jerk / 10.0)  # Assuming jerk < 10 is comfortable

    def _calculate_lane_departure_rate(self, log: SimulationLog) -> float:
        """Calculate lane departure rate (fraction of time off-track)."""
        if not log.steps:
            return 0.0

        off_track_count = sum(
            1 for step in log.steps if getattr(step.vehicle_state, "off_track", False)
        )
        return float(off_track_count) / len(log.steps)
