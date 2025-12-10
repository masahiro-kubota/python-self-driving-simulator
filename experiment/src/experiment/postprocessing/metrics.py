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

    def calculate(self, log: SimulationLog, reason: str = "unknown") -> EvaluationMetrics:
        """Calculate all metrics from log.

        Args:
            log: Simulation log
            reason: Termination reason string

        Returns:
            Calculated metrics
        """
        # Lap time
        lap_time_sec = log.steps[-1].timestamp if log.steps else 0.0

        # Success (Placeholder, to be set by caller)
        success = 0

        # Collision (placeholder)
        collision_count = 0

        # Get goal_count from last step info if available
        goal_count = 0
        if log.steps:
            goal_count = log.steps[-1].info.get("goal_count", 0)

        # Termination Code
        # 0: unknown, 1: goal_reached, 2: off_track, 3: timeout
        reason_map = {
            "unknown": 0,
            "goal_reached": 1,
            "off_track": 2,
            "timeout": 3,
            "simulator_completed": 4,
        }
        termination_code = reason_map.get(reason, 0)

        return EvaluationMetrics(
            lap_time_sec=lap_time_sec,
            collision_count=collision_count,
            success=success,
            termination_code=termination_code,
            goal_count=goal_count,
        )
