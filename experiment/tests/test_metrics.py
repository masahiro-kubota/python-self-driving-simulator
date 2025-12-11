from core.data import SimulationLog
from experiment.postprocessing.metrics import MetricsCalculator


def test_collision_count_metric():
    calculator = MetricsCalculator()
    log = SimulationLog(steps=[], metadata={})

    # Test with reason="collision"
    metrics = calculator.calculate(log, reason="collision")
    assert metrics.collision_count == 1
    assert metrics.termination_code == 5

    # Test with other reason
    metrics = calculator.calculate(log, reason="goal_reached")
    assert metrics.collision_count == 0
    assert metrics.termination_code == 1

    # Test with unknown reason
    metrics = calculator.calculate(log, reason="unknown")
    assert metrics.collision_count == 0
    assert metrics.termination_code == 0
