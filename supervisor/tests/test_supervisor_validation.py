import pytest
from pydantic import ValidationError

from supervisor.supervisor_node import SupervisorNode


def test_supervisor_validation_success():
    """Test successful initialization with valid parameters."""
    # Complete goal params
    config = {"goal_x": 10.0, "goal_y": 20.0, "goal_radius": 5.0}
    node = SupervisorNode(config=config, rate_hz=10.0)
    assert node.goal_x == 10.0
    assert node.goal_y == 20.0

    # No goal params
    config = {"goal_x": None, "goal_y": None}
    node = SupervisorNode(config=config, rate_hz=10.0)
    assert node.goal_x is None


def test_supervisor_validation_failure():
    """Test initialization failure with invalid parameters."""
    # Missing goal_y
    with pytest.raises(ValidationError) as excinfo:
        SupervisorNode(config={"goal_x": 10.0, "goal_y": None}, rate_hz=10.0)
    assert "Both goal_x and goal_y must be provided" in str(excinfo.value)

    # Missing goal_x
    with pytest.raises(ValidationError) as excinfo:
        SupervisorNode(config={"goal_x": None, "goal_y": 20.0}, rate_hz=10.0)
    assert "Both goal_x and goal_y must be provided" in str(excinfo.value)
