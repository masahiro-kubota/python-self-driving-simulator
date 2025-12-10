import pytest
from pydantic import ValidationError

from supervisor.supervisor_node import SupervisorNode


def test_supervisor_validation_success():
    """Test successful initialization with valid parameters."""
    # Complete goal params
    config = {"goal_x": 10.0, "goal_y": 20.0, "goal_radius": 5.0, "max_steps": 100}
    node = SupervisorNode(config=config, rate_hz=10.0)
    assert node.config.goal_x == 10.0
    assert node.config.goal_y == 20.0
    assert node.config.max_steps == 100

    # Defaults (partial config)
    # Since fields have defaults in Pydantic model (0.0, 1000), empty config should be valid?
    # Yes, we set defaults in SupervisorConfig: goal_x=0.0 etc.
    config = {}
    node = SupervisorNode(config=config, rate_hz=10.0)
    assert node.config.goal_x == 0.0
    assert node.config.max_steps == 1000


def test_supervisor_validation_failure():
    """Test initialization failure with invalid parameters."""
    # Explicit None should fail
    with pytest.raises(ValidationError) as excinfo:
        SupervisorNode(config={"goal_x": None}, rate_hz=10.0)
    assert "Input should be a valid number" in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
        SupervisorNode(config={"max_steps": None}, rate_hz=10.0)
    assert "Input should be a valid integer" in str(excinfo.value)
