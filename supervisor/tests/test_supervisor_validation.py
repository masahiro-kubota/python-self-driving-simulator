import pytest
from pydantic import ValidationError
from supervisor.supervisor_node import SupervisorNode


def test_supervisor_nested_config():
    """Test successful initialization with nested configuration."""
    config = {
        "goal": {"x": 10.0, "y": 20.0, "radius": 5.0, "enabled": True, "min_elapsed_time": 10.0},
        "off_track": {"enabled": True},
    }
    node = SupervisorNode(config=config, rate_hz=10.0)
    assert node.config.goal.x == 10.0
    assert node.config.goal.y == 20.0
    assert node.config.goal.radius == 5.0
    assert node.config.goal.enabled is True
    assert node.config.goal.min_elapsed_time == 10.0
    assert node.config.off_track.enabled is True


def test_supervisor_validation_failure():
    """Test initialization failure with invalid parameters."""
    # Missing required goal parameters
    with pytest.raises(ValidationError) as excinfo:
        SupervisorNode(config={}, rate_hz=10.0)
    assert "goal" in str(excinfo.value).lower()

    # Invalid type for nested config
    with pytest.raises(ValidationError) as excinfo:
        SupervisorNode(config={"goal": {"x": "invalid"}}, rate_hz=10.0)
    assert "Input should be a valid number" in str(excinfo.value)
