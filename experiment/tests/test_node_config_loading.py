import pytest
from core.data import ComponentConfig
from core.data.node_io import NodeIO
from core.interfaces.node import Node, NodeExecutionResult
from pydantic import ValidationError


class MockNodeConfig(ComponentConfig):
    """Configuration for MockNode."""

    param_int: int
    param_str: str = "default"
    param_list: list[int]


class MockNode(Node[MockNodeConfig]):
    """Test node for verification."""

    def __init__(self, config: MockNodeConfig, rate_hz: float = 10.0):
        super().__init__("MockNode", rate_hz, config)

    def get_node_io(self) -> NodeIO:
        return NodeIO(inputs={}, outputs={})

    def on_run(self, _current_time: float) -> NodeExecutionResult:
        return NodeExecutionResult.SUCCESS


def test_node_creation_success():
    """Test successful node creation with valid configuration."""
    config_dict = {
        "param_int": 42,
        "param_str": "test",
        "param_list": [1, 2, 3],
    }

    node = MockNode.from_dict(
        rate_hz=10.0,
        config_class=MockNodeConfig,
        config_dict=config_dict,
    )

    assert node.config.param_int == 42
    assert node.config.param_str == "test"
    assert node.config.param_list == [1, 2, 3]


def test_node_default_parameter():
    """Test node creation using default parameter value."""
    config_dict = {
        "param_int": 42,
        "param_list": [1, 2, 3],
        # param_str omitted, should use default
    }

    node = MockNode.from_dict(
        rate_hz=10.0,
        config_class=MockNodeConfig,
        config_dict=config_dict,
    )

    assert node.config.param_int == 42
    assert node.config.param_str == "default"
    assert node.config.param_list == [1, 2, 3]


def test_node_missing_parameter():
    """Test validation error for missing required parameter."""
    config_dict = {
        # param_int missing
        "param_str": "test",
        "param_list": [1, 2, 3],
    }

    with pytest.raises(ValidationError) as excinfo:
        MockNode.from_dict(
            rate_hz=10.0,
            config_class=MockNodeConfig,
            config_dict=config_dict,
        )

    assert "param_int" in str(excinfo.value)
    assert "Field required" in str(excinfo.value)


def test_node_extra_parameter():
    """Test validation error for extra unknown parameter."""
    config_dict = {
        "param_int": 42,
        "param_str": "test",
        "param_list": [1, 2, 3],
        "unknown_param": "should fail",
    }

    with pytest.raises(ValidationError) as excinfo:
        MockNode.from_dict(
            rate_hz=10.0,
            config_class=MockNodeConfig,
            config_dict=config_dict,
        )

    # Note: Exact error message depends on pydantic version, usually "Extra inputs are not permitted"
    assert "unknown_param" in str(excinfo.value)
    assert "Extra inputs are not permitted" in str(excinfo.value)
