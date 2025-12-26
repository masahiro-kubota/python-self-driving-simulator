import pytest
from core.data import ComponentConfig
from core.data.node_io import NodeIO
from core.interfaces.node import Node, NodeExecutionResult
from pydantic import ValidationError


class SimpleConfig(ComponentConfig):
    param: int


class SimpleNode(Node[SimpleConfig]):
    def __init__(self, config: SimpleConfig, rate_hz: float = 10.0):
        super().__init__("Simple", rate_hz, config)

    def get_node_io(self) -> NodeIO:
        return NodeIO(inputs={}, outputs={})

    def on_run(self, _current_time: float) -> NodeExecutionResult:
        return NodeExecutionResult.SUCCESS


def test_node_strict_validation():
    # Valid config
    config = SimpleConfig(param=10)
    node = SimpleNode(config=config)
    assert node.config.param == 10

    # Extra param should fail
    with pytest.raises(ValidationError) as excinfo:
        SimpleConfig(param=10, extra_param=100)

    # Check that error is about extra fields
    assert "Extra inputs are not permitted" in str(excinfo.value)
