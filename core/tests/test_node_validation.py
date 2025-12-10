from typing import Any

import pytest
from pydantic import ValidationError

from core.interfaces.node import Node, NodeConfig


class SimpleConfig(NodeConfig):
    param: int


class SimpleNode(Node[SimpleConfig]):
    def __init__(self, config: dict[str, Any], rate_hz: float = 10.0):
        super().__init__("Simple", rate_hz, config, config_model=SimpleConfig)

    def on_run(self, _current_time: float) -> bool:
        return True


def test_node_strict_validation():
    # Valid config
    node = SimpleNode(config={"param": 10})
    assert node.config.param == 10

    # Extra param should fail
    with pytest.raises(ValidationError) as excinfo:
        SimpleNode(config={"param": 10, "extra_param": 100})

    # Check that error is about extra fields
    assert "Extra inputs are not permitted" in str(excinfo.value)
