"""Test priority-based node execution ordering."""

from core.clock.stepped import SteppedClock
from core.data import ComponentConfig, NodeExecutionResult
from core.executor.single_process import SingleProcessExecutor
from core.interfaces.node import Node


class MockConfig(ComponentConfig):
    """Mock configuration for testing."""

    pass


class MockNode(Node[MockConfig]):
    """Mock node for testing."""

    def __init__(self, name: str, rate_hz: float, priority: int = 100):
        config = MockConfig()
        super().__init__(name=name, rate_hz=rate_hz, config=config, priority=priority)
        self.execution_order = []

    def get_node_io(self):
        from core.data.node_io import NodeIO

        return NodeIO(inputs=[], outputs=[])

    def on_run(self, _current_time: float) -> NodeExecutionResult:
        return NodeExecutionResult.SUCCESS


def test_nodes_sorted_by_priority():
    """Test that nodes are sorted by priority in ascending order."""
    # Create nodes with different priorities
    node_high = MockNode(name="HighPriority", rate_hz=10.0, priority=1)
    node_medium = MockNode(name="MediumPriority", rate_hz=10.0, priority=50)
    node_low = MockNode(name="LowPriority", rate_hz=10.0, priority=99)

    # Create executor with nodes in random order
    nodes = [node_medium, node_low, node_high]
    clock = SteppedClock(start_time=0.0, dt=0.1)
    executor = SingleProcessExecutor(nodes, clock)

    # Verify nodes are sorted by priority
    assert executor.nodes[0].name == "HighPriority"
    assert executor.nodes[0].priority == 1
    assert executor.nodes[1].name == "MediumPriority"
    assert executor.nodes[1].priority == 50
    assert executor.nodes[2].name == "LowPriority"
    assert executor.nodes[2].priority == 99


def test_nodes_with_same_priority_maintain_order():
    """Test that nodes with same priority maintain their original order (stable sort)."""
    # Create nodes with same priority
    node_a = MockNode(name="NodeA", rate_hz=10.0, priority=50)
    node_b = MockNode(name="NodeB", rate_hz=10.0, priority=50)
    node_c = MockNode(name="NodeC", rate_hz=10.0, priority=50)

    # Create executor with specific order
    nodes = [node_a, node_b, node_c]
    clock = SteppedClock(start_time=0.0, dt=0.1)
    executor = SingleProcessExecutor(nodes, clock)

    # Verify order is maintained
    assert executor.nodes[0].name == "NodeA"
    assert executor.nodes[1].name == "NodeB"
    assert executor.nodes[2].name == "NodeC"


def test_mixed_priorities():
    """Test sorting with mixed priority values."""
    # Create nodes with various priorities
    nodes = [
        MockNode(name="Node1", rate_hz=10.0, priority=100),  # default
        MockNode(name="Node2", rate_hz=10.0, priority=1),  # highest
        MockNode(name="Node3", rate_hz=10.0, priority=50),  # medium
        MockNode(name="Node4", rate_hz=10.0, priority=99),  # low
        MockNode(name="Node5", rate_hz=10.0, priority=10),  # high
    ]

    clock = SteppedClock(start_time=0.0, dt=0.1)
    executor = SingleProcessExecutor(nodes, clock)

    # Verify correct order
    expected_order = ["Node2", "Node5", "Node3", "Node4", "Node1"]
    actual_order = [node.name for node in executor.nodes]
    assert actual_order == expected_order

    # Verify priorities
    expected_priorities = [1, 10, 50, 99, 100]
    actual_priorities = [node.priority for node in executor.nodes]
    assert actual_priorities == expected_priorities
