from pathlib import Path

import pytest
from core.data import VehicleState
from core.data.ad_components.sensing import Sensing
from core.data.ros import MarkerArray
from core.interfaces.node import NodeExecutionResult
from planning_utils.types import ReferencePath, ReferencePathPoint
from static_avoidance_planner.avoidance_planner_node import (
    AvoidancePlannerNode,
    AvoidancePlannerNodeConfig,
)


# Mock load_track_csv to avoid file I/O
@pytest.fixture
def mock_loader(monkeypatch):
    def mock_load(_):
        # Return a simple trajectory
        p1 = ReferencePathPoint(x=0.0, y=0.0, yaw=0.0, velocity=10.0)
        p2 = ReferencePathPoint(x=10.0, y=0.0, yaw=0.0, velocity=10.0)
        return ReferencePath(points=[p1, p2])

    monkeypatch.setattr("static_avoidance_planner.avoidance_planner_node.load_track_csv", mock_load)


@pytest.fixture
def node(_):
    config = AvoidancePlannerNodeConfig(track_path=Path("dummy.csv"))
    # We need to mock get_project_root if path is relative, or use absolute.
    # The node checks absolute.
    # We used "dummy.csv" relative. It calls get_project_root.
    # Let's mock get_project_root too or use absolute path.
    config.track_path = Path("/tmp/dummy.csv")

    return AvoidancePlannerNode(config, rate_hz=10.0)


def test_node_instantiation(node):
    assert node.config.lookahead_distance == 30.0


def test_node_execution_skipped(node):
    # No inputs
    result = node.on_run(0.0)
    # The Node base class sets frame_data.
    # If frame_data is None?
    # We should setup frame_data.
    pass  # Needs Node test harness.

    # Manually setup frame_data
    from core.data.frame_data import FrameData

    node.frame_data = FrameData()

    # Missing sensing -> Skipped
    result = node.on_run(0.0)
    assert result == NodeExecutionResult.SKIPPED


def test_node_execution_success(node):
    from core.data.frame_data import FrameData

    node.frame_data = FrameData()

    # Add input
    sensing = Sensing(vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0), obstacles=[])
    node.frame_data.sensing = sensing

    result = node.on_run(0.1)
    assert result == NodeExecutionResult.SUCCESS

    # Check output
    from core.data.autoware import Trajectory

    assert isinstance(node.frame_data.trajectory, Trajectory)
    assert isinstance(getattr(node.frame_data, "planning/marker"), MarkerArray)
