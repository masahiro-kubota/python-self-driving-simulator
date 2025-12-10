"""Tests for SingleProcessExecutor."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from core.clock import SteppedClock
from core.data import Action, SimulationLog, Trajectory, TrajectoryPoint, VehicleState
from core.data.frame_data import create_frame_data_type
from core.data.node_io import NodeIO
from core.executor import SingleProcessExecutor
from core.interfaces import Simulator
from core.interfaces.node import Node
from core.nodes import PhysicsNode


@pytest.fixture
def mock_simulator():
    sim = MagicMock(spec=Simulator)
    state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0, timestamp=0.0)
    sim.step.return_value = (state, False, {})
    # Mock log for log injection
    sim.log = MagicMock(spec=SimulationLog)
    sim.log.steps = []

    # Add attributes for goal check
    sim.goal_x = 100.0
    sim.goal_y = 0.0

    return sim


@pytest.fixture
def mock_planner():
    planner = MagicMock()
    planner.process.return_value = Trajectory(
        points=[TrajectoryPoint(x=1.0, y=1.0, yaw=0.0, velocity=1.0)]
    )
    return planner


@pytest.fixture
def mock_controller():
    controller = MagicMock()
    controller.process.return_value = Action(acceleration=1.0, steering=0.1)
    return controller


@pytest.fixture
def stepped_clock():
    from core.clock import SteppedClock

    return SteppedClock(start_time=0.0, dt=0.01)


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.simulator.rate_hz = 10.0
    config.execution.goal_radius = 5.0
    return config


def _create_context_for_nodes(nodes) -> Any:
    from core.data.frame_data import collect_node_output_fields

    fields = collect_node_output_fields(nodes)

    DynamicFrameData = create_frame_data_type(fields)  # noqa: N806
    return DynamicFrameData()


class MockNode(Node):
    """Mock node for testing executor."""

    def __init__(
        self, name: str, rate_hz: float, inputs: list[str], outputs: list[str], process_func=None
    ):
        super().__init__(name, rate_hz)
        self.io = NodeIO(inputs={k: Any for k in inputs}, outputs={k: Any for k in outputs})
        self.process_func = process_func

    def get_node_io(self) -> NodeIO:
        return self.io

    def on_run(self, _current_time: float) -> bool:
        if self.frame_data is None:
            return False

        # Collect inputs
        inputs = {}
        for k in self.io.inputs:
            val = getattr(self.frame_data, k, None)
            if val is None:
                return False
            inputs[k] = val

        # Run process
        if self.process_func:
            output = self.process_func(**inputs)
            # Write outputs
            if isinstance(output, dict):
                for k, v in output.items():
                    if k in self.io.outputs:
                        setattr(self.frame_data, k, v)
            elif len(self.io.outputs) == 1:
                k = next(iter(self.io.outputs))
                setattr(self.frame_data, k, output)

        return True


def test_executor_timing(mock_simulator, mock_planner, mock_controller):
    """Test that nodes run at expected rates."""

    clock = SteppedClock(start_time=0.0, dt=0.01)

    # Physics Node
    physics_node = PhysicsNode(mock_simulator, rate_hz=10.0)
    physics_node.on_run = MagicMock(wraps=physics_node.on_run)

    # Sensor Node (Mock)
    sensor_node = MockNode(
        "Sensor",
        rate_hz=10.0,
        inputs=["sim_state"],
        outputs=["vehicle_state"],
        process_func=lambda sim_state: sim_state,
    )  # Pass through
    sensor_node.on_run = MagicMock(wraps=sensor_node.on_run)

    # Planning Node (Mock)
    planning_node = MockNode(
        "Planning",
        rate_hz=5.0,
        inputs=["vehicle_state"],
        outputs=["trajectory"],
        process_func=lambda vehicle_state: mock_planner.process(vehicle_state),
    )  # Adapt mock
    planning_node.on_run = MagicMock(wraps=planning_node.on_run)

    # Control Node (Mock)
    control_node = MockNode(
        "Control",
        rate_hz=10.0,
        inputs=["trajectory", "vehicle_state"],
        outputs=["action"],
        process_func=lambda trajectory, vehicle_state: mock_controller.process(
            trajectory, vehicle_state
        ),
    )
    control_node.on_run = MagicMock(wraps=control_node.on_run)

    nodes = [physics_node, sensor_node, planning_node, control_node]

    frame_data = _create_context_for_nodes(nodes)

    for node in nodes:
        node.set_frame_data(frame_data)
    executor = SingleProcessExecutor(nodes, clock)

    # Run for 0.42 seconds
    executor.run(duration=0.42)

    assert physics_node.on_run.call_count == 5
    assert planning_node.on_run.call_count == 3
    assert control_node.on_run.call_count == 5
    assert sensor_node.on_run.call_count == 5


def test_executor_data_flow(mock_simulator, mock_planner, mock_controller):
    """Test data flow between nodes via FrameData."""
    clock = SteppedClock(start_time=0.0, dt=0.01)

    # Physics
    physics_node = PhysicsNode(mock_simulator, rate_hz=10.0)

    # Sensor
    sensor_node = MockNode(
        "Sensor",
        rate_hz=10.0,
        inputs=["sim_state"],
        outputs=["vehicle_state"],
        process_func=lambda sim_state: sim_state,
    )

    # Planning
    planning_node = MockNode(
        "Planning",
        rate_hz=10.0,
        inputs=["vehicle_state"],
        outputs=["trajectory"],
        process_func=mock_planner.process,
    )

    # Control
    control_node = MockNode(
        "Control",
        rate_hz=30.0,
        inputs=["trajectory", "vehicle_state"],
        outputs=["action"],
        process_func=mock_controller.process,
    )

    nodes = [physics_node, sensor_node, planning_node, control_node]

    frame_data = _create_context_for_nodes(nodes)

    for node in nodes:
        node.set_frame_data(frame_data)
    executor = SingleProcessExecutor(nodes, clock)

    # Run one step
    executor.run(duration=0.05)

    # Physics should have updated sim_state
    mock_simulator.step.assert_called()
    assert frame_data.sim_state is not None

    # Sensor should have updated vehicle_state
    assert frame_data.vehicle_state is not None

    # Planning should have produced trajectory
    mock_planner.process.assert_called()
    assert frame_data.trajectory is not None

    # Control should have produced action
    mock_controller.process.assert_called()
    assert frame_data.action is not None
