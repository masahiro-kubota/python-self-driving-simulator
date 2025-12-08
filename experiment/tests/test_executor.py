"""Tests for SingleProcessExecutor."""

from unittest.mock import MagicMock

import pytest

from core.data import Action, Observation, SimulationLog, Trajectory, TrajectoryPoint, VehicleState
from core.data.node_io import NodeIO
from core.data.simulation_context import SimulationContext
from core.executor import SingleProcessExecutor
from core.interfaces import Simulator
from core.nodes import GenericProcessingNode, PhysicsNode
from core.processors.perception import BasicPerceptionProcessor
from core.processors.sensor import IdealSensorProcessor


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


def test_executor_timing(mock_simulator, mock_planner, mock_controller):
    """Test that nodes run at expected rates."""

    context = SimulationContext()

    # Physics Node
    physics_node = PhysicsNode(mock_simulator, rate_hz=10.0)
    physics_node.on_run = MagicMock(wraps=physics_node.on_run)

    # Sensor Node (Generic)
    sensor_processor = IdealSensorProcessor()
    sensor_io = NodeIO(inputs=["sim_state"], output="vehicle_state")
    sensor_node = GenericProcessingNode("Sensor", sensor_processor, sensor_io, rate_hz=10.0)
    sensor_node.on_run = MagicMock(wraps=sensor_node.on_run)

    # Planning Node (Generic)
    planner_io = NodeIO(inputs=["vehicle_state", "observation"], output="trajectory")
    planning_node = GenericProcessingNode("Planning", mock_planner, planner_io, rate_hz=5.0)
    planning_node.on_run = MagicMock(wraps=planning_node.on_run)

    # Control Node (Generic)
    controller_io = NodeIO(inputs=["trajectory", "vehicle_state", "observation"], output="action")
    control_node = GenericProcessingNode("Control", mock_controller, controller_io, rate_hz=10.0)
    control_node.on_run = MagicMock(wraps=control_node.on_run)

    nodes = [physics_node, sensor_node, planning_node, control_node]
    executor = SingleProcessExecutor(nodes, context)

    # Run for 0.42 seconds
    executor.run(duration=0.42, dt=0.01)

    assert physics_node.on_run.call_count == 5
    assert planning_node.on_run.call_count == 3
    assert control_node.on_run.call_count == 5
    assert sensor_node.on_run.call_count == 5


def test_executor_data_flow(mock_simulator, mock_planner, mock_controller):
    """Test data flow between nodes via Context."""
    context = SimulationContext()

    # Physics
    physics_node = PhysicsNode(mock_simulator, rate_hz=10.0)

    # Sensor
    sensor_node = GenericProcessingNode(
        "Sensor",
        IdealSensorProcessor(),
        NodeIO(inputs=["sim_state"], output="vehicle_state"),
        rate_hz=10.0,
    )

    # Perception
    # Need perception to produce "observation" for planner/controller
    perception_node = GenericProcessingNode(
        "Perception",
        BasicPerceptionProcessor(),
        NodeIO(inputs=["vehicle_state"], output="observation"),
        rate_hz=10.0,
    )

    # Planning
    planning_node = GenericProcessingNode(
        "Planning",
        mock_planner,
        NodeIO(inputs=["vehicle_state", "observation"], output="trajectory"),
        rate_hz=10.0,
    )

    # Control
    control_node = GenericProcessingNode(
        "Control",
        mock_controller,
        NodeIO(inputs=["trajectory", "vehicle_state", "observation"], output="action"),
        rate_hz=10.0,
    )

    nodes = [physics_node, sensor_node, perception_node, planning_node, control_node]
    executor = SingleProcessExecutor(nodes, context)

    # Run one step
    executor.run(duration=0.05, dt=0.01)

    # Physics should have updated sim_state
    mock_simulator.step.assert_called()
    assert context.sim_state is not None

    # Sensor should have updated vehicle_state
    assert context.vehicle_state is not None

    # Perception should have updated observation
    assert context.observation is not None
    assert isinstance(context.observation, Observation)

    # Planning should have produced trajectory
    mock_planner.process.assert_called()
    assert context.trajectory is not None

    # Control should have produced action
    mock_controller.process.assert_called()
    assert context.action is not None
