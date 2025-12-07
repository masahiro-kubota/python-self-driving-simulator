"""Tests for SingleProcessExecutor."""

from unittest.mock import MagicMock

import pytest
from experiment_runner.executor import SingleProcessExecutor

from core.data import Action, SimulationLog, Trajectory, TrajectoryPoint, VehicleState
from core.interfaces import Controller, Planner, Simulator
from core.interfaces.node import SimulationContext
from core.nodes import ControlNode, PhysicsNode, PlanningNode, SensorNode


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
    planner = MagicMock(spec=Planner)
    planner.plan.return_value = Trajectory(
        points=[TrajectoryPoint(x=1.0, y=1.0, yaw=0.0, velocity=1.0)]
    )
    return planner


@pytest.fixture
def mock_controller():
    controller = MagicMock(spec=Controller)
    controller.control.return_value = Action(acceleration=1.0, steering=0.1)
    return controller


def test_executor_timing(mock_simulator, mock_planner, mock_controller):
    """Test that nodes run at expected rates."""

    context = SimulationContext()

    # Manually assembling nodes (like StandardADComponent would do)
    physics_node = PhysicsNode(mock_simulator, rate_hz=10.0)
    planning_node = PlanningNode(mock_planner, rate_hz=5.0)
    control_node = ControlNode(mock_controller, rate_hz=10.0)
    sensor_node = SensorNode(rate_hz=10.0)

    physics_node.on_run = MagicMock(wraps=physics_node.on_run)
    planning_node.on_run = MagicMock(wraps=planning_node.on_run)
    control_node.on_run = MagicMock(wraps=control_node.on_run)
    # Sensor node needs mocks for on_run context updates?
    # SensorNode logic is simple, so no need to mock logic, just count calls
    sensor_node.on_run = MagicMock(wraps=sensor_node.on_run)

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

    physics_node = PhysicsNode(mock_simulator, rate_hz=10.0)
    sensor_node = SensorNode(rate_hz=10.0)
    planning_node = PlanningNode(mock_planner, rate_hz=10.0)
    control_node = ControlNode(mock_controller, rate_hz=10.0)

    nodes = [physics_node, sensor_node, planning_node, control_node]
    executor = SingleProcessExecutor(nodes, context)

    # Run one step
    executor.run(duration=0.05, dt=0.01)

    # Physics should have updated sim_state
    mock_simulator.step.assert_called()
    assert context.sim_state is not None

    # Sensor should have updated vehicle_state & observation
    assert context.vehicle_state is not None
    assert context.observation is not None

    # Planning should have produced trajectory
    mock_planner.plan.assert_called()
    assert context.trajectory is not None

    # Control should have produced action
    mock_controller.control.assert_called()
    assert context.action is not None
