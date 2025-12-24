"""Tests for Simulator Node."""

import unittest.mock
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.data import Action, VehicleParameters, VehicleState
from core.data.node_io import NodeIO
from core.interfaces.node import NodeExecutionResult
from simulator.simulator import Simulator, SimulatorConfig

DUMMY_VP = VehicleParameters(
    wheelbase=2.5,
    width=1.8,
    front_overhang=1.0,
    rear_overhang=1.0,
    max_steering_angle=0.6,
    max_velocity=20.0,
    max_acceleration=3.0,
    mass=1500.0,
    inertia=2500.0,
    lf=1.2,
    lr=1.3,
    cf=80000.0,
    cr=80000.0,
    c_drag=0.3,
    c_roll=0.015,
    max_drive_force=5000.0,
    max_brake_force=8000.0,
    tire_params={},
)

DUMMY_INITIAL_STATE = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0, timestamp=0.0)


@pytest.fixture
def _mock_map():
    """Mock LaneletMap to avoid needing real map files for generic tests."""
    with unittest.mock.patch("simulator.map.LaneletMap") as mock_map_class:
        instance = mock_map_class.return_value
        instance.is_drivable.return_value = True
        instance.is_drivable_polygon.return_value = True
        yield mock_map_class


class TestSimulatorNode:
    """Tests for Simulator as a Node."""

    def test_initialization(self, _mock_map) -> None:
        """Test Simulator initialization with config."""
        config = SimulatorConfig(
            vehicle_params=DUMMY_VP,
            initial_state=VehicleState(x=10.0, y=5.0, yaw=1.0, velocity=2.0, timestamp=0.0),
            map_path=Path("dummy_map.osm"),
        )

        sim = Simulator(config=config, rate_hz=10.0)

        assert sim.name == "Simulator"
        assert sim.rate_hz == 10.0
        assert sim.config.initial_state.x == 10.0
        assert sim.config.initial_state.y == 5.0
        assert isinstance(sim.config.vehicle_params, VehicleParameters)

    def test_node_io(self, _mock_map) -> None:
        """Test that Simulator defines correct node IO."""
        config = SimulatorConfig(
            vehicle_params=DUMMY_VP,
            initial_state=DUMMY_INITIAL_STATE,
            map_path=Path("dummy_map.osm"),
        )
        sim = Simulator(config=config, rate_hz=10.0)

        node_io = sim.get_node_io()

        assert isinstance(node_io, NodeIO)
        assert "action" in node_io.inputs
        assert "sim_state" in node_io.outputs

    def test_on_init(self, _mock_map) -> None:
        """Test on_init initializes state correctly."""
        config = SimulatorConfig(
            vehicle_params=DUMMY_VP,
            initial_state=VehicleState(x=5.0, y=3.0, yaw=0.5, velocity=1.0, timestamp=0.0),
            map_path=Path("dummy_map.osm"),
        )

        sim = Simulator(config=config, rate_hz=10.0)
        sim.on_init()

        # Check internal state is initialized
        assert sim._current_state is not None
        assert sim._current_state.x == 5.0
        assert sim._current_state.y == 3.0
        assert sim.current_time == 0.0
        assert len(sim.log.steps) == 0

    def test_on_run_basic(self, _mock_map) -> None:
        """Test on_run executes physics step."""
        config = SimulatorConfig(
            vehicle_params=DUMMY_VP,
            initial_state=DUMMY_INITIAL_STATE,
            map_path=Path("dummy_map.osm"),
        )
        sim = Simulator(config=config, rate_hz=10.0)
        sim.on_init()

        # Create frame data manually
        frame_data = SimpleNamespace()
        frame_data.action = Action(steering=0.1, acceleration=1.0)
        frame_data.termination_signal = False
        sim.set_frame_data(frame_data)

        # Execute
        result = sim.on_run(0.0)

        assert result == NodeExecutionResult.SUCCESS
        assert hasattr(frame_data, "sim_state")
        assert frame_data.sim_state is not None
        assert isinstance(frame_data.sim_state, VehicleState)

    def test_on_run_updates_state(self, _mock_map) -> None:
        """Test that on_run updates vehicle state."""
        config = SimulatorConfig(
            vehicle_params=DUMMY_VP,
            initial_state=VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0, timestamp=0.0),
            map_path=Path("dummy_map.osm"),
        )
        sim = Simulator(config=config, rate_hz=10.0)
        sim.on_init()

        frame_data = SimpleNamespace()
        frame_data.action = Action(steering=0.0, acceleration=1.0)
        frame_data.termination_signal = False
        sim.set_frame_data(frame_data)

        # Run multiple steps
        for i in range(5):
            result = sim.on_run(i * 0.1)
            assert result == NodeExecutionResult.SUCCESS

        # State should have changed
        final_state = frame_data.sim_state
        assert final_state.velocity > 0.0  # Should have accelerated

    def test_on_run_without_action(self, _mock_map) -> None:
        """Test on_run with no action (should use default)."""
        config = SimulatorConfig(
            vehicle_params=DUMMY_VP,
            initial_state=DUMMY_INITIAL_STATE,
            map_path=Path("dummy_map.osm"),
        )
        sim = Simulator(config=config, rate_hz=10.0)
        sim.on_init()

        frame_data = SimpleNamespace()
        frame_data.termination_signal = False
        # Don't set action
        sim.set_frame_data(frame_data)

        result = sim.on_run(0.0)

        # Should still succeed with default action
        assert result == NodeExecutionResult.SUCCESS
        assert frame_data.sim_state is not None

    def test_on_run_with_termination_signal(self, _mock_map) -> None:
        """Test that on_run skips when termination signal is set."""
        config = SimulatorConfig(
            vehicle_params=DUMMY_VP,
            initial_state=DUMMY_INITIAL_STATE,
            map_path=Path("dummy_map.osm"),
        )
        sim = Simulator(config=config, rate_hz=10.0)
        sim.on_init()

        frame_data = SimpleNamespace()
        frame_data.termination_signal = True
        sim.set_frame_data(frame_data)

        result = sim.on_run(0.0)

        # Should succeed but not update
        assert result == NodeExecutionResult.SUCCESS

    def test_logging(self, _mock_map) -> None:
        """Test that simulation steps are logged."""
        config = SimulatorConfig(
            vehicle_params=DUMMY_VP,
            initial_state=DUMMY_INITIAL_STATE,
            map_path=Path("dummy_map.osm"),
        )
        sim = Simulator(config=config, rate_hz=10.0)
        sim.on_init()

        frame_data = SimpleNamespace()
        frame_data.action = Action(steering=0.1, acceleration=0.5)
        frame_data.termination_signal = False
        sim.set_frame_data(frame_data)

        # Run 3 steps
        for i in range(3):
            sim.on_run(i * 0.1)

        log = sim.get_log()
        assert len(log.steps) == 3

        # Check logged data
        for step in log.steps:
            assert step.timestamp is not None
            assert step.vehicle_state is not None
            assert step.action is not None

    def test_reset_via_on_init(self, _mock_map) -> None:
        """Test that on_init resets state."""
        config = SimulatorConfig(
            vehicle_params=DUMMY_VP,
            initial_state=DUMMY_INITIAL_STATE,
            map_path=Path("dummy_map.osm"),
        )
        sim = Simulator(config=config, rate_hz=10.0)
        sim.on_init()

        frame_data = SimpleNamespace()
        frame_data.action = Action(steering=0.0, acceleration=1.0)
        frame_data.termination_signal = False
        sim.set_frame_data(frame_data)

        # Run some steps
        for i in range(5):
            sim.on_run(i * 0.1)

        assert len(sim.log.steps) == 5

        # Reset
        sim.on_init()

        # Log should be cleared
        assert len(sim.log.steps) == 0
        assert sim.current_time == 0.0


class TestSimulatorWithMap:
    """Tests for Simulator with map integration."""

    def test_map_loading(self, tmp_path) -> None:
        """Test that map is loaded when map_path is provided."""
        # Create a minimal OSM file
        osm_content = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="1">
    <tag k="local_x" v="0.0"/>
    <tag k="local_y" v="0.0"/>
  </node>
  <node id="2">
    <tag k="local_x" v="10.0"/>
    <tag k="local_y" v="0.0"/>
  </node>
  <node id="3">
    <tag k="local_x" v="10.0"/>
    <tag k="local_y" v="5.0"/>
  </node>
  <node id="4">
    <tag k="local_x" v="0.0"/>
    <tag k="local_y" v="5.0"/>
  </node>
  <way id="10">
    <nd ref="1"/>
    <nd ref="2"/>
  </way>
  <way id="11">
    <nd ref="4"/>
    <nd ref="3"/>
  </way>
  <relation id="100">
    <tag k="type" v="lanelet"/>
    <tag k="subtype" v="road"/>
    <member type="way" ref="10" role="left"/>
    <member type="way" ref="11" role="right"/>
  </relation>
</osm>"""

        map_file = tmp_path / "test_map.osm"
        map_file.write_text(osm_content)

        config = SimulatorConfig(
            vehicle_params=DUMMY_VP,
            initial_state=DUMMY_INITIAL_STATE,
            map_path=map_file,
        )

        sim = Simulator(config=config, rate_hz=10.0)
        sim.on_init()

        # Map should be loaded
        assert sim.map is not None
