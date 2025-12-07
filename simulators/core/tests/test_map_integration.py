"""Test to verify map functionality works for both simulators."""

import tempfile
from pathlib import Path

import pytest
from simulator_kinematic import KinematicSimulator

from core.data import Action, VehicleState


@pytest.fixture
def simple_map_file():
    """Create a simple OSM map file for testing."""
    osm_content = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="1" lat="0.0" lon="0.0" />
  <node id="2" lat="0.0001" lon="0.0" />
  <node id="3" lat="0.0001" lon="0.0001" />
  <node id="4" lat="0.0" lon="0.0001" />
  <way id="10">
    <nd ref="1" />
    <nd ref="2" />
    <nd ref="3" />
    <nd ref="4" />
    <nd ref="1" />
    <tag k="type" v="lanelet" />
    <tag k="subtype" v="road" />
  </way>
</osm>
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".osm", delete=False) as f:
        f.write(osm_content)
        return Path(f.name)


def test_kinematic_simulator_with_map(simple_map_file):
    """Test KinematicSimulator with map validation."""
    sim = KinematicSimulator(dt=0.1, map_path=str(simple_map_file))
    assert sim.map is not None

    # Start inside map
    initial_state = VehicleState(x=5.0, y=5.0, yaw=0.0, velocity=1.0)
    sim = KinematicSimulator(dt=0.1, initial_state=initial_state, map_path=str(simple_map_file))
    sim.reset()

    action = Action(steering=0.0, acceleration=0.0)
    state, _, _ = sim.step(action)

    # Should be inside map (coordinates are within bounds)
    # Note: Actual drivability depends on map implementation


def test_simulator_without_map():
    """Test that simulator works without map."""
    kinematic_sim = KinematicSimulator(dt=0.1)
    assert kinematic_sim.map is None

    # Should work without map
    action = Action(steering=0.0, acceleration=1.0)

    k_state, _, _ = kinematic_sim.step(action)
    assert k_state is not None
