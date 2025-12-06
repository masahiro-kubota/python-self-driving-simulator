"""Test course boundary detection."""

from pathlib import Path

import pytest
from simulator_kinematic.simulator import KinematicSimulator

from core.data import Action, VehicleState


class TestBoundary:
    """Tests for course boundary detection."""

    @pytest.fixture
    def map_path(self) -> str:
        """Get path to the lanelet2 map."""
        # Assuming the test is run from the workspace root or using correct relative path
        workspace_root = Path(__file__).parent.parent.parent.parent
        return str(workspace_root / "simulators/core/assets/lanelet2_map.osm")

    def test_off_track_detection(self, map_path: str) -> None:
        """Test that vehicle is detected as off-track when outside the map."""
        # Initialize simulator with map
        sim = KinematicSimulator(map_path=map_path)

        # 1. Place vehicle inside the track
        # Based on OSM data, a valid point should be around the nodes
        # e.g., node id="8" lat="35.62581953015" lon="139.78142932763"
        # local_x="89653.9564", local_y="43131.2322"
        inside_x = 89653.9564
        inside_y = 43131.2322

        sim = KinematicSimulator(
            map_path=map_path,
            initial_state=VehicleState(x=inside_x, y=inside_y, yaw=0.0, velocity=0.0),
        )
        # sim.reset() only resets to initial_state passed in __init__
        state = sim.reset()

        # Verify initial state is on track (off_track should be False)
        # However, reset() doesn't return off_track in the state unless updated.
        # But _current_state should have it.
        # Wait, VehicleState default off_track is False.
        # We need to step to trigger the check.

        action = Action(steering=0.0, acceleration=0.0)
        state, _, _ = sim.step(action)

        # Check if we are still on track
        # Note: LaneletMap check might fail if (89653.9564, 43131.2322) is not exactly inside a lanelet polygon
        # (it might be a border node).
        # Let's check a point that is more likely inside.
        # Node 8 is part of way 9162 and 9167.
        # Way 9167 is right bound of relation 14. Way 9165 is left bound.
        # It's safer to rely on the fact that (0,0) is definitely OUTSIDE.

        # 2. Place vehicle outside the track (e.g., origin)
        outside_x = 0.0
        outside_y = 0.0

        sim = KinematicSimulator(
            map_path=map_path,
            initial_state=VehicleState(x=outside_x, y=outside_y, yaw=0.0, velocity=0.0),
        )
        state = sim.reset()
        state, _, _ = sim.step(action)

        assert state.off_track is True, "Vehicle should be off-track at (0,0)"

    def test_on_track_detection(self, map_path: str) -> None:
        """Test that vehicle is detected as on-track when inside the map."""

        # We need a point that is definitely inside a lanelet.
        mirror_node_8_x = 89653.9564
        mirror_node_8_y = 43131.2322
        mirror_node_11_x = 89660.3701
        mirror_node_11_y = 43128.8083

        mid_x = (mirror_node_8_x + mirror_node_11_x) / 2
        mid_y = (mirror_node_8_y + mirror_node_11_y) / 2

        # Initialize simulator with map and specific initial state
        sim = KinematicSimulator(
            map_path=map_path, initial_state=VehicleState(x=mid_x, y=mid_y, yaw=0.0, velocity=0.0)
        )
        sim.reset()

        action = Action(steering=0.0, acceleration=0.0)
        state, _, _ = sim.step(action)

        assert state.off_track is False, f"Vehicle should be on-track at ({mid_x}, {mid_y})"
