"""Unit tests for LaneletMap class."""

from pathlib import Path

import pytest
from simulator_core.map import LaneletMap


class TestLaneletMap:
    """Tests for LaneletMap parsing and geometric operations."""

    @pytest.fixture
    def map_path(self) -> Path:
        """Get path to the lanelet2 map."""
        # Test is in simulators/core/tests, map is in simulators/core/assets
        return Path(__file__).parent.parent / "assets/lanelet2_map.osm"

    @pytest.fixture
    def lanelet_map(self, map_path: Path) -> LaneletMap:
        """Create LaneletMap instance."""
        return LaneletMap(map_path)

    def test_lanelet_map_initialization(self, map_path: Path) -> None:
        """Test that LaneletMap can be initialized from OSM file."""
        lanelet_map = LaneletMap(map_path)
        assert lanelet_map is not None
        assert lanelet_map.drivable_area is not None

    def test_is_drivable_outside_map(self, lanelet_map: LaneletMap) -> None:
        """Test that points outside the map are correctly identified."""
        # Origin is definitely outside the map (map coordinates are ~89000, ~43000)
        assert lanelet_map.is_drivable(0.0, 0.0) is False
        assert lanelet_map.is_drivable(-1000.0, -1000.0) is False
        assert lanelet_map.is_drivable(1000000.0, 1000000.0) is False

    def test_is_drivable_inside_map(self, lanelet_map: LaneletMap) -> None:
        """Test that points inside the map are correctly identified."""
        # Use midpoint between two known nodes to ensure we're inside a lanelet
        # Node 8: (89653.9564, 43131.2322)
        # Node 11: (89660.3701, 43128.8083)
        mid_x = (89653.9564 + 89660.3701) / 2
        mid_y = (43131.2322 + 43128.8083) / 2

        assert lanelet_map.is_drivable(mid_x, mid_y) is True

    def test_is_drivable_multiple_points(self, lanelet_map: LaneletMap) -> None:
        """Test multiple points to verify consistent behavior."""
        # Inside points (verified midpoint between nodes)
        mid_x = (89653.9564 + 89660.3701) / 2
        mid_y = (43131.2322 + 43128.8083) / 2
        inside_points = [
            (mid_x, mid_y),  # Verified midpoint
        ]

        # Outside points
        outside_points = [
            (0.0, 0.0),
            (50000.0, 50000.0),
            (100000.0, 100000.0),
        ]

        for x, y in inside_points:
            assert lanelet_map.is_drivable(x, y) is True, f"Point ({x}, {y}) should be inside"

        for x, y in outside_points:
            assert lanelet_map.is_drivable(x, y) is False, f"Point ({x}, {y}) should be outside"
