"""Unit tests for LaneletMap class."""

from pathlib import Path
from unittest.mock import patch

import pytest

from simulator.map import LaneletMap


class TestLaneletMap:
    """Tests for LaneletMap parsing and geometric operations."""

    @pytest.fixture
    def map_path(self) -> Path:
        """Get path to the lanelet2 map."""
        return Path(__file__).parent / "assets/lanelet2_map.osm"

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

    def test_empty_map_behavior(self) -> None:
        """Test behavior when map fails to load or has no lanelets."""
        # Patch parse_osm_for_collision to return None
        with (
            patch("simulator.map.parse_osm_for_collision", return_value=None),
            pytest.raises(ValueError, match="Failed to load drivable area"),
        ):
            LaneletMap(Path("dummy.osm"))


class TestLaneletMapPolygon:
    """Tests for polygon drivability checking."""

    @pytest.fixture
    def map_path(self) -> Path:
        """Get path to the lanelet2 map."""
        return Path(__file__).parent / "assets/lanelet2_map.osm"

    @pytest.fixture
    def lanelet_map(self, map_path: Path) -> LaneletMap:
        """Create LaneletMap instance."""
        return LaneletMap(map_path)

    def test_is_drivable_polygon_inside(self, lanelet_map: LaneletMap) -> None:
        """Test polygon completely inside drivable area."""
        from shapely.geometry import Polygon

        # Create small polygon inside the map
        mid_x = (89653.9564 + 89660.3701) / 2
        mid_y = (43131.2322 + 43128.8083) / 2

        # Small polygon around the midpoint
        polygon = Polygon(
            [
                (mid_x - 0.5, mid_y - 0.5),
                (mid_x + 0.5, mid_y - 0.5),
                (mid_x + 0.5, mid_y + 0.5),
                (mid_x - 0.5, mid_y + 0.5),
            ]
        )

        assert lanelet_map.is_drivable_polygon(polygon) is True

    def test_is_drivable_polygon_outside(self, lanelet_map: LaneletMap) -> None:
        """Test polygon completely outside drivable area."""
        from shapely.geometry import Polygon

        # Polygon far from the map
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        assert lanelet_map.is_drivable_polygon(polygon) is False

    def test_is_drivable_polygon_empty_map(self) -> None:
        """Test polygon checking with empty map."""
        with (
            patch("simulator.map.parse_osm_for_collision", return_value=None),
            pytest.raises(ValueError, match="Failed to load drivable area"),
        ):
            LaneletMap(Path("dummy.osm"))


class TestLaneletMapErrorHandling:
    """Tests for error handling in LaneletMap."""

    def test_nonexistent_file(self) -> None:
        """Test loading nonexistent OSM file."""
        # parse_osm_for_collision returns None for nonexistent files
        # LaneletMap should raise ValueError
        with pytest.raises(ValueError, match="Failed to load drivable area"):
            LaneletMap(Path("/nonexistent/file.osm"))

    def test_invalid_osm_format(self, tmp_path: Path) -> None:
        """Test loading invalid OSM file."""
        # Create invalid XML file
        invalid_file = tmp_path / "invalid.osm"
        invalid_file.write_text("This is not valid XML")

        # parse_osm_for_collision catches exceptions and returns None
        with pytest.raises(ValueError, match="Failed to load drivable area"):
            LaneletMap(invalid_file)

    def test_empty_osm_file(self, tmp_path: Path) -> None:
        """Test loading empty OSM file."""
        # Create valid but empty OSM file
        empty_file = tmp_path / "empty.osm"
        empty_file.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<osm></osm>')

        # Should raise ValueError because no drivable area is found
        with pytest.raises(ValueError, match="Failed to load drivable area"):
            LaneletMap(empty_file)
