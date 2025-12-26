"""Lanelet2 map implementation using Shapely."""

from pathlib import Path

from core.utils.osm_parser import parse_osm_for_collision
from shapely.geometry import Point, Polygon


class LaneletMap:
    """Lanelet2 map for drivable area checking."""

    def __init__(self, osm_path: Path) -> None:
        """Initialize LaneletMap.

        Args:
            osm_path: Path to the .osm file
        """
        self.drivable_area: Polygon | None = parse_osm_for_collision(osm_path)
        if self.drivable_area is None:
            raise ValueError(f"Failed to load drivable area from {osm_path}")

    def is_drivable(self, x: float, y: float) -> bool:
        """Check if the point is within the drivable area.

        Args:
            x: X coordinate [m]
            y: Y coordinate [m]

        Returns:
            True if within drivable area, False otherwise
        """
        # Drivable area is guaranteed to be loaded in init
        point = Point(x, y)
        return self.drivable_area.contains(point)

    def is_drivable_polygon(self, polygon: Polygon) -> bool:
        """Check if the polygon is within the drivable area.

        Args:
            polygon: Shapely Polygon to check

        Returns:
            True if fully within drivable area, False otherwise
        """
        if self.drivable_area is None:
            return True

        return self.drivable_area.contains(polygon)
