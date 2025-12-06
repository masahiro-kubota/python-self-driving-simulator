"""Lanelet2 map implementation using Shapely."""

import xml.etree.ElementTree as ET
from pathlib import Path

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


class LaneletMap:
    """Lanelet2 map for drivable area checking."""

    def __init__(self, osm_path: Path) -> None:
        """Initialize LaneletMap.

        Args:
            osm_path: Path to the .osm file
        """
        self.drivable_area: Polygon | None = None
        self._load_map(osm_path)

    def _load_map(self, osm_path: Path) -> None:
        """Load and parse OSM file.

        Args:
            osm_path: Path to the .osm file
        """
        tree = ET.parse(osm_path)
        root = tree.getroot()

        nodes: dict[int, tuple[float, float]] = {}
        ways: dict[int, list[int]] = {}
        lanelets: list[Polygon] = []

        # Parse nodes
        for node in root.findall("node"):
            node_id = int(node.get("id", 0))
            local_x = 0.0
            local_y = 0.0

            for tag in node.findall("tag"):
                if tag.get("k") == "local_x":
                    local_x = float(tag.get("v", 0.0))
                elif tag.get("k") == "local_y":
                    local_y = float(tag.get("v", 0.0))

            nodes[node_id] = (local_x, local_y)

        # Parse ways
        for way in root.findall("way"):
            way_id = int(way.get("id", 0))
            nd_refs = [int(nd.get("ref", 0)) for nd in way.findall("nd")]
            ways[way_id] = nd_refs

        # Parse relations (lanelets)
        for relation in root.findall("relation"):
            is_lanelet = False
            left_way_id = None
            right_way_id = None

            for tag in relation.findall("tag"):
                if tag.get("k") == "type" and tag.get("v") == "lanelet":
                    is_lanelet = True

            if not is_lanelet:
                continue

            for member in relation.findall("member"):
                role = member.get("role")
                ref = int(member.get("ref", 0))
                if role == "left":
                    left_way_id = ref
                elif role == "right":
                    right_way_id = ref

            if left_way_id and right_way_id and left_way_id in ways and right_way_id in ways:
                left_nodes = ways[left_way_id]
                right_nodes = ways[right_way_id]

                # Create polygon from left and right boundaries
                # Left boundary points (forward)
                polygon_points = [nodes[nid] for nid in left_nodes if nid in nodes]
                # Right boundary points (reverse to close loop)
                polygon_points.extend([nodes[nid] for nid in reversed(right_nodes) if nid in nodes])

                if len(polygon_points) >= 3:
                    lanelets.append(Polygon(polygon_points))

        # Merge all lanelets into a single drivable area
        if lanelets:
            self.drivable_area = unary_union(lanelets)

    def is_drivable(self, x: float, y: float) -> bool:
        """Check if the point is within the drivable area.

        Args:
            x: X coordinate [m]
            y: Y coordinate [m]

        Returns:
            True if within drivable area, False otherwise
        """
        if self.drivable_area is None:
            return True  # If map failed to load, assume everywhere is drivable

        point = Point(x, y)
        return self.drivable_area.contains(point)
