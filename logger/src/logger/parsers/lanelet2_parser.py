"""Lanelet2 OSM parser for map data."""

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


@dataclass
class Lanelet2Node:
    """Represents a node in Lanelet2 map."""

    id: str
    x: float
    y: float


@dataclass
class Lanelet2Way:
    """Represents a way (sequence of nodes) in Lanelet2 map."""

    id: str
    node_ids: list[str]


@dataclass
class Lanelet2Lanelet:
    """Represents a lanelet (lane) in Lanelet2 map."""

    id: str
    left_way_id: str | None
    right_way_id: str | None


@dataclass
class Lanelet2Map:
    """Parsed Lanelet2 map data."""

    nodes: dict[str, Lanelet2Node]
    ways: dict[str, Lanelet2Way]
    lanelets: list[Lanelet2Lanelet]


class Lanelet2Parser:
    """Parser for Lanelet2 OSM format maps."""

    def __init__(self, map_path: str | Path) -> None:
        """Initialize parser.

        Args:
            map_path: Path to Lanelet2 OSM file.
        """
        self.map_path = Path(map_path)

    def parse(self) -> Lanelet2Map:
        """Parse Lanelet2 OSM file.

        Returns:
            Parsed map data.

        Raises:
            FileNotFoundError: If map file doesn't exist.
            Exception: If parsing fails.
        """
        if not self.map_path.exists():
            raise FileNotFoundError(f"Map file not found: {self.map_path}")

        tree = ElementTree.parse(self.map_path)
        root = tree.getroot()

        nodes = self._parse_nodes(root)
        ways = self._parse_ways(root)
        lanelets = self._parse_lanelets(root)

        return Lanelet2Map(nodes=nodes, ways=ways, lanelets=lanelets)

    def _parse_nodes(self, root: ElementTree.Element) -> dict[str, Lanelet2Node]:
        """Parse nodes from OSM XML.

        Args:
            root: Root element of OSM XML.

        Returns:
            Dictionary mapping node ID to Lanelet2Node.
        """
        nodes: dict[str, Lanelet2Node] = {}

        for node in root.findall("node"):
            nid = node.get("id")
            if not nid:
                continue

            local_x = None
            local_y = None

            for tag in node.findall("tag"):
                k = tag.get("k")
                v = tag.get("v")
                if k == "local_x" and v:
                    local_x = float(v)
                elif k == "local_y" and v:
                    local_y = float(v)

            if local_x is not None and local_y is not None:
                nodes[nid] = Lanelet2Node(id=nid, x=local_x, y=local_y)

        return nodes

    def _parse_ways(self, root: ElementTree.Element) -> dict[str, Lanelet2Way]:
        """Parse ways from OSM XML.

        Args:
            root: Root element of OSM XML.

        Returns:
            Dictionary mapping way ID to Lanelet2Way.
        """
        ways: dict[str, Lanelet2Way] = {}

        for way in root.findall("way"):
            way_id = way.get("id")
            if not way_id:
                continue

            node_ids = [nd.get("ref") for nd in way.findall("nd") if nd.get("ref")]
            ways[way_id] = Lanelet2Way(id=way_id, node_ids=node_ids)

        return ways

    def _parse_lanelets(self, root: ElementTree.Element) -> list[Lanelet2Lanelet]:
        """Parse lanelets from OSM XML.

        Args:
            root: Root element of OSM XML.

        Returns:
            List of Lanelet2Lanelet objects.
        """
        lanelets: list[Lanelet2Lanelet] = []

        for relation in root.findall("relation"):
            # Check if this is a lanelet
            is_lanelet = False
            for tag in relation.findall("tag"):
                if tag.get("k") == "type" and tag.get("v") == "lanelet":
                    is_lanelet = True
                    break

            if not is_lanelet:
                continue

            lanelet_id = relation.get("id")
            if not lanelet_id:
                continue

            # Get left and right boundary ways
            left_way_id = None
            right_way_id = None

            for member in relation.findall("member"):
                role = member.get("role")
                ref = member.get("ref")
                if role == "left" and ref:
                    left_way_id = ref
                elif role == "right" and ref:
                    right_way_id = ref

            lanelets.append(
                Lanelet2Lanelet(id=lanelet_id, left_way_id=left_way_id, right_way_id=right_way_id)
            )

        return lanelets
