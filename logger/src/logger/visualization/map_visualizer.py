"""Map visualizer for creating map markers from Lanelet2 data."""

from core.data.ros import ColorRGBA, Header, Marker, MarkerArray, Point, Pose, Quaternion, Vector3

from logger.parsers.lanelet2_parser import Lanelet2Map, Lanelet2Parser
from logger.ros_message_builder import to_ros_time


class MapVisualizer:
    """Creates visualization markers for Lanelet2 maps."""

    def __init__(
        self,
        parser: Lanelet2Parser,
        left_color: ColorRGBA | None = None,
        right_color: ColorRGBA | None = None,
    ) -> None:
        """Initialize map visualizer.

        Args:
            parser: Lanelet2 parser instance.
            left_color: Color for the left lane boundaries.
            right_color: Color for the right lane boundaries.
        """
        self.parser = parser
        self.left_color = left_color or ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.8)
        self.right_color = right_color or ColorRGBA(r=0.8, g=0.8, b=0.8, a=0.8)
        self.map_data: Lanelet2Map | None = None

    def load_map(self) -> None:
        """Load and parse the map data."""
        self.map_data = self.parser.parse()

    def create_marker_array(self, timestamp: float) -> MarkerArray:
        """Create marker array for the map.

        Args:
            timestamp: Current timestamp.

        Returns:
            MarkerArray containing map boundary markers.
        """
        if self.map_data is None:
            self.load_map()

        markers = []
        marker_id = 0

        for lanelet in self.map_data.lanelets:
            # Create marker for left boundary
            if lanelet.left_way_id and lanelet.left_way_id in self.map_data.ways:
                left_marker = self._create_boundary_marker(
                    way_id=lanelet.left_way_id,
                    namespace="lanelet_left",
                    marker_id=marker_id,
                    timestamp=timestamp,
                    color=self.left_color,  # White
                )
                if left_marker:
                    markers.append(left_marker)
                    marker_id += 1

            # Create marker for right boundary
            if lanelet.right_way_id and lanelet.right_way_id in self.map_data.ways:
                right_marker = self._create_boundary_marker(
                    way_id=lanelet.right_way_id,
                    namespace="lanelet_right",
                    marker_id=marker_id,
                    timestamp=timestamp,
                    color=self.right_color,  # Light gray
                )
                if right_marker:
                    markers.append(right_marker)
                    marker_id += 1

        return MarkerArray(markers=markers)

    def _create_boundary_marker(
        self,
        way_id: str,
        namespace: str,
        marker_id: int,
        timestamp: float,
        color: ColorRGBA,
    ) -> Marker | None:
        """Create a marker for a boundary line.

        Args:
            way_id: Way ID to visualize.
            namespace: Marker namespace.
            marker_id: Marker ID.
            timestamp: Current timestamp.
            color: Marker color.

        Returns:
            Marker for the boundary, or None if insufficient nodes.
        """
        way = self.map_data.ways[way_id]
        valid_node_ids = [nid for nid in way.node_ids if nid in self.map_data.nodes]

        if len(valid_node_ids) < 2:
            return None

        ros_time = to_ros_time(timestamp)

        marker = Marker(
            header=Header(stamp=ros_time, frame_id="map"),
            ns=namespace,
            id=marker_id,
            type=4,  # LINE_STRIP
            action=0,
            scale=Vector3(x=0.15, y=0.0, z=0.0),
            color=color,
            pose=Pose(orientation=Quaternion(w=1.0)),
            frame_locked=True,
        )

        for nid in valid_node_ids:
            node = self.map_data.nodes[nid]
            marker.points.append(Point(x=node.x, y=node.y, z=0.0))

        return marker
