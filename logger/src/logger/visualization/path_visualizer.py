"""Path visualizer for creating vehicle trajectory markers."""

from core.data.ros import ColorRGBA, Header, Marker, Point, Pose, Quaternion, Vector3

from logger.ros_message_builder import to_ros_time


class PathVisualizer:
    """Creates visualization markers for vehicle path history."""

    def __init__(self, max_history: int = 10000, color: ColorRGBA | None = None) -> None:
        """Initialize path visualizer.

        Args:
            max_history: Maximum number of positions to keep in history.
                         Set to 0 for unlimited history.
            color: Color for the path line.
        """
        self.max_history = max_history
        self.color = color or ColorRGBA(r=0.0, g=0.8, b=1.0, a=0.9)
        self.positions: list[tuple[float, float]] = []

    def add_position(self, x: float, y: float) -> None:
        """Add a position to the path history.

        Args:
            x: X coordinate in map frame.
            y: Y coordinate in map frame.
        """
        self.positions.append((x, y))

        # Limit history size if max_history is set
        if self.max_history > 0 and len(self.positions) > self.max_history:
            self.positions.pop(0)

    def set_positions(self, positions: list[tuple[float, float]]) -> None:
        """Set all positions at once.

        Args:
            positions: List of (x, y) tuples representing positions in map frame.
        """
        self.positions = positions

    def create_marker(self, timestamp: float) -> Marker | None:
        """Create a LINE_STRIP marker for the vehicle path.

        Args:
            timestamp: Current timestamp.

        Returns:
            Marker representing the vehicle path, or None if no positions.
        """
        if not self.positions:
            return None

        ros_time = to_ros_time(timestamp)
        identity_quat = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        # Create points for LINE_STRIP
        points = [Point(x=pos[0], y=pos[1], z=0.1) for pos in self.positions]

        return Marker(
            header=Header(stamp=ros_time, frame_id="map"),
            ns="vehicle_path",
            id=0,
            type=4,  # LINE_STRIP
            action=0,
            pose=Pose(
                position=Point(x=0.0, y=0.0, z=0.0),  # Points are absolute
                orientation=identity_quat,
            ),
            scale=Vector3(x=0.15, y=0.0, z=0.0),  # x is line width
            color=self.color,
            points=points,
            frame_locked=True,
        )

    def clear(self) -> None:
        """Clear all position history."""
        self.positions.clear()
