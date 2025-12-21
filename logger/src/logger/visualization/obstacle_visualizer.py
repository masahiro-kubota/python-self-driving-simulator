"""Obstacle visualizer for creating obstacle markers."""

from typing import Any

from core.data.ros import ColorRGBA, Header, Marker, MarkerArray, Point, Pose, Vector3
from logger.ros_message_builder import quaternion_from_yaw, to_ros_time


class ObstacleVisualizer:
    """Creates visualization markers for obstacles."""

    def create_marker_array(self, obstacles: list, timestamp: float) -> MarkerArray:
        """Create marker array for all obstacles.

        Args:
            obstacles: List of obstacle objects.
            timestamp: Current timestamp.

        Returns:
            MarkerArray containing all obstacle markers.
        """
        markers = []

        for idx, obs in enumerate(obstacles):
            marker = self._create_single_marker(obs, idx, timestamp)
            if marker:
                markers.append(marker)

        return MarkerArray(markers=markers)

    def _create_single_marker(self, obstacle: Any, index: int, timestamp: float) -> Marker | None:
        """Create a single marker for an obstacle.

        Args:
            obstacle: Obstacle object.
            index: Obstacle index (for marker ID).
            timestamp: Current timestamp.

        Returns:
            Marker for the obstacle, or None if shape is unsupported.
        """
        from simulator.obstacle import get_obstacle_state

        obs_state = get_obstacle_state(obstacle, timestamp)
        ros_time = to_ros_time(timestamp)
        obs_q = quaternion_from_yaw(obs_state.yaw)

        # Determine marker type and scale based on obstacle shape
        marker_type, scale = self._determine_marker_type_and_scale(obstacle.shape)
        if marker_type is None:
            return None

        return Marker(
            header=Header(stamp=ros_time, frame_id="map"),
            ns="obstacles",
            id=index,
            type=marker_type,
            action=0,
            pose=Pose(
                position=Point(x=obs_state.x, y=obs_state.y, z=scale.z / 2),
                orientation=obs_q,
            ),
            scale=scale,
            color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.7),  # Red for obstacles
            frame_locked=True,
        )

    def _determine_marker_type_and_scale(self, shape: Any) -> tuple[int | None, Vector3 | None]:
        """Determine marker type and scale from obstacle shape.

        Args:
            shape: Obstacle shape object.

        Returns:
            Tuple of (marker_type, scale), or (None, None) if unsupported.
        """
        if shape.type == "rectangle":
            marker_type = 1  # CUBE
            scale = Vector3(
                x=shape.length,
                y=shape.width,
                z=1.5,  # Default height
            )
        elif shape.type == "circle":
            marker_type = 2  # SPHERE
            scale = Vector3(x=shape.radius * 2, y=shape.radius * 2, z=shape.radius * 2)
        else:
            return None, None

        return marker_type, scale
