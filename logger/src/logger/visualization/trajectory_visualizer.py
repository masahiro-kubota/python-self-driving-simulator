"""Trajectory visualizer for creating trajectory markers."""

from core.data import Trajectory
from core.data.ros import ColorRGBA, Header, Marker, Point, Pose, Quaternion, Vector3
from logger.ros_message_builder import to_ros_time


class TrajectoryVisualizer:
    """Creates visualization markers for trajectories."""

    def create_marker(self, trajectory: Trajectory, timestamp: float) -> Marker:
        """Create a marker for the trajectory (lookahead point).

        Args:
            trajectory: Trajectory data.
            timestamp: Current timestamp.

        Returns:
            Marker representing the trajectory (target point).
        """
        ros_time = to_ros_time(timestamp)
        identity_quat = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        # If trajectory has only one point, it's likely a lookahead target
        # We visualize it as a Sphere
        if len(trajectory.points) == 1:
            point = trajectory.points[0]
            return Marker(
                header=Header(stamp=ros_time, frame_id="map"),
                ns="lookahead_point",
                id=0,
                type=2,  # SPHERE
                action=0,
                pose=Pose(
                    position=Point(x=point.x, y=point.y, z=0.5),  # Slightly elevated
                    orientation=identity_quat,
                ),
                scale=Vector3(x=0.5, y=0.5, z=0.5),
                color=ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.8),  # Magenta
                frame_locked=True,
            )

        # If multiple points, visualize as Path (Line Strip)
        # (Not fully needed for current PurePursuit implementation but good for future)
        points = [Point(x=p.x, y=p.y, z=0.2) for p in trajectory.points]
        return Marker(
            header=Header(stamp=ros_time, frame_id="map"),
            ns="trajectory",
            id=1,
            type=4,  # LINE_STRIP
            action=0,
            pose=Pose(
                position=Point(x=0.0, y=0.0, z=0.0),  # Points are absolute
                orientation=identity_quat,
            ),
            scale=Vector3(x=0.2, y=0.0, z=0.0),  # x is line width
            color=ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8),  # Green
            points=points,
            frame_locked=True,
        )
