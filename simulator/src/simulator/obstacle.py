"""Obstacle management and collision detection."""

import math
from typing import TYPE_CHECKING

from core.data import CsvPathTrajectory, ObstacleTrajectory, TrajectoryWaypoint

if TYPE_CHECKING:
    from core.data import ObstacleState, SimulatorObstacle
    from shapely.geometry import Polygon
import csv
from pathlib import Path


def load_csv_trajectory(trajectory: CsvPathTrajectory) -> ObstacleTrajectory:
    """Load a CSV path and convert it into a waypoint trajectory."""

    expected_fields = {"x", "y", "z", "x_quat", "y_quat", "z_quat", "w_quat", "speed"}

    path = Path(trajectory.path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        msg = f"CSV trajectory file not found: {path}"
        raise FileNotFoundError(msg)

    waypoints: list[TrajectoryWaypoint] = []
    current_time = 0.0
    last_speed = 0.0

    with path.open() as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
        missing = expected_fields - header
        if missing:
            msg = f"CSV trajectory missing fields: {sorted(missing)}"
            raise ValueError(msg)

        prev_x = None
        prev_y = None

        for row in reader:
            try:
                x = float(row["x"])
                y = float(row["y"])
                z_quat = float(row["z_quat"])
                w_quat = float(row["w_quat"])
                speed = float(row["speed"])
            except (TypeError, ValueError) as exc:
                msg = f"Invalid numeric value in CSV trajectory row: {row}"
                raise ValueError(msg) from exc

            # Compute yaw from z-w quaternion (2D assumption)
            yaw = 2.0 * math.atan2(z_quat, w_quat)

            if prev_x is not None and prev_y is not None:
                dist = math.hypot(x - prev_x, y - prev_y)
                effective_speed = (
                    speed if speed > 1e-3 else (last_speed if last_speed > 1e-3 else 1.0)
                )
                dt = dist / effective_speed if effective_speed > 0 else 0.0
                current_time += dt

            waypoints.append(TrajectoryWaypoint(time=current_time, x=x, y=y, yaw=yaw))

            prev_x, prev_y = x, y
            if speed > 1e-3:
                last_speed = speed

    if not waypoints:
        msg = f"CSV trajectory is empty: {path}"
        raise ValueError(msg)

    return ObstacleTrajectory(
        type="waypoint",
        interpolation="linear",
        waypoints=waypoints,
        loop=trajectory.loop,
    )


def get_obstacle_state(obstacle: "SimulatorObstacle", time: float) -> "ObstacleState":
    """Get obstacle state at a specific time.

    Args:
        obstacle: Obstacle definition
        time: Current simulation time [s]

    Returns:
        ObstacleState at the specified time
    """
    from core.data import ObstacleState

    if obstacle.type == "static":
        if obstacle.position is None:
            msg = "Static obstacle must have position"
            raise ValueError(msg)
        return ObstacleState(
            x=obstacle.position.x,
            y=obstacle.position.y,
            yaw=obstacle.position.yaw,
            timestamp=time,
        )

    # Dynamic obstacle
    if obstacle.trajectory is None:
        msg = "Dynamic obstacle must have trajectory"
        raise ValueError(msg)

    trajectory = obstacle.trajectory
    waypoints = trajectory.waypoints

    if len(waypoints) == 0:
        msg = "Trajectory must have at least one waypoint"
        raise ValueError(msg)

    # Handle looping
    if trajectory.loop and len(waypoints) > 1:
        # Calculate total duration
        total_duration = waypoints[-1].time - waypoints[0].time
        if total_duration > 0:
            # Normalize time to [0, total_duration)
            time_offset = waypoints[0].time
            normalized_time = (time - time_offset) % total_duration + time_offset
        else:
            normalized_time = time
    else:
        normalized_time = time

    # Find surrounding waypoints
    if normalized_time <= waypoints[0].time:
        # Before first waypoint
        wp = waypoints[0]
        return ObstacleState(x=wp.x, y=wp.y, yaw=wp.yaw, timestamp=time)

    if normalized_time >= waypoints[-1].time:
        # After last waypoint
        wp = waypoints[-1]
        return ObstacleState(x=wp.x, y=wp.y, yaw=wp.yaw, timestamp=time)

    # Find interpolation interval
    for i in range(len(waypoints) - 1):
        if waypoints[i].time <= normalized_time <= waypoints[i + 1].time:
            wp1 = waypoints[i]
            wp2 = waypoints[i + 1]

            # Linear interpolation
            if trajectory.interpolation == "linear":
                dt = wp2.time - wp1.time
                if dt > 0:
                    alpha = (normalized_time - wp1.time) / dt
                else:
                    alpha = 0.0

                x = wp1.x + alpha * (wp2.x - wp1.x)
                y = wp1.y + alpha * (wp2.y - wp1.y)

                # Interpolate yaw (handle angle wrapping)
                dyaw = wp2.yaw - wp1.yaw
                # Normalize to [-pi, pi]
                while dyaw > math.pi:
                    dyaw -= 2 * math.pi
                while dyaw < -math.pi:
                    dyaw += 2 * math.pi
                yaw = wp1.yaw + alpha * dyaw

                return ObstacleState(x=x, y=y, yaw=yaw, timestamp=time)

            # Cubic spline interpolation
            if trajectory.interpolation == "cubic_spline":
                # Use scipy for cubic spline
                try:
                    from scipy.interpolate import CubicSpline

                    times = [wp.time for wp in waypoints]
                    xs = [wp.x for wp in waypoints]
                    ys = [wp.y for wp in waypoints]
                    yaws = [wp.yaw for wp in waypoints]

                    cs_x = CubicSpline(times, xs)
                    cs_y = CubicSpline(times, ys)
                    cs_yaw = CubicSpline(times, yaws)

                    x = float(cs_x(normalized_time))
                    y = float(cs_y(normalized_time))
                    yaw = float(cs_yaw(normalized_time))

                    return ObstacleState(x=x, y=y, yaw=yaw, timestamp=time)
                except ImportError:
                    # Fallback to linear if scipy not available
                    dt = wp2.time - wp1.time
                    if dt > 0:
                        alpha = (normalized_time - wp1.time) / dt
                    else:
                        alpha = 0.0

                    x = wp1.x + alpha * (wp2.x - wp1.x)
                    y = wp1.y + alpha * (wp2.y - wp1.y)
                    dyaw = wp2.yaw - wp1.yaw
                    while dyaw > math.pi:
                        dyaw -= 2 * math.pi
                    while dyaw < -math.pi:
                        dyaw += 2 * math.pi
                    yaw = wp1.yaw + alpha * dyaw

                    return ObstacleState(x=x, y=y, yaw=yaw, timestamp=time)

    # Should not reach here
    wp = waypoints[-1]
    return ObstacleState(x=wp.x, y=wp.y, yaw=wp.yaw, timestamp=time)


def get_obstacle_polygon(obstacle: "SimulatorObstacle", state: "ObstacleState") -> "Polygon":
    """Get obstacle polygon for collision detection.

    Args:
        obstacle: Obstacle definition
        state: Obstacle state

    Returns:
        Shapely Polygon
    """
    from shapely.geometry import Point, Polygon

    shape = obstacle.shape

    if shape.type == "circle":
        if shape.radius is None:
            msg = "Circle shape must have radius"
            raise ValueError(msg)
        # Create circle as polygon with 16 points
        center = Point(state.x, state.y)
        return center.buffer(shape.radius)

    if shape.type == "rectangle":
        if shape.width is None or shape.length is None:
            msg = "Rectangle shape must have width and length"
            raise ValueError(msg)

        # Create rectangle polygon
        # Rectangle is centered at (state.x, state.y) with orientation state.yaw
        half_width = shape.width / 2.0
        half_length = shape.length / 2.0

        # Rectangle corners in local frame (x forward, y left)
        corners = [
            (half_length, half_width),
            (half_length, -half_width),
            (-half_length, -half_width),
            (-half_length, half_width),
        ]

        # Transform to global frame
        cos_yaw = math.cos(state.yaw)
        sin_yaw = math.sin(state.yaw)

        points = []
        for lx, ly in corners:
            # Rotate
            gx = lx * cos_yaw - ly * sin_yaw
            gy = lx * sin_yaw + ly * cos_yaw
            # Translate
            points.append((gx + state.x, gy + state.y))

        return Polygon(points)

    msg = f"Unknown shape type: {shape.type}"
    raise ValueError(msg)


def check_collision(vehicle_polygon: "Polygon", obstacle_polygon: "Polygon") -> bool:
    """Check collision between vehicle and obstacle.

    Args:
        vehicle_polygon: Vehicle polygon
        obstacle_polygon: Obstacle polygon

    Returns:
        True if collision detected
    """
    return vehicle_polygon.intersects(obstacle_polygon)


class ObstacleManager:
    """Manage multiple obstacles and check collisions."""

    def __init__(self, obstacles: list["SimulatorObstacle"]) -> None:
        """Initialize obstacle manager.

        Args:
            obstacles: List of obstacles
        """
        self.obstacles = obstacles

    def check_vehicle_collision(self, vehicle_polygon: "Polygon", current_time: float) -> bool:
        """Check if vehicle collides with any obstacle.

        Args:
            vehicle_polygon: Vehicle polygon
            current_time: Current simulation time [s]

        Returns:
            True if collision detected
        """
        for obstacle in self.obstacles:
            obstacle_state = get_obstacle_state(obstacle, current_time)
            obstacle_polygon = get_obstacle_polygon(obstacle, obstacle_state)

            if check_collision(vehicle_polygon, obstacle_polygon):
                return True

        return False
