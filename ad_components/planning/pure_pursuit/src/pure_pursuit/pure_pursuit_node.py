import math
from pathlib import Path

from core.data import ComponentConfig, SimulatorObstacle, VehicleParameters, VehicleState
from core.data.autoware import Trajectory
from core.data.node_io import NodeIO
from core.data.ros import MarkerArray
from core.interfaces.node import Node, NodeExecutionResult
from core.utils.geometry import distance, euler_to_quaternion
from planning_utils import ReferencePathPoint
from pydantic import Field
from simulator.obstacle import get_obstacle_polygon, get_obstacle_state


class PurePursuitConfig(ComponentConfig):
    """Configuration for PurePursuitNode."""

    track_path: Path = Field(..., description="Path to reference trajectory CSV")
    min_lookahead_distance: float = Field(..., description="Minimum lookahead distance [m]")
    max_lookahead_distance: float = Field(..., description="Maximum lookahead distance [m]")
    lookahead_speed_ratio: float = Field(..., description="Lookahead distance speed ratio [s]")
    vehicle_params: VehicleParameters = Field(..., description="Vehicle parameters")
    lookahead_marker_color: str = Field("#FF00FFCC", description="Lookahead marker color")


class PurePursuitNode(Node[PurePursuitConfig]):
    """Pure Pursuit path tracking node."""

    def __init__(
        self,
        config: PurePursuitConfig,
        rate_hz: float,
        priority: int,
    ):
        super().__init__("PurePursuit", rate_hz, config, priority)
        self.vehicle_params = config.vehicle_params
        self.reference_trajectory = None
        # self.config is set by base class

        # Path resolution is handled by node_factory.create_node()
        from core.utils import get_project_root
        from planning_utils import load_track_csv

        track_path = self.config.track_path
        if not track_path.is_absolute():
            track_path = get_project_root() / track_path

        self.reference_trajectory = load_track_csv(track_path)

    def get_node_io(self) -> NodeIO:
        return NodeIO(
            inputs={"vehicle_state": VehicleState, "obstacles": list},
            outputs={
                "trajectory": Trajectory,
                "lookahead_marker": MarkerArray,
            },
        )

    def on_run(self, current_time: float) -> NodeExecutionResult:
        if self.frame_data is None:
            return NodeExecutionResult.FAILED

        self.current_time = current_time

        # Get Input
        vehicle_state = self.subscribe("vehicle_state")
        if vehicle_state is None:
            return NodeExecutionResult.SKIPPED

        # Process
        trajectory = self._plan(vehicle_state)

        # Set Output
        self.publish("trajectory", trajectory)

        # Output Debug Marker
        from core.data.ros import ColorRGBA, MarkerArray
        from planning_utils.visualization import create_trajectory_marker

        marker = create_trajectory_marker(
            trajectory=trajectory,
            timestamp=current_time,
            ns="pure_pursuit_lookahead",
            color=ColorRGBA.from_hex(self.config.lookahead_marker_color),
        )

        marker_array = MarkerArray(markers=[marker])
        self.publish("lookahead_marker", marker_array)

        return NodeExecutionResult.SUCCESS

    def _plan(self, vehicle_state: VehicleState) -> Trajectory:
        if self.reference_trajectory is None or len(self.reference_trajectory) < 2:
            return Trajectory(points=[])

        # 1. Calculate dynamic lookahead
        current_speed = vehicle_state.velocity
        lookahead = max(
            self.config.min_lookahead_distance,
            min(
                self.config.max_lookahead_distance,
                current_speed * self.config.lookahead_speed_ratio,
            ),
        )

        # 2. Find nearest point
        min_dist = float("inf")
        nearest_idx = 0

        for i, point in enumerate(self.reference_trajectory):
            d = distance(vehicle_state.x, vehicle_state.y, point.x, point.y)
            if d < min_dist:
                min_dist = d
                nearest_idx = i

        # 3. Find lookahead point
        target_point = self.reference_trajectory[nearest_idx]
        accumulated_dist = 0.0
        current_idx = nearest_idx

        # Collect points along the path for collision detection
        check_points = [target_point]

        while accumulated_dist < lookahead:
            if current_idx >= len(self.reference_trajectory) - 1:
                target_point = self.reference_trajectory[-1]
                check_points.append(target_point)
                break

            p1 = self.reference_trajectory[current_idx]
            p2 = self.reference_trajectory[current_idx + 1]
            d = distance(p1.x, p1.y, p2.x, p2.y)

            if accumulated_dist + d >= lookahead:
                remaining = lookahead - accumulated_dist
                ratio = remaining / d
                target_x = p1.x + (p2.x - p1.x) * ratio
                target_y = p1.y + (p2.y - p1.y) * ratio
                target_v = p1.velocity + (p2.velocity - p1.velocity) * ratio

                segment_yaw = math.atan2(p2.y - p1.y, p2.x - p1.x)
                target_yaw = segment_yaw

                target_point = ReferencePathPoint(
                    x=target_x, y=target_y, yaw=target_yaw, velocity=target_v
                )
                check_points.append(target_point)
                break

            accumulated_dist += d
            current_idx += 1
            check_points.append(self.reference_trajectory[current_idx])
            target_point = self.reference_trajectory[current_idx]

        # Obstacle Avoidance (Internal logic)
        obstacles = self.subscribe("obstacles") or []
        if obstacles:
            target_point = self._avoid_obstacles(
                target_point, obstacles, self.current_time, vehicle_state
            )

        # Convert to Autoware Trajectory
        from core.data.autoware import Duration, TrajectoryPoint
        from core.data.ros import Header, Point, Pose, Quaternion
        from core.utils.ros_message_builder import to_ros_time

        quat = euler_to_quaternion(0.0, 0.0, target_point.yaw)

        aw_point = TrajectoryPoint(
            time_from_start=Duration(sec=0, nanosec=0),  # Simplified
            pose=Pose(
                position=Point(x=target_point.x, y=target_point.y, z=0.0),
                orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
            ),
            longitudinal_velocity_mps=target_point.velocity,
        )

        return Trajectory(
            header=Header(stamp=to_ros_time(self.current_time), frame_id="map"), points=[aw_point]
        )

    def _avoid_obstacles(
        self,
        target_point: ReferencePathPoint,
        obstacles: list[SimulatorObstacle],
        current_time: float,
        vehicle_state: VehicleState,
    ) -> ReferencePathPoint:
        from shapely.geometry import LineString

        # Create path polygon (LineString buffer) representing the swept volume
        # Buffer 1.5m (approx vehicle width/2 + margin)
        path_line = LineString(
            [(vehicle_state.x, vehicle_state.y), (target_point.x, target_point.y)]
        )
        path_poly = path_line.buffer(1.5)

        collision_detected = False
        for obstacle in obstacles:
            obs_state = get_obstacle_state(obstacle, current_time)
            obs_poly = get_obstacle_polygon(obstacle, obs_state)
            if path_poly.intersects(obs_poly):
                collision_detected = True
                break

        if not collision_detected:
            return target_point

        # Simple avoidance: Try left and right offsets with varying distances
        import math

        # Ranges of offsets to try (in meters)
        # We try smaller offsets first? Or larger?
        # If we are colliding, we probably want to move away significantly.
        # Let's try a range.
        offsets_to_try = [3.0, 4.0, 5.0, 6.0]

        candidates = []
        for offset_dist in offsets_to_try:
            for direction in [1, -1]:  # 1: Left, -1: Right
                # Calculate offset perpendicular to target point yaw
                dx = -direction * offset_dist * math.sin(target_point.yaw)
                dy = direction * offset_dist * math.cos(target_point.yaw)

                cand_p = ReferencePathPoint(
                    x=target_point.x + dx,
                    y=target_point.y + dy,
                    yaw=target_point.yaw,
                    velocity=target_point.velocity,
                )
                candidates.append(cand_p)

        # Check candidates
        valid_candidates = []
        for cand in candidates:
            # Check path to candidate
            cand_line = LineString([(vehicle_state.x, vehicle_state.y), (cand.x, cand.y)])
            cand_poly = cand_line.buffer(1.5)

            is_colliding = False
            for obstacle in obstacles:
                obs_state = get_obstacle_state(obstacle, current_time)
                obs_poly = get_obstacle_polygon(obstacle, obs_state)
                if cand_poly.intersects(obs_poly):
                    is_colliding = True
                    break
            if not is_colliding:
                valid_candidates.append(cand)

        if valid_candidates:
            # Pick the first valid one (Left priority if valid)
            chosen = valid_candidates[0]
            return chosen

        return target_point  # If all fail, return original
