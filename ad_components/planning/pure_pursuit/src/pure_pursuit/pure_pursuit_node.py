import math
from pathlib import Path

from pydantic import Field

from core.data import ComponentConfig, SimulatorObstacle, VehicleParameters, VehicleState
from core.data.ad_components import Trajectory, TrajectoryPoint
from core.data.node_io import NodeIO
from core.interfaces.node import Node, NodeExecutionResult
from core.utils.geometry import distance
from simulator.obstacle import get_obstacle_polygon, get_obstacle_state


class PurePursuitConfig(ComponentConfig):
    """Configuration for PurePursuitNode."""

    track_path: Path = Field(..., description="Path to reference trajectory CSV")
    min_lookahead_distance: float = Field(..., description="Minimum lookahead distance [m]")
    max_lookahead_distance: float = Field(..., description="Maximum lookahead distance [m]")
    lookahead_speed_ratio: float = Field(..., description="Lookahead distance speed ratio [s]")
    vehicle_params: VehicleParameters = Field(..., description="Vehicle parameters")


class PurePursuitNode(Node[PurePursuitConfig]):
    """Pure Pursuit path tracking node."""

    def __init__(
        self,
        config: PurePursuitConfig,
        rate_hz: float,
    ):
        super().__init__("PurePursuit", rate_hz, config)
        self.vehicle_params = config.vehicle_params
        self.reference_trajectory: Trajectory | None = None
        # self.config is set by base class

        # Path resolution is handled by node_factory.create_node()
        from planning_utils import load_track_csv

        from core.utils import get_project_root

        track_path = self.config.track_path
        if not track_path.is_absolute():
            track_path = get_project_root() / track_path

        self.reference_trajectory = load_track_csv(track_path)

    def get_node_io(self) -> NodeIO:
        return NodeIO(
            inputs={"vehicle_state": VehicleState, "obstacles": list},
            outputs={"trajectory": Trajectory},
        )

    def on_run(self, current_time: float) -> NodeExecutionResult:
        if self.frame_data is None:
            return NodeExecutionResult.FAILED

        self.current_time = current_time

        # Get Input
        vehicle_state = getattr(self.frame_data, "vehicle_state", None)
        if vehicle_state is None:
            return NodeExecutionResult.SKIPPED

        # Process
        trajectory = self._plan(vehicle_state)

        # Set Output
        self.frame_data.trajectory = trajectory
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

        # Optimization: verify if we can start search from previous nearest index?
        # For now, keep it simple stateless or full search to be safe
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

                # Interpolate yaw (handling wrap-around)
                # diff = p2.yaw - p1.yaw
                # if diff > np.pi: diff -= 2*np.pi
                # elif diff < -np.pi: diff += 2*np.pi
                # target_yaw = p1.yaw + diff * ratio
                # For simplicity in this context, simple linear interp might suffice or use p1's yaw
                # But best to do it right? Or just use p2's yaw to be safe?
                # Using p2's yaw is safer for lookahead. Or just use atan2 of the segment?
                # Segment yaw is better for path following.
                segment_yaw = math.atan2(p2.y - p1.y, p2.x - p1.x)
                target_yaw = segment_yaw

                target_point = TrajectoryPoint(
                    x=target_x, y=target_y, yaw=target_yaw, velocity=target_v
                )
                check_points.append(target_point)
                break

            accumulated_dist += d
            current_idx += 1
            # Add intermediate points to check list
            check_points.append(self.reference_trajectory[current_idx])
            target_point = self.reference_trajectory[current_idx]

        # Obstacle Avoidance
        obstacles = getattr(self.frame_data, "obstacles", [])
        if obstacles:
            target_point = self._avoid_obstacles(
                target_point, obstacles, self.current_time, vehicle_state
            )

        return Trajectory(points=[target_point])

    def _avoid_obstacles(
        self,
        target_point: TrajectoryPoint,
        obstacles: list[SimulatorObstacle],
        current_time: float,
        vehicle_state: VehicleState,
    ) -> TrajectoryPoint:
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

                cand_p = TrajectoryPoint(
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
