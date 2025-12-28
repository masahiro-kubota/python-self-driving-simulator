import math
from dataclasses import dataclass

from core.data.ad_components import VehicleState
from core.data.environment.obstacle import Obstacle

from lateral_shift_planner.frenet_converter import FrenetConverter


@dataclass
class TargetObstacle:
    """Obstacle filtered for avoidance."""

    id: str
    s: float
    lat: float
    length: float
    width: float
    left_boundary_dist: float  # Distance from obstacle to left road boundary [m]
    right_boundary_dist: float  # Distance from obstacle to right road boundary [m]
    raw: Obstacle | None = None


class ObstacleManager:
    """Manages obstacle filtering and projection."""

    def __init__(
        self,
        converter: FrenetConverter,
        road_map,  # RoadWidthMap instance
        lookahead_distance: float,
        lookbehind_distance: float,
        vehicle_width: float,
        safe_margin: float,
    ):
        """Initialize ObstacleManager.

        Args:
            converter: FrenetConverter instance
            road_map: RoadWidthMap instance for boundary calculations
            lookahead_distance: Max distance to consider obstacles ahead [m]
            lookbehind_distance: Max distance to consider obstacles behind [m]
            vehicle_width: Ego vehicle width [m]
            safe_margin: Safety margin [m]
        """
        self.converter = converter
        self.road_map = road_map
        self.lookahead = lookahead_distance
        self.lookbehind = lookbehind_distance
        self.vehicle_width = vehicle_width
        self.safe_margin = safe_margin

    def get_target_obstacles(
        self, ego_state: VehicleState, obstacles: list[Obstacle]
    ) -> list[TargetObstacle]:
        """Convert obstacles to Frenet frame and filter.

        Args:
            ego_state: Current vehicle state
            obstacles: List of detected obstacles

        Returns:
            List of TargetObstacle
        """
        targets = []

        # Ego position in Frenet
        s_ego, _l_ego = self.converter.global_to_frenet(ego_state.x, ego_state.y)

        # Get road boundaries at ego position for lateral filtering
        ego_boundaries = self.road_map.get_lateral_boundaries(ego_state.x, ego_state.y)
        if ego_boundaries is not None:
            ego_left_bound, ego_right_bound = ego_boundaries
        else:
            # Fallback: try to get width or use default
            width = self.road_map.get_lateral_width(ego_state.x, ego_state.y)
            if width is not None:
                ego_left_bound = width / 2.0
                ego_right_bound = width / 2.0
            else:
                # Last resort fallback if map completely fails
                ego_left_bound = 3.0
                ego_right_bound = 3.0

        for obs in obstacles:
            # Convert to Frenet
            s_obj, l_obj = self.converter.global_to_frenet(obs.x, obs.y)

            # 1. Distance check (forward and backward)
            dist = s_obj - s_ego

            # Check if obstacle is within detection range (forward or backward)
            if dist > 0:
                # Forward obstacle
                if dist > self.lookahead:
                    continue
            else:
                # Backward obstacle
                if abs(dist) > self.lookbehind:
                    continue

            # 3. Lateral boundary check (filter out obstacles in adjacent lanelets)
            # Check if obstacle is within the road boundaries at ego position
            if l_obj > ego_left_bound or l_obj < -ego_right_bound:
                # Obstacle is outside the ego lanelet boundaries
                import logging

                logger = logging.getLogger(__name__)
                logger.info(
                    f"[ObstacleManager] Filtering out obstacle at s={s_obj:.2f}, l={l_obj:.2f} "
                    f"(outside ego boundaries: L={ego_left_bound:.2f}, R={ego_right_bound:.2f})"
                )
                continue

            # Map dimensions with YAW consideration
            # Calculate Frenet Bounding Box
            # 1. Get 4 corners in Global Frame
            # Obstacle has width (lateral) and height (longitudinal)
            # Local frame: x-axis is forward (height/length), y-axis is lateral (width)
            half_length = obs.height / 2.0
            half_width = obs.width / 2.0

            corners_local = [
                (half_length, half_width),
                (half_length, -half_width),
                (-half_length, -half_width),
                (-half_length, half_width),
            ]

            ct = math.cos(obs.yaw)
            st = math.sin(obs.yaw)

            s_vals = []
            l_vals = []

            for cx, cy in corners_local:
                # Rotate
                gx = cx * ct - cy * st + obs.x
                gy = cx * st + cy * ct + obs.y

                # Convert to Frenet
                cs, cl = self.converter.global_to_frenet(gx, gy)
                s_vals.append(cs)
                l_vals.append(cl)

            # 4. Determine Frenet Bounding Box
            s_min = min(s_vals)
            s_max = max(s_vals)
            l_min = min(l_vals)
            l_max = max(l_vals)

            # 5. Calculate effective dimensions in Frenet
            length_frenet = s_max - s_min
            width_frenet = l_max - l_min
            s_center_frenet = (s_min + s_max) / 2.0
            l_center_frenet = (l_min + l_max) / 2.0

            # Get road boundaries at obstacle position
            # Use obstacle's global position and yaw from centerline
            # Get position at obstacle's s position from centerline
            obs_global_x, obs_global_y = self.converter.frenet_to_global(
                s_center_frenet, l_center_frenet
            )

            boundaries = self.road_map.get_lateral_boundaries(obs_global_x, obs_global_y)
            if boundaries is not None:
                left_boundary_dist, right_boundary_dist = boundaries
            else:
                # Fallback: try to get width or use default
                width = self.road_map.get_lateral_width(obs_global_x, obs_global_y)
                if width is not None:
                    left_boundary_dist = width / 2.0
                    right_boundary_dist = width / 2.0
                else:
                    # Last resort fallback
                    left_boundary_dist = 3.0
                    right_boundary_dist = 3.0

            targets.append(
                TargetObstacle(
                    id=obs.id,
                    s=s_center_frenet,
                    lat=l_center_frenet,
                    length=length_frenet,
                    width=width_frenet,
                    left_boundary_dist=left_boundary_dist,
                    right_boundary_dist=right_boundary_dist,
                    raw=obs,
                )
            )

        # Sort by absolute distance from ego (closest first, regardless of direction)
        targets.sort(key=lambda o: abs(o.s - s_ego))

        if len(targets) > 0:
            import logging

            logger = logging.getLogger(__name__)
            logger.info(f"[ObstacleManager] Found {len(targets)} targets:")
            for t in targets:
                logger.info(
                    f"  ID={t.id} s={t.s:.2f} l={t.lat:.2f} w={t.width:.2f} l_raw={t.raw.x:.1f},{t.raw.y:.1f}"
                )

        return targets
