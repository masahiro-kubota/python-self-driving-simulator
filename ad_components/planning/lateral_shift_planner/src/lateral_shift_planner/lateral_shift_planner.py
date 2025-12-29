import logging
from dataclasses import dataclass

import numpy as np
from core.data import VehicleState
from core.data.environment.obstacle import Obstacle
from planning_utils.types import ReferencePath, ReferencePathPoint

from lateral_shift_planner.frenet_converter import FrenetConverter
from lateral_shift_planner.obstacle_manager import ObstacleManager, TargetObstacle
from lateral_shift_planner.shift_profile import ShiftProfile, merge_profiles

logger = logging.getLogger(__name__)


@dataclass
class LateralShiftPlannerConfig:
    """Config for LateralShiftPlanner."""

    # Perception / Planning Horizon
    lookahead_distance: float
    lookbehind_distance: float
    avoidance_maneuver_length: float

    # Margins and Dimensions
    longitudinal_margin_front: float
    longitudinal_margin_rear: float
    vehicle_width: float
    safe_margin: float

    # Trajectory Generation
    trajectory_resolution: float


@dataclass
class AvoidanceDebugData:
    """Debug data for visualization."""

    trajectory: ReferencePath
    target_obstacles: list[TargetObstacle]
    shift_profiles: list[ShiftProfile]
    merged_lat: np.ndarray
    s_samples: np.ndarray
    merged_profile: list
    collision_detected: bool


class LateralShiftPlanner:
    """Planning logic for static avoidance using lateral shift."""

    def __init__(self, ref_path: np.ndarray, config: LateralShiftPlannerConfig, road_map):
        """Initialize LateralShiftPlanner.

        Args:
            ref_path: Reference path (centerline) [N, 4] (x, y, yaw, velocity)
            config: Configuration
            road_map: RoadWidthMap instance for boundary calculations
        """
        self.ref_path = ref_path
        self.config = config
        self.converter = FrenetConverter(ref_path)
        self.obstacle_manager = ObstacleManager(
            converter=self.converter,
            road_map=road_map,
            lookahead_distance=config.lookahead_distance,
            lookbehind_distance=config.lookbehind_distance,
            vehicle_width=config.vehicle_width,
            safe_margin=config.safe_margin,
        )
        self.logger = logging.getLogger(__name__)

    def plan(
        self,
        ego_state: VehicleState,
        obstacles: list[Obstacle],
        _road_width: float = 6.0,
    ) -> AvoidanceDebugData:
        """Generate avoidance trajectory.

        Args:
            ego_state: Current vehicle state
            obstacles: List of obstacles
            _road_width: Road width [m] (fallback)
        """

        # 1. Get Target Obstacles (Frenet)
        targets = self.obstacle_manager.get_target_obstacles(ego_state, obstacles)

        # 2. Get Ego Frenet State
        s_ego, l_ego = self.converter.global_to_frenet(ego_state.x, ego_state.y)

        self.logger.info(f"[LSP] s_ego={s_ego:.2f}, l_ego={l_ego:.2f}, targets={len(targets)}")

        # 3. Define s range for planning
        s_end = s_ego + self.config.lookahead_distance
        s_samples = np.arange(s_ego, s_end, self.config.trajectory_resolution)

        if len(s_samples) == 0:
            # Should not happen typically
            return ReferencePath(points=[])

        if len(targets) == 0:
            self.logger.info(f"[LSP] s_ego={s_ego:.2f}, l_ego={l_ego:.2f}, targets=0")
            # No obstacles: Plan centerline (all zeros)
            lat_target_array = np.zeros_like(s_samples)
            collision_detected = False
            profiles = []
        else:
            # 4. Create Shift Profiles for each obstacle
            profiles: list[ShiftProfile] = []
            for target in targets:
                profile = ShiftProfile(
                    obstacle=target,
                    longitudinal_margin_front=self.config.longitudinal_margin_front,
                    longitudinal_margin_rear=self.config.longitudinal_margin_rear,
                    avoidance_maneuver_length=self.config.avoidance_maneuver_length,
                    safe_margin=self.config.safe_margin,
                    vehicle_width=self.config.vehicle_width,
                )
                profiles.append(profile)

            # 5. Merge Profiles (take max shift)
            lat_target_array, collision_detected = merge_profiles(s_samples, profiles)

        if len(profiles) > 0:
            self.logger.info(
                f"[LSP] Merged {len(profiles)} profiles. Collision={collision_detected}"
            )
            pass

        if collision_detected:
            self.logger.info("[LSP] COLLISION DETECTED in merge!")
            # TODO: Handle collision (e.g. stop).
            # Current simplest behavior: stop or follow centerline?
            # planning.md says "Yield (Stop)".
            # If stopping, we should output a trajectory that stops.
            # For now, let's output centerline but maybe with 0 velocity?
            # Or just best effort centerline.
            # We can log a warning or set velocity to 0.
            pass

        # 6. Generate Global Trajectory
        trajectory_points = []
        for i, s in enumerate(s_samples):
            lat = lat_target_array[i]

            # Smooth transition from current ego lateral position?
            # If s is close to s_ego, we might be at l_ego != l_target[0].
            # The current logic plans from current s forward.
            # If l_target[0] jumps, the path jumps.
            # Ideally we should interpolate from l_ego to l_target over some distance.
            # But ShiftProfile handles ramping.
            # However, ShiftProfile is relative to Obstacles.
            # If no obstacles, l_target is 0.
            # If ego is at l=1 (drifted), and plans 0, it jumps.
            # We should probably filter the output or start from ego position.

            # Simple approach: The converter returns global coords.
            # Controller handles tracking error.
            # But if we output a path starting 1m away, the controller might react harshly.
            # Better to assume the planner outputs the *desired* path, and controller converges.

            x, y = self.converter.frenet_to_global(s, lat)

            # Target speed
            # Detailed speed planning is listed as future work (Expansion).
            # Simple constant speed for now? Or maintain current?
            # Or use reference speed from map?
            # Reference map has velocity.
            # We need to find velocity at s.
            # Assuming constant for now or look up.

            # Look up reference velocity at s
            # FrenetConverter doesn't expose it directly, but we can search.
            # For efficiency let's use a default or nearest.
            v_ref = 10.0  # Default
            # TODO: Look up v_ref from ref_path

            # Use yaw from frenet_to_global implicitly?
            # frenet_to_global output x, y. Yaw needs to be calculated from dx, dy of the generated path.

            trajectory_points.append(
                ReferencePathPoint(x=x, y=y, yaw=0.0, velocity=v_ref)
            )  # yaw updated later

        if len(s_samples) > 0:
            min_l = np.min(lat_target_array)
            max_l = np.max(lat_target_array)
            if abs(min_l) > 0.01 or abs(max_l) > 0.01:
                logger.info(f"[LSP] Planned Path Range L: [{min_l:.2f}, {max_l:.2f}]")

        # Calculate yaw from points
        for i in range(len(trajectory_points)):
            if i < len(trajectory_points) - 1:
                dx = trajectory_points[i + 1].x - trajectory_points[i].x
                dy = trajectory_points[i + 1].y - trajectory_points[i].y
                yaw = np.arctan2(dy, dx)
                trajectory_points[i].yaw = yaw
            else:
                trajectory_points[i].yaw = trajectory_points[i - 1].yaw

        return AvoidanceDebugData(
            trajectory=ReferencePath(points=trajectory_points),
            target_obstacles=targets,
            shift_profiles=profiles,
            merged_lat=lat_target_array,
            s_samples=s_samples,
            merged_profile=lat_target_array.tolist(),
            collision_detected=collision_detected,
        )
