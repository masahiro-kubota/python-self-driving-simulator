import math
from typing import TYPE_CHECKING

import numpy as np
from core.data import LidarConfig, LidarScan, VehicleState
from numba import jit

if TYPE_CHECKING:
    from core.data import LidarConfig, LidarScan, VehicleState

    from simulator.map import LaneletMap
    from simulator.obstacle import ObstacleManager


class LidarSensor:
    """LiDAR sensor simulation."""

    def __init__(
        self,
        config: LidarConfig,
        map_instance: "LaneletMap | None" = None,
        obstacle_manager: "ObstacleManager | None" = None,
    ) -> None:
        """Initialize LiDAR sensor.

        Args:
            config: Lidar configuration
            map_instance: LaneletMap instance for map boundary raycasting
            obstacle_manager: ObstacleManager for obstacle raycasting
        """
        self.config = config
        self.map = map_instance
        self.obstacle_manager = obstacle_manager

        # Precompute angles
        fov_rad = math.radians(self.config.fov)
        if self.config.angle_increment > 0:
            self.increment = self.config.angle_increment
        else:
            self.increment = fov_rad / self.config.num_beams

        self.start_angle = -fov_rad / 2.0

        # Cache for map segments [M, 2, 2]
        self._cached_map_segments: np.ndarray | None = None

    def scan(self, vehicle_state: VehicleState) -> LidarScan:
        """Perform LiDAR scan using 2D vectorized NumPy operations and spatial culling.

        Args:
            vehicle_state: Current vehicle state

        Returns:
            LidarScan data
        """
        sensor_x, sensor_y = self._get_sensor_pose(vehicle_state)
        sensor_pos = np.array([sensor_x, sensor_y], dtype=np.float64)

        # 1. Vectorized angle and direction computation
        beam_indices = np.arange(self.config.num_beams)
        angles = (
            self.start_angle + beam_indices * self.increment + vehicle_state.yaw + self.config.yaw
        )
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        # Ray directions [N, 2]
        ray_dirs = np.stack([cos_angles, sin_angles], axis=1)

        # Initialize ranges with infinity
        ranges = np.full(self.config.num_beams, np.inf, dtype=np.float64)

        # 2. Collect and Cull Segments (Map + Obstacles)
        all_segments = self._get_culled_segments(sensor_pos, vehicle_state)

        if len(all_segments) > 0:
            # 3. 2D Vectorized Ray-Segment Intersection (Narrow-phase)
            ranges = self._batch_ray_segment_intersection(
                sensor_pos, ray_dirs, all_segments, ranges
            )

        return LidarScan(
            timestamp=vehicle_state.timestamp, config=self.config, ranges=ranges.tolist()
        )

    def _get_culled_segments(
        self, sensor_pos: np.ndarray, vehicle_state: VehicleState
    ) -> np.ndarray:
        """Collect nearby segments from map and obstacles.

        Returns:
            NumPy array of segments [M, 2 (start/end), 2 (x/y)]
        """
        # A. Map Segments (Cached)
        if self._cached_map_segments is None and self.map is not None:
            self._cache_map_segments()

        segments_list = []
        if self._cached_map_segments is not None:
            # BROAD-PHASE CULLING: Reject segments far from sensor
            # Calculate distance from sensor to each segment
            seg_a = self._cached_map_segments[:, 0, :]
            seg_b = self._cached_map_segments[:, 1, :]
            v = seg_b - seg_a
            w = sensor_pos - seg_a

            # Vectorized Point-to-Segment distance
            v_norm_sq = np.sum(v**2, axis=1)
            v_norm_sq[v_norm_sq == 0] = 1.0  # avoid div by zero
            proj = np.sum(w * v, axis=1) / v_norm_sq
            proj = np.clip(proj, 0.0, 1.0)
            closest_points = seg_a + proj[:, np.newaxis] * v
            dist_sq = np.sum((sensor_pos - closest_points) ** 2, axis=1)

            mask = dist_sq < (self.config.range_max + 1.0) ** 2
            segments_list.append(self._cached_map_segments[mask])

        # B. Obstacle Segments
        obstacle_boundaries = self._get_obstacle_boundaries(
            vehicle_state, sensor_pos[0], sensor_pos[1]
        )
        for boundary in obstacle_boundaries:
            coords = np.array(boundary.coords)
            if len(coords) < 2:
                continue
            segs = np.stack([coords[:-1], coords[1:]], axis=1)
            segments_list.append(segs)

        if not segments_list:
            return np.empty((0, 2, 2), dtype=np.float64)

        return np.concatenate(segments_list, axis=0)

    def _cache_map_segments(self) -> None:
        """Extract and cache all segments from the map for fast vectorized access."""
        map_boundaries = self._get_map_boundaries()
        all_coords = []
        for boundary in map_boundaries:
            coords = np.array(boundary.coords)
            if len(coords) < 2:
                continue
            segs = np.stack([coords[:-1], coords[1:]], axis=1)
            all_coords.append(segs)

        if all_coords:
            self._cached_map_segments = np.concatenate(all_coords, axis=0)
        else:
            self._cached_map_segments = np.empty((0, 2, 2), dtype=np.float64)

    def _batch_ray_segment_intersection(
        self,
        sensor_pos: np.ndarray,
        ray_dirs: np.ndarray,
        segments: np.ndarray,
        current_ranges: np.ndarray,
    ) -> np.ndarray:
        """2D Vectorized Ray-Segment Intersection using Numba JIT."""
        return _numba_intersection_kernel(
            sensor_pos,
            ray_dirs,
            segments,
            current_ranges,
            self.config.range_min,
            self.config.range_max,
        )

    def _get_map_boundaries(self) -> list:
        """Extract boundaries from the map."""
        map_boundaries = []
        if (
            self.map is not None
            and self.map.drivable_area is not None
            and hasattr(self.map.drivable_area, "boundary")
        ):
            boundary = self.map.drivable_area.boundary
            if boundary.geom_type == "LineString":
                map_boundaries.append(boundary)
            elif boundary.geom_type == "MultiLineString":
                map_boundaries.extend(boundary.geoms)
            elif boundary.geom_type == "LinearRing":
                map_boundaries.append(boundary)
        return map_boundaries

    def _get_obstacle_boundaries(
        self, vehicle_state: "VehicleState", sensor_x: float, sensor_y: float
    ) -> list:
        """Extract boundaries from nearby obstacles."""
        from simulator.obstacle import get_obstacle_polygon, get_obstacle_state

        obstacle_boundaries = []
        if self.obstacle_manager is not None:
            range_max = self.config.range_max
            for obs in self.obstacle_manager.obstacles:
                try:
                    st = get_obstacle_state(obs, vehicle_state.timestamp)
                    dist = math.hypot(st.x - sensor_x, st.y - sensor_y)
                    if dist < range_max + 10.0:
                        poly = get_obstacle_polygon(obs, st)
                        boundary = poly.boundary
                        if boundary.geom_type in ("LineString", "LinearRing"):
                            obstacle_boundaries.append(boundary)
                        elif boundary.geom_type == "MultiLineString":
                            obstacle_boundaries.extend(boundary.geoms)
                except Exception:
                    continue
        return obstacle_boundaries

    def _get_sensor_pose(self, vehicle_state: VehicleState) -> tuple[float, float]:
        """Calculate sensor position in global frame."""
        cos_yaw = math.cos(vehicle_state.yaw)
        sin_yaw = math.sin(vehicle_state.yaw)
        rx = self.config.x * cos_yaw - self.config.y * sin_yaw
        ry = self.config.x * sin_yaw + self.config.y * cos_yaw
        gx = vehicle_state.x + rx
        gy = vehicle_state.y + ry
        return gx, gy


@jit(nopython=True, cache=True)
def _numba_intersection_kernel(
    sensor_pos: np.ndarray,
    ray_dirs: np.ndarray,
    segments: np.ndarray,
    ranges: np.ndarray,
    range_min: float,
    range_max: float,
) -> np.ndarray:
    """JIT-compiled kernel for ray-segment intersection.

    Args:
        sensor_pos: [2] (x, y)
        ray_dirs: [N, 2] (cos, sin)
        segments: [M, 2, 2] (start/end, x/y)
        ranges: [N] Current ranges (modified in-place or returned)
        range_min: Minimum valid range
        range_max: Maximum valid range
    """
    n_rays = ray_dirs.shape[0]
    m_segments = segments.shape[0]

    sensor_x = sensor_pos[0]
    sensor_y = sensor_pos[1]

    # Loop over rays
    for i in range(n_rays):
        ray_dx = ray_dirs[i, 0]
        ray_dy = ray_dirs[i, 1]

        min_dist = ranges[i]  # Start with current best

        # Loop over segments
        for j in range(m_segments):
            p1_x = segments[j, 0, 0]
            p1_y = segments[j, 0, 1]
            p2_x = segments[j, 1, 0]
            p2_y = segments[j, 1, 1]

            # Segment vector
            seg_dx = p2_x - p1_x
            seg_dy = p2_y - p1_y

            # Cross product (Determinant)
            det = ray_dx * seg_dy - ray_dy * seg_dx

            # Parallel check
            if abs(det) < 1e-10:
                continue

            # Calculate relative position (Sensor - SegmentStart)
            # Matches NumPy implementation: dx = sensor_pos[0] - seg_a[:, 0]
            dx = sensor_x - p1_x
            dy = sensor_y - p1_y

            # Intersection parameters
            u_num = dy * ray_dx - dx * ray_dy
            t_num = dy * seg_dx - dx * seg_dy

            u = u_num / det
            t = t_num / det

            # Validity check
            # Validity check
            if t >= range_min and t <= range_max and u >= 0.0 and u <= 1.0 and t < min_dist:
                min_dist = t

        ranges[i] = min_dist

    return ranges
