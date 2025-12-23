import math
from typing import TYPE_CHECKING

import numpy as np

from core.data import LidarConfig, LidarScan, VehicleState

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
        """2D Vectorized Ray-Segment Intersection handled in batches for memory efficiency."""
        m_segments = segments.shape[0]

        # Batch segments to stay within reasonable memory limits (N_rays * M_batch elements)
        batch_size = 500
        for i in range(0, m_segments, batch_size):
            m_batch = min(batch_size, m_segments - i)

            seg_a = segments[i : i + m_batch, 0, :]  # [M_batch, 2]
            seg_b = segments[i : i + m_batch, 1, :]  # [M_batch, 2]
            seg_v = seg_b - seg_a  # [M_batch, 2]

            # Determinant: ray_dx * seg_dy - ray_dy * seg_dx
            det = (
                ray_dirs[:, np.newaxis, 0] * seg_v[np.newaxis, :, 1]
                - ray_dirs[:, np.newaxis, 1] * seg_v[np.newaxis, :, 0]
            )

            # Intersection parameters
            dx = sensor_pos[0] - seg_a[:, 0]  # [M_batch]
            dy = sensor_pos[1] - seg_a[:, 1]  # [M_batch]

            # Param u: ( (O_y-A_y)D_x - (O_x-A_x)D_y ) / det
            u_num = (
                dy[np.newaxis, :] * ray_dirs[:, np.newaxis, 0]
                - dx[np.newaxis, :] * ray_dirs[:, np.newaxis, 1]
            )

            # Param t: ( (O_y-A_y)V_x - (O_x-A_x)V_y ) / det
            t_num = (
                dy[np.newaxis, :] * seg_v[np.newaxis, :, 0]
                - dx[np.newaxis, :] * seg_v[np.newaxis, :, 1]
            )

            with np.errstate(divide="ignore", invalid="ignore"):
                u = u_num / det
                t = t_num / det

            # Valid intersection mask
            mask = (
                (np.abs(det) > 1e-10)
                & (t >= self.config.range_min)
                & (t <= self.config.range_max)
                & (u >= 0.0)
                & (u <= 1.0)
            )

            # Find minimum t for each ray in this batch
            batch_min_t = np.where(mask, t, np.inf)
            batch_min = np.min(batch_min_t, axis=1)

            current_ranges = np.minimum(current_ranges, batch_min)

        return current_ranges

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
