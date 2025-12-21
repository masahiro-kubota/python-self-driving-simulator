import math
from typing import TYPE_CHECKING

import numpy as np

from core.data import LidarConfig, LidarScan, VehicleState

if TYPE_CHECKING:
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

    def scan(self, vehicle_state: VehicleState) -> LidarScan:
        """Perform LiDAR scan using vectorized NumPy operations.

        Args:
            vehicle_state: Current vehicle state

        Returns:
            LidarScan data
        """
        range_max = self.config.range_max
        sensor_x, sensor_y = self._get_sensor_pose(vehicle_state)

        # Vectorized angle computation for all beams
        beam_indices = np.arange(self.config.num_beams)
        angles = (
            self.start_angle + beam_indices * self.increment + vehicle_state.yaw + self.config.yaw
        )

        # Vectorized ray endpoint computation
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        ray_end_x = sensor_x + range_max * cos_angles
        ray_end_y = sensor_y + range_max * sin_angles

        # Initialize ranges with infinity
        ranges = np.full(self.config.num_beams, np.inf, dtype=np.float64)

        # Get map boundaries
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

        # Check map boundaries with vectorized intersection
        ranges = self._update_ranges_from_boundaries(
            ranges, map_boundaries, sensor_x, sensor_y, ray_end_x, ray_end_y
        )

        # Get obstacle polygons
        obstacle_polygons = []
        if self.obstacle_manager is not None:
            from simulator.obstacle import get_obstacle_polygon, get_obstacle_state

            for obs in self.obstacle_manager.obstacles:
                try:
                    st = get_obstacle_state(obs, vehicle_state.timestamp)
                    # Simple distance check before polygon creation
                    dist = math.hypot(st.x - sensor_x, st.y - sensor_y)
                    if dist < range_max + 10.0:  # +10 safety margin
                        poly = get_obstacle_polygon(obs, st)
                        obstacle_polygons.append(poly)
                except Exception:
                    continue

        # Check obstacles with vectorized intersection
        obstacle_boundaries = []
        for poly in obstacle_polygons:
            boundary = poly.boundary
            if boundary.geom_type == "LineString" or boundary.geom_type == "LinearRing":
                obstacle_boundaries.append(boundary)

        ranges = self._update_ranges_from_boundaries(
            ranges, obstacle_boundaries, sensor_x, sensor_y, ray_end_x, ray_end_y
        )

        return LidarScan(
            timestamp=vehicle_state.timestamp, config=self.config, ranges=ranges.tolist()
        )

    def _update_ranges_from_boundaries(
        self,
        ranges: np.ndarray,
        boundaries: list,
        sensor_x: float,
        sensor_y: float,
        ray_end_x: np.ndarray,
        ray_end_y: np.ndarray,
    ) -> np.ndarray:
        """Update sensor ranges by checking intersections with boundaries.

        Args:
            ranges: Current range data
            boundaries: List of boundaries to check
            sensor_x: Sensor x position
            sensor_y: Sensor y position
            ray_end_x: Array of ray endpoint x coordinates
            ray_end_y: Array of ray endpoint y coordinates

        Returns:
            Updated range data
        """
        for boundary in boundaries:
            coords = np.array(boundary.coords)
            if len(coords) < 2:
                continue

            # Process each line segment in the boundary
            for i in range(len(coords) - 1):
                seg_x1, seg_y1 = coords[i]
                seg_x2, seg_y2 = coords[i + 1]

                # Vectorized line-segment intersection
                distances = self._vectorized_ray_segment_intersection(
                    sensor_x, sensor_y, ray_end_x, ray_end_y, seg_x1, seg_y1, seg_x2, seg_y2
                )

                # Update ranges with minimum distances
                valid_mask = (distances >= self.config.range_min) & (distances < ranges)
                ranges = np.where(valid_mask, distances, ranges)

        return ranges

    def _vectorized_ray_segment_intersection(
        self,
        ray_x: float,
        ray_y: float,
        ray_end_x: np.ndarray,
        ray_end_y: np.ndarray,
        seg_x1: float,
        seg_y1: float,
        seg_x2: float,
        seg_y2: float,
    ) -> np.ndarray:
        """Compute ray-segment intersections for all rays at once.

        Uses vectorized line-line intersection algorithm.

        Args:
            ray_x: Ray origin x
            ray_y: Ray origin y
            ray_end_x: Array of ray endpoint x coordinates
            ray_end_y: Array of ray endpoint y coordinates
            seg_x1: Segment start x
            seg_y1: Segment start y
            seg_x2: Segment end x
            seg_y2: Segment end y

        Returns:
            Array of distances (inf if no intersection)
        """
        # Ray direction vectors
        ray_dx = ray_end_x - ray_x
        ray_dy = ray_end_y - ray_y

        # Segment direction vector
        seg_dx = seg_x2 - seg_x1
        seg_dy = seg_y2 - seg_y1

        # Compute determinant (cross product)
        det = ray_dx * seg_dy - ray_dy * seg_dx

        # Parallel rays (det ≈ 0)
        eps = 1e-10
        parallel_mask = np.abs(det) < eps

        # Compute intersection parameters
        dx = seg_x1 - ray_x
        dy = seg_y1 - ray_y

        # Parameter along ray (t) and segment (u)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (dx * seg_dy - dy * seg_dx) / det
            u = (dx * ray_dy - dy * ray_dx) / det

        # Valid intersection: t >= 0 (forward along ray), 0 <= u <= 1 (within segment)
        valid_mask = ~parallel_mask & (t >= 0) & (u >= 0) & (u <= 1)

        # Compute distances
        distances = np.full_like(t, np.inf)
        distances[valid_mask] = t[valid_mask] * np.sqrt(
            ray_dx[valid_mask] ** 2 + ray_dy[valid_mask] ** 2
        )

        return distances

    def _get_sensor_pose(self, vehicle_state: VehicleState) -> tuple[float, float]:
        """Calculate sensor position in global frame."""
        # Vehicle -> Global rotation
        cos_yaw = math.cos(vehicle_state.yaw)
        sin_yaw = math.sin(vehicle_state.yaw)

        # Rotate sensor offset
        rx = self.config.x * cos_yaw - self.config.y * sin_yaw
        ry = self.config.x * sin_yaw + self.config.y * cos_yaw

        # Global position
        gx = vehicle_state.x + rx
        gy = vehicle_state.y + ry

        return gx, gy
