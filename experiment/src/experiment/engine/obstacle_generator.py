import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon

from core.utils.osm_parser import parse_osm_file

logger = logging.getLogger(__name__)


class ObstacleGenerator:
    """Generates obstacles based on configuration."""

    def __init__(self, map_path: Path, seed: int | None = None) -> None:
        """Initialize ObstacleGenerator.

        Args:
            map_path: Path to the Lanelet2 OSM map file.
            seed: Random seed.
        """
        # Resolve map path using Hydra utilities if available
        # This handles the case where Hydra changes the working directory
        try:
            import hydra

            self.map_path = Path(hydra.utils.to_absolute_path(str(map_path)))
        except (ImportError, ValueError):
            # Fallback if hydra is not initialized or path issue
            self.map_path = map_path

        self.rng = np.random.default_rng(seed)
        self.centerlines: list[list[tuple[float, float, float]]] = []  # List of (x, y, yaw)
        self.drivable_area: Polygon | None = None
        self._load_map()

    def _load_map(self) -> None:
        """Load and parse the map to extract centerlines and drivable area."""
        if not self.map_path.exists():
            msg = f"Map file not found: {self.map_path}"
            raise FileNotFoundError(msg)

        try:
            from core.utils.osm_parser import parse_osm_for_collision

            self.drivable_area = parse_osm_for_collision(self.map_path)

            osm_data = parse_osm_file(self.map_path)
            nodes = osm_data["nodes"]
            _ways = osm_data["ways"]  # Unused
            lanelets = osm_data["lanelets"]

            for left_nodes, right_nodes in lanelets:
                # Basic centerline calculation
                left_points = [nodes[nid] for nid in left_nodes if nid in nodes]
                right_points = [nodes[nid] for nid in right_nodes if nid in nodes]

                # Ensure same direction (assuming right is reversed in raw data?
                # core.utils.osm_parser.parse_osm_for_collision reverses right for polygon loop.
                # In parse_osm_file, it just returns node IDs.
                # Lanelet2 standard: left and right boundaries are linestrings in same direction.
                # Let's assume consistent direction for now.)

                # Resample or match points to average
                # Simple approach: interpolate along the shorter one
                n_points = min(len(left_points), len(right_points))
                if n_points < 2:
                    continue

                centerline = []
                for i in range(n_points):
                    lx, ly = left_points[i]
                    rx, ry = right_points[i]
                    cx, cy = (lx + rx) / 2.0, (ly + ry) / 2.0

                    # Calculate yaw
                    yaw = 0.0
                    if i < n_points - 1:
                        lx_next, ly_next = left_points[i + 1]
                        rx_next, ry_next = right_points[i + 1]
                        cx_next, cy_next = (lx_next + rx_next) / 2.0, (ly_next + ry_next) / 2.0
                        yaw = math.atan2(cy_next - cy, cx_next - cx)
                    elif i > 0:
                        # Use previous yaw for last point
                        _, _, prev_yaw = centerline[-1]
                        yaw = prev_yaw

                    centerline.append((cx, cy, yaw))

                if centerline:
                    self.centerlines.append(centerline)

        except Exception as e:
            logger.error(f"Failed to parse map for obstacle generation: {e}")

    def generate(self, config: DictConfig) -> list[dict[str, Any]]:
        """Generate obstacles based on the provided configuration.

        Args:
            config: Configuration dictionary (ObstacleGenerationConfig properties).
                    Expected keys: enabled, groups

        Returns:
            List of obstacle dictionaries compatible with SimulatorConfig.
        """
        obstacles: list[dict[str, Any]] = []

        # If generation is disabled or not configured, return empty
        if not config.get("enabled", False):
            return obstacles

        groups = config.get("groups", [])
        for group in groups:
            generated = self._generate_group(group, obstacles)  # Pass existing for collision check
            obstacles.extend(generated)

        return obstacles

    def _generate_group(
        self, group_config: DictConfig, existing_obstacles: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Generate a group of obstacles."""
        generated = []
        count = group_config.get("count", 0)
        placement = group_config.get("placement", {})
        strategy = placement.get("strategy", "random_map")

        obs_type = group_config.get("type", "static")
        shape_config = group_config.get("shape", {"type": "rectangle", "width": 1.0, "length": 1.0})

        for _ in range(count):
            # Attempt to place obstacle
            for attempt in range(100):  # Max attempts
                pose = None
                if strategy == "static_fixed":
                    # Should typically be count=1 for fixed, or a list of poses
                    # If it's a group of fixed obstacles, the config might be structure differently.
                    # But assuming the user might want "fixed" type in the same structure:
                    logger.warning(
                        "static_fixed strategy in generation group is not fully supported for multiple counts. Using default."
                    )
                    pose = (0, 0, 0)  # Placeholder

                elif strategy == "random_map":
                    pose = self._place_random_map(placement)

                elif strategy == "random_track":
                    pose = self._place_random_track(placement)

                else:
                    logger.warning(f"Unknown placement strategy: {strategy}")
                    break

                if pose:
                    x, y, yaw = pose
                    obstacle = {
                        "type": obs_type,
                        "shape": OmegaConf.to_container(shape_config, resolve=True),
                        "position": {"x": x, "y": y, "yaw": yaw},
                    }

                    if self._validate_placement(obstacle, existing_obstacles + generated):
                        generated.append(obstacle)
                        break

        return generated

    def _place_random_map(self, placement_config: DictConfig) -> tuple[float, float, float] | None:
        """Generate a random pose within bounds."""
        bounds = placement_config.get("bounds", {})
        # Default bounds if not specified
        x_min = bounds.get("x_min", -100000)
        x_max = bounds.get("x_max", 100000)
        y_min = bounds.get("y_min", -100000)
        y_max = bounds.get("y_max", 100000)

        x = self.rng.uniform(x_min, x_max)
        y = self.rng.uniform(y_min, y_max)
        yaw = self.rng.uniform(-math.pi, math.pi)

        return (x, y, yaw)

    def _place_random_track(
        self, placement_config: DictConfig
    ) -> tuple[float, float, float] | None:
        """Generate a random pose along the track."""
        if not self.centerlines:
            return None

        # Pick a random centerline
        # Ideally weighted by length, but simple random for now
        centerline_idx = self.rng.integers(0, len(self.centerlines))
        centerline = self.centerlines[centerline_idx]

        if not centerline:
            return None

        # Pick a random point on centerline
        # Linear memory: segment lengths
        total_length = 0.0
        segments = []
        for i in range(len(centerline) - 1):
            p1 = centerline[i]
            p2 = centerline[i + 1]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            segments.append((dist, p1, p2))
            total_length += dist

        if total_length <= 0:
            return None

        target_dist = self.rng.uniform(0, total_length)

        current_dist = 0.0
        chosen_segment = segments[-1]
        for dist, p1, p2 in segments:
            if current_dist + dist >= target_dist:
                chosen_segment = (dist, p1, p2)
                break
            current_dist += dist

        # Interpolate
        seg_len, (x1, y1, yaw1), (x2, y2, yaw2) = chosen_segment
        remaining = target_dist - current_dist
        ratio = remaining / seg_len if seg_len > 0 else 0

        base_x = x1 + ratio * (x2 - x1)
        base_y = y1 + ratio * (y2 - y1)
        # Interpolate yaw properly
        dyaw = yaw2 - yaw1
        while dyaw > math.pi:
            dyaw -= 2 * math.pi
        while dyaw < -math.pi:
            dyaw += 2 * math.pi
        base_yaw = yaw1 + ratio * dyaw

        # Apply offsets
        offset_range = placement_config.get("offset", {"min": -1.0, "max": 1.0})
        lateral_offset = self.rng.uniform(
            offset_range.get("min", -1.0), offset_range.get("max", 1.0)
        )

        # Calculate offset position
        # Normal vector (-sin, cos)
        offset_x = base_x - math.sin(base_yaw) * lateral_offset
        offset_y = base_y + math.cos(base_yaw) * lateral_offset

        # Yaw can be random or aligned
        yaw_mode = placement_config.get("yaw_mode", "aligned")  # aligned, random
        if yaw_mode == "random":
            yaw = self.rng.uniform(-math.pi, math.pi)
        else:
            yaw = base_yaw

        return (offset_x, offset_y, yaw)

    def _validate_placement(
        self, candidate: dict[str, Any], existing: list[dict[str, Any]]
    ) -> bool:
        """Validate if the candidate placement is valid (no collisions, inside map if required)."""
        shape_cfg = candidate["shape"]
        pos = candidate["position"]

        # Construct candidate polygon
        width = shape_cfg.get("width", 2.0)
        length = shape_cfg.get("length", 4.0)

        # Create unrotated box centered at origin
        box = Polygon(
            [
                (-length / 2, -width / 2),
                (length / 2, -width / 2),
                (length / 2, width / 2),
                (-length / 2, width / 2),
            ]
        )

        # Rotate and translate
        box = rotate(box, pos["yaw"], origin=(0, 0), use_radians=True)
        box = translate(box, pos["x"], pos["y"])

        # Check map bounds (if drivable area is available)
        # For random_track, we expect it to be on track, but offsets might push it off.
        # But maybe we wantobstacles to be ON the track.
        if self.drivable_area is not None and not self.drivable_area.intersects(box):
            # If it doesn't even touch drivable area, it's definitely off
            return False

        # Check collision with existing obstacles
        for obs in existing:
            # Similar polygon construction
            o_shape = obs["shape"]
            o_pos = obs["position"]
            o_w = o_shape.get("width", 2.0)
            o_l = o_shape.get("length", 4.0)

            o_box = Polygon(
                [(-o_l / 2, -o_w / 2), (o_l / 2, -o_w / 2), (o_l / 2, o_w / 2), (-o_l / 2, o_w / 2)]
            )
            o_box = rotate(o_box, o_pos["yaw"], origin=(0, 0), use_radians=True)
            o_box = translate(o_box, o_pos["x"], o_pos["y"])

            if box.intersects(o_box):
                return False

        return True
