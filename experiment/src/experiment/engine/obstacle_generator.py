import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from core.utils.osm_parser import parse_osm_file
from omegaconf import DictConfig, OmegaConf
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


class ObstacleGenerator:
    """Generates obstacles based on configuration."""

    def __init__(
        self, map_path: Path, track_path: Path | None = None, seed: int | None = None
    ) -> None:
        """Initialize ObstacleGenerator.

        Args:
            map_path: Path to the Lanelet2 OSM map file.
            map_path: Path to the Lanelet2 OSM map file.
            track_path: Path to the reference track CSV file (optional).
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

        # Resolve track path similarly
        self.track_path = None
        if track_path:
            try:
                import hydra

                self.track_path = Path(hydra.utils.to_absolute_path(str(track_path)))
            except (ImportError, ValueError):
                self.track_path = track_path

        self.rng = np.random.default_rng(seed)
        self.centerlines: list[list[tuple[float, float, float]]] = []  # List of (x, y, yaw)
        self.global_centerline: list[tuple[float, float, float, float]] | None = (
            None  # (x, y, yaw, dist_from_start)
        )
        self.total_track_length: float = 0.0

        self.drivable_area: Polygon | None = None
        self.initial_state: dict[str, float] | None = None
        self.exclusion_zone: dict[str, Any] | None = None
        self._load_map()
        self._load_track()

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

    def _load_track(self) -> None:
        """Load global reference track if available."""
        if not self.track_path or not self.track_path.exists():
            return

        try:
            import pandas as pd

            df = pd.read_csv(self.track_path)
            # Assuming CSV cols: x, y, z, x_quat, y_quat, z_quat, w_quat, speed
            # We need x, y, yaw. Calculate yaw from quat or adjacent points.
            # Usually simulation track CSVs are dense enough.

            path_points = []
            cumulative_dist = 0.0
            prev_x, prev_y = None, None

            for _, row in df.iterrows():
                x, y = row["x"], row["y"]
                # Approximate yaw if not available, or use quat to yaw.
                # Here we just store x, y and calc yaw/dist on the fly or pre-calc.
                # Let's trust the track is ordered.

                # Simple yaw calculation from quaternion (x,y,z,w)
                qx, qy, qz, qw = row["x_quat"], row["y_quat"], row["z_quat"], row["w_quat"]
                # yaw (z-axis rotation) = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
                siny_cosp = 2 * (qw * qz + qx * qy)
                cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                yaw = math.atan2(siny_cosp, cosy_cosp)

                dist = 0.0
                if prev_x is not None:
                    dist = math.hypot(x - prev_x, y - prev_y)

                cumulative_dist += dist
                path_points.append((x, y, yaw, cumulative_dist))

                prev_x, prev_y = x, y

            self.global_centerline = path_points
            self.total_track_length = cumulative_dist
            logger.info(
                f"Loaded global track from {self.track_path}, length={self.total_track_length:.2f}m"
            )

        except Exception as e:
            logger.warning(f"Failed to load global track: {e}")

    def generate(
        self, config: DictConfig, initial_state: dict[str, float] | None = None
    ) -> list[dict[str, Any]]:
        """Generate obstacles based on the provided configuration.

        Args:
            config: Configuration dictionary (ObstacleGenerationConfig properties).
                    Expected keys: enabled, groups
            initial_state: Initial vehicle state (x, y, yaw, velocity).

        Returns:
            List of obstacle dictionaries compatible with SimulatorConfig.
        """
        # Store initial state and exclusion zone config for validation
        self.initial_state = initial_state
        self.exclusion_zone = config.get("exclusion_zone", None)

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

        min_distance = placement.get("min_distance", 0.0)

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
                    if len(pose) == 3:
                        x, y, yaw = pose
                        centerline_info = {}
                    else:
                        x, y, yaw, centerline_info = pose

                    obstacle = {
                        "type": obs_type,
                        "shape": OmegaConf.to_container(shape_config, resolve=True),
                        "position": {
                            "x": x,
                            "y": y,
                            "yaw": yaw,
                            **centerline_info,  # Merge centerline info (index, dist)
                        },
                    }

                    if self._validate_placement(
                        obstacle, existing_obstacles + generated, min_distance
                    ):
                        generated.append(obstacle)
                        break

        return generated

    def _place_random_map(
        self, placement_config: DictConfig
    ) -> tuple[float, float, float] | tuple[float, float, float, dict[str, Any]] | None:
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
    ) -> tuple[float, float, float, dict[str, Any]] | None:
        """Generate a random pose along the track."""
        # Use global centerline if available for consistent distance
        if self.global_centerline:
            target_dist = self.rng.uniform(0, self.total_track_length)

            # Find point on track
            # Binary search or simple scan (optimized for read simplicity)
            # global_centerline is list of (x, y, yaw, dist)

            # Simple linear interpolation
            p1 = self.global_centerline[-1]
            p2 = self.global_centerline[-1]  # fallback

            for i in range(len(self.global_centerline) - 1):
                if self.global_centerline[i + 1][3] >= target_dist:
                    p1 = self.global_centerline[i]
                    p2 = self.global_centerline[i + 1]
                    break

            # Interpolate
            d1, d2 = p1[3], p2[3]
            seg_len = d2 - d1
            ratio = (target_dist - d1) / seg_len if seg_len > 1e-6 else 0

            base_x = p1[0] + ratio * (p2[0] - p1[0])
            base_y = p1[1] + ratio * (p2[1] - p1[1])

            # Yaw interplation
            yaw1, yaw2 = p1[2], p2[2]
            dyaw = yaw2 - yaw1
            while dyaw > math.pi:
                dyaw -= 2 * math.pi
            while dyaw < -math.pi:
                dyaw += 2 * math.pi
            base_yaw = yaw1 + ratio * dyaw

            centerline_idx = -1  # Special ID for global track
            current_dist = target_dist

        elif self.centerlines:
            # Fallback to Lanelet centerlines
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
                p1_c = centerline[i]
                p2_c = centerline[i + 1]
                dist = math.hypot(p2_c[0] - p1_c[0], p2_c[1] - p1_c[1])
                segments.append((dist, p1_c, p2_c))
                total_length += dist

            if total_length <= 0:
                return None

            target_dist = self.rng.uniform(0, total_length)

            current_dist = 0.0
            chosen_segment = segments[-1]
            for dist, p1_c, p2_c in segments:
                if current_dist + dist >= target_dist:
                    chosen_segment = (dist, p1_c, p2_c)
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
        else:
            return None

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

        return (
            float(offset_x),
            float(offset_y),
            float(yaw),
            {"centerline_index": int(centerline_idx), "centerline_dist": float(current_dist)},
        )

    def _validate_placement(
        self,
        candidate: dict[str, Any],
        existing: list[dict[str, Any]],
        min_distance: float = 0.0,
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

        # Check exclusion zone (initial position)
        # Check exclusion zone (initial position)
        if self.exclusion_zone and self.exclusion_zone.get("enabled", False) and self.initial_state:
            exclusion_distance = self.exclusion_zone.get("distance", 10.0)
            init_x = self.initial_state["x"]
            init_y = self.initial_state["y"]

            # Calculate distance from obstacle to initial position
            dist_to_init = math.hypot(pos["x"] - init_x, pos["y"] - init_y)
            if dist_to_init < exclusion_distance:
                return False

        # Validate minimum distance along centerline (if applicable)
        c_dist = candidate["position"].get("centerline_dist")
        c_idx = candidate["position"].get("centerline_index")

        if min_distance > 0.0 and c_dist is not None and c_idx is not None:
            for obs in existing:
                o_pos = obs["position"]
                o_dist = o_pos.get("centerline_dist")
                o_idx = o_pos.get("centerline_index")

                # Only compare if on the same centerline
                if o_dist is not None and o_idx == c_idx:
                    dist_diff = abs(c_dist - o_dist)

                    # Handle lap wrap-around if on global track
                    if c_idx == -1 and self.total_track_length > 0:
                        dist_diff = min(dist_diff, self.total_track_length - dist_diff)

                    if dist_diff < min_distance:
                        return False

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
