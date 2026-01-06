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

        self.initial_state: dict[str, float] | None = None
        self.exclusion_zone: dict[str, Any] | None = None
        
        from experiment.engine.pose_sampler import PoseSampler
        self.pose_sampler = PoseSampler(self.map_path, self.track_path, seed)
        
        # Expose properties for legacy access if needed (or minimal support)
        self.drivable_area = self.pose_sampler.drivable_area
        self.global_centerline = self.pose_sampler.global_centerline
        self.total_track_length = self.pose_sampler.total_track_length
        
        self.centerlines: list[list[tuple[float, float, float]]] = [] 
        self._load_lanelets_centerlines() # Fallback for non-global track usage

    def _load_lanelets_centerlines(self) -> None:
        """Load lanelet centerlines for fallback."""
        if not self.map_path.exists():
            return

        try:
            from core.utils.osm_parser import parse_osm_file
            
            # Note: drivable_area is loaded by PoseSampler
            
            osm_data = parse_osm_file(self.map_path)
            nodes = osm_data["nodes"]
            lanelets = osm_data["lanelets"]

            for left_nodes, right_nodes in lanelets:
                # Basic centerline calculation
                left_points = [nodes[nid] for nid in left_nodes if nid in nodes]
                right_points = [nodes[nid] for nid in right_nodes if nid in nodes]

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
            logger.error(f"Failed to parse map for lanelets: {e}")

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
        require_within_bounds = placement.get("require_within_bounds", False)

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

                elif strategy == "track_forward":
                    pose = self._place_track_forward(placement)

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
                        obstacle, 
                        existing_obstacles + generated, 
                        min_distance,
                        require_within_bounds
                    ):
                        generated.append(obstacle)
                        break
                else:
                    logger.warning(
                        f"Failed to place obstacle {len(generated)+1} for group '{group_config.get('name')}' after 100 attempts."
                    )

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
        lateral_offset_range = placement_config.get("lateral_offset_range", [-1.0, 1.0])
        yaw_mode = placement_config.get("yaw_mode", "aligned")
        
        yaw_offset_dict = placement_config.get("yaw_offset_range", [0.0, 0.0])
        # Convert list/tuple if needed (config handles list from yaml)
        if isinstance(yaw_offset_dict, (list, tuple)):
            yaw_offset_range = (yaw_offset_dict[0], yaw_offset_dict[1])
        else:
            # Fallback for dict likely not needed due to schema enforcement but defensive
            yaw_offset_range = (0.0, 0.0)

        if self.pose_sampler.global_centerline:
             return self.pose_sampler.sample_track_pose(
                 lateral_offset_range=(lateral_offset_range[0], lateral_offset_range[1]),
                 yaw_mode=yaw_mode,
                 yaw_offset_range=yaw_offset_range,
             )
        
        logger.warning("No global track loaded. Cannot use random_track strategy.")
        return None

    def _place_track_forward(
        self, placement_config: DictConfig
    ) -> tuple[float, float, float, dict[str, Any]] | None:
        """Generate a pose a fixed distance ahead of initial position along the track."""
        if not self.initial_state or not self.pose_sampler.global_centerline:
            logger.warning("track_forward strategy requires initial_state and global track.")
            return None

        # Find closest point on global centerline to initial state
        init_x = self.initial_state["x"]
        init_y = self.initial_state["y"]
        
        # We need current_dist of the initial state. 
        # PoseSampler doesn't have "get_dist_from_point" helper yet, 
        # so lets implement strict logic here or add helper to PoseSampler later.
        # But reusing the previous linear search logic is okay for now, or assume 
        # initial state *is* on track if we knew. 
        # For robustness, we search. 
        
        # Optimization: Reuse global_centerline search.
        # Note: existing code did this search efficiently.
        
        # Let's add a helper to PoseSampler later? 
        # For now, quick search locally or just reimplement using pose_sampler data.
        
        # Search closest
        closest_idx = -1
        min_dist_sq = float("inf")
        
        for i, pt in enumerate(self.pose_sampler.global_centerline):
            dx = pt[0] - init_x
            dy = pt[1] - init_y
            d2 = dx * dx + dy * dy
            if d2 < min_dist_sq:
                min_dist_sq = d2
                closest_idx = i
                
        if closest_idx == -1: return None
        
        current_dist = self.pose_sampler.global_centerline[closest_idx][3]
        
        # Forward distance
        forward_distance_range = placement_config.get("forward_distance_range", None)
        if forward_distance_range is not None:
            forward_dist = self.rng.uniform(forward_distance_range[0], forward_distance_range[1])
        else:
            forward_dist = placement_config.get("forward_distance", 3.0)
            
        target_dist = current_dist + forward_dist
        
        lateral_offset_range = placement_config.get("lateral_offset_range", [0.0, 0.0])
        yaw_mode = placement_config.get("yaw_mode", "aligned")
        
        yaw_offset_dict = placement_config.get("yaw_offset_range", [0.0, 0.0])
        if isinstance(yaw_offset_dict, (list, tuple)):
            yaw_offset_range = (yaw_offset_dict[0], yaw_offset_dict[1])
        else:
             yaw_offset_range = (0.0, 0.0)

        return self.pose_sampler.sample_track_pose(
            target_dist=target_dist,
            lateral_offset_range=(lateral_offset_range[0], lateral_offset_range[1]),
            yaw_mode=yaw_mode,
            yaw_offset_range=yaw_offset_range,
        )

    def _validate_placement(
        self,
        candidate: dict[str, Any],
        existing: list[dict[str, Any]],
        min_distance: float = 0.0,
        require_within_bounds: bool = False,
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

        if min_distance > 0.0 and c_dist is not None:
            for obs in existing:
                o_pos = obs["position"]
                o_dist = o_pos.get("centerline_dist")
                o_idx = o_pos.get("centerline_index")

                # Checks if obstacles are on the same track reference.
                # If centerline_index is present (Lanelet), they must match.
                # If centerline_index is None (PoseSampler/Global), we compare if both are None (or assume single global track).
                indices_match = (o_idx == c_idx)
                
                # If one is None and other is not, we technically shouldn't compare, 
                # but currently we only mix mostly uniform obstacles.
                # Safest is to compare if equal. (None == None is True).

                if o_dist is not None and indices_match:
                    dist_diff = abs(c_dist - o_dist)

                    # Handle lap wrap-around if on global track (c_idx is None implies global via PoseSampler usually, 
                    # or c_idx == -1 for legacy).
                    # Check invalid/None index or explicit -1 convention
                    if (c_idx is None or c_idx == -1) and self.total_track_length > 0:
                        dist_diff = min(dist_diff, self.total_track_length - dist_diff)

                    if dist_diff < min_distance:
                        return False

        # Check map bounds using PoseSampler
        if not self.pose_sampler.validate_pose(
            (pos["x"], pos["y"], pos["yaw"]), 
            shape=shape_cfg, 
            require_fully_contained=require_within_bounds if require_within_bounds else False # intersects if false
        ):
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
