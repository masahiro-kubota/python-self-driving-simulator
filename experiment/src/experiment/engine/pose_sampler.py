import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon

try:
    from simulator.map import LaneletMap
except ImportError:
    # Allow usage without simulator package if needed, though map loading depends on it usually
    LaneletMap = Any

logger = logging.getLogger(__name__)


class PoseSampler:
    """Shared logic for sampling poses from track or map."""

    def __init__(
        self, map_path: Path, track_path: Path | None = None, seed: int | None = None
    ) -> None:
        """Initialize PoseSampler.

        Args:
            map_path: Path to the Lanelet2 OSM map file.
            track_path: Path to the reference track CSV file (optional).
            seed: Random seed.
        """
        # Resolve paths using Hydra if available
        self.map_path = self._resolve_path(map_path)
        self.track_path = self._resolve_path(track_path) if track_path else None

        self.rng = np.random.default_rng(seed)
        
        # Load Map
        self.drivable_area: Polygon | None = None
        self._load_map()

        # Load Track
        self.global_centerline: list[tuple[float, float, float, float]] | None = None
        self.total_track_length: float = 0.0
        self._load_track()

    def _resolve_path(self, path: Path) -> Path:
        try:
            import hydra
            return Path(hydra.utils.to_absolute_path(str(path)))
        except (ImportError, ValueError):
            return path

    def _load_map(self) -> None:
        if not self.map_path.exists():
            raise FileNotFoundError(f"Map file not found: {self.map_path}")
        
        try:
            from core.utils.osm_parser import parse_osm_for_collision
            self.drivable_area = parse_osm_for_collision(self.map_path)
            # LaneletMap instance could be useful if we want to use its methods, 
            # but usually just polygon check is enough here.
            # If we need strictly LaneletMap wrapper:
            # self.lanelet_map = LaneletMap(self.map_path)
        except Exception as e:
            logger.error(f"Failed to parse map: {e}")

    def _load_track(self) -> None:
        if not self.track_path or not self.track_path.exists():
            return

        try:
            df = pd.read_csv(self.track_path)
            
            path_points = []
            cumulative_dist = 0.0
            prev_x, prev_y = None, None

            for _, row in df.iterrows():
                x, y = row["x"], row["y"]
                
                # Calculate yaw from quaternion
                if "x_quat" in row:
                    qx, qy, qz, qw = row["x_quat"], row["y_quat"], row["z_quat"], row["w_quat"]
                    siny_cosp = 2 * (qw * qz + qx * qy)
                    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                    yaw = math.atan2(siny_cosp, cosy_cosp)
                elif "yaw" in row:
                    yaw = row["yaw"]
                else:
                    yaw = 0.0 # Should calc from points if missing, but assuming quat or yaw exists

                dist = 0.0
                if prev_x is not None:
                    dist = math.hypot(x - prev_x, y - prev_y)

                cumulative_dist += dist
                path_points.append((x, y, yaw, cumulative_dist))

                prev_x, prev_y = x, y

            self.global_centerline = path_points
            self.total_track_length = cumulative_dist
            logger.info(f"Loaded global track, length={self.total_track_length:.2f}m")

        except Exception as e:
            logger.warning(f"Failed to load track: {e}")

    def sample_track_pose(
        self,
        target_dist: float | None = None,
        lateral_offset_range: tuple[float, float] = (0.0, 0.0),
        yaw_mode: str = "aligned", # aligned, random
        yaw_offset_range: tuple[float, float] = (0.0, 0.0), # Only used if aligned
    ) -> tuple[float, float, float, dict[str, Any]] | None:
        """Sample a pose on the track.

        Args:
            target_dist: Specific distance along track. If None, random sampling.
            lateral_offset_range: (min, max)
            yaw_mode: 'aligned' or 'random'
            yaw_offset_range: (min, max) used when yaw_mode is 'aligned'

        Returns:
            (x, y, yaw, metadata) or None
        """
        if not self.global_centerline:
            return None

        # Determine target distance
        if target_dist is None:
            target_dist = self.rng.uniform(0, self.total_track_length)
        else:
            # Handle wrap-around
            if self.total_track_length > 0:
                target_dist %= self.total_track_length

        # Find segment
        # Linear search optimization (can be binary search if needed)
        p1 = self.global_centerline[-1]
        p2 = self.global_centerline[0] # loop? No, simplified.
        
        # Proper lookup
        # If target_dist is 0, p1=0, p2=1 usually
        found_idx = -1
        # Quick check if points are sorted (they are)
        # Using binary search (bisect) could be better for large tracks, 
        # but linear for now as refactor step 1 (same as before).
        
        # Optimization: if we keep track of last index, we can search locally.
        # But for random access, search is needed.
        
        # Let's use simple logic similar to previous implementation for consistency first.
        p1 = self.global_centerline[-1]
        p2 = self.global_centerline[-1]
        
        for i in range(len(self.global_centerline) - 1):
             if self.global_centerline[i+1][3] >= target_dist:
                 p1 = self.global_centerline[i]
                 p2 = self.global_centerline[i+1]
                 found_idx = i
                 break
        
        # Interpolate
        d1, d2 = p1[3], p2[3]
        seg_len = d2 - d1
        ratio = (target_dist - d1) / seg_len if seg_len > 1e-6 else 0

        base_x = p1[0] + ratio * (p2[0] - p1[0])
        base_y = p1[1] + ratio * (p2[1] - p1[1])
        
        # Yaw interpolation
        yaw1, yaw2 = p1[2], p2[2]
        dyaw = yaw2 - yaw1
        while dyaw > math.pi: dyaw -= 2 * math.pi
        while dyaw < -math.pi: dyaw += 2 * math.pi
        base_yaw = yaw1 + ratio * dyaw

        # Apply Lateral Offset
        lat_offset = self.rng.uniform(lateral_offset_range[0], lateral_offset_range[1])
        
        # -sin, cos for normal
        offset_x = base_x - math.sin(base_yaw) * lat_offset
        offset_y = base_y + math.cos(base_yaw) * lat_offset

        # Final Yaw
        if yaw_mode == "random":
            final_yaw = self.rng.uniform(-math.pi, math.pi)
        else:
            yaw_noise = self.rng.uniform(yaw_offset_range[0], yaw_offset_range[1])
            final_yaw = base_yaw + yaw_noise

        meta = {
            "centerline_dist": float(target_dist),
            "lateral_offset": float(lat_offset),
            "base_yaw": float(base_yaw)
        }
        
        return (float(offset_x), float(offset_y), float(final_yaw), meta)


    def validate_pose(
        self,
        pose: tuple[float, float, float],
        shape: dict[str, float] | None = None, # width, length
        require_fully_contained: bool = False,
    ) -> bool:
        """Validate if a pose is valid within map bounds."""
        if not self.drivable_area:
            return True # No map to check against

        x, y, yaw = pose
        
        if shape:
            width = shape.get("width", 2.0)
            length = shape.get("length", 4.0)
            
            box = Polygon([
                (-length/2, -width/2),
                (length/2, -width/2),
                (length/2, width/2),
                (-length/2, width/2)
            ])
            box = rotate(box, yaw, origin=(0,0), use_radians=True)
            box = translate(box, x, y)
            
            if require_fully_contained:
                return self.drivable_area.contains(box)
            else:
                return self.drivable_area.intersects(box)
        else:
            # Point check
            from shapely.geometry import Point
            p = Point(x, y)
            if require_fully_contained:
                return self.drivable_area.contains(p)
            else:
                return self.drivable_area.intersects(p) # effectively same for point
