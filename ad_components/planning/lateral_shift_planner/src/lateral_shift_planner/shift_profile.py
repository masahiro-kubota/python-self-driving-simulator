from dataclasses import dataclass
from enum import Enum

import numpy as np

from lateral_shift_planner.obstacle_manager import TargetObstacle


class AvoidanceDirection(Enum):
    LEFT = 1
    RIGHT = -1


@dataclass
class ProfilePoint:
    s: float
    lat_req: float
    direction: AvoidanceDirection


class ShiftProfile:
    """Generates shift profile for a single obstacle."""

    def __init__(
        self,
        obstacle: TargetObstacle,
        vehicle_width: float,
        safe_margin: float = 0.5,
        avoidance_maneuver_length: float = 10.0,
        longitudinal_margin_front: float = 2.0,
        longitudinal_margin_rear: float = 2.0,
    ):
        """Initialize ShiftProfile.

        Args:
            obstacle: Target obstacle (includes road boundary info)
            vehicle_width: Ego width
            safe_margin: Safety margin
            avoidance_maneuver_length: Longitudinal distance for lane change
            longitudinal_margin_front: Buffer distance before obstacle
            longitudinal_margin_rear: Buffer distance after obstacle
        """
        self.obs = obstacle

        # Determine direction based on available space to road boundaries
        # obstacle.lat is in Frenet frame (lateral offset from centerline)
        # obstacle.left_boundary_dist and right_boundary_dist are ALREADY distances
        # from the obstacle's position to the road edges, so we don't need to adjust by lat

        # Distance from obstacle to left boundary
        space_to_left = obstacle.left_boundary_dist

        # Distance from obstacle to right boundary
        space_to_right = obstacle.right_boundary_dist

        # Choose direction with more space
        if space_to_left >= space_to_right:
            self.sign = 1.0  # Left (positive lat)
        else:
            self.sign = -1.0  # Right (negative lat)

        # Calculate required shift amount
        required_clearance = obstacle.width / 2.0 + vehicle_width / 2.0 + safe_margin
        raw_target_lat = obstacle.lat + self.sign * required_clearance

        # Clamp target against centerline (0.0)
        # We only want to shift if the centerline is blocked/unsafe.
        # If obstacle is far Left (sign < 0), raw_target might be +1.25.
        # We should NOT move to +1.25 from 0.0. We should stay at min(0, 1.25) = 0.
        if self.sign > 0:  # Left shift
            self.target_lat = max(0.0, raw_target_lat)
        else:  # Right shift
            self.target_lat = min(0.0, raw_target_lat)

        # Update s range
        # Update s range
        # obstacle.s is the center of the obstacle
        # Start avoidance: center - half_length - margin - maneuver_length
        half_length = obstacle.length / 2.0
        self.s_start_action = (
            obstacle.s - half_length - longitudinal_margin_front - avoidance_maneuver_length
        )
        self.s_full_avoid = obstacle.s - half_length - longitudinal_margin_front
        self.s_keep_avoid = obstacle.s + half_length + longitudinal_margin_rear
        self.s_end_action = (
            obstacle.s + half_length + longitudinal_margin_rear + avoidance_maneuver_length
        )

        import logging

        logging.getLogger(__name__).info(
            f"[ShiftProfile] ID={obstacle.id} s_obs={obstacle.s:.2f} len={obstacle.length:.2f} (half={half_length:.2f}) "
            f"margins(F/R)={longitudinal_margin_front:.2f}/{longitudinal_margin_rear:.2f} "
            f"s_full={self.s_full_avoid:.2f} s_keep={self.s_keep_avoid:.2f} "
            f"obs_front={obstacle.s - half_length:.2f} obs_rear={obstacle.s + half_length:.2f}"
        )

    def get_lat(self, s: float) -> float:
        """Get required lat at s."""
        if s < self.s_start_action or s > self.s_end_action:
            return 0.0

        if s < self.s_full_avoid:
            # Rampping up
            ratio = (s - self.s_start_action) / (self.s_full_avoid - self.s_start_action)
            # Smoothstep
            k = ratio * ratio * (3 - 2 * ratio)
            return k * self.target_lat

        if s > self.s_keep_avoid:
            # Rampping down
            ratio = (self.s_end_action - s) / (self.s_end_action - self.s_keep_avoid)
            k = ratio * ratio * (3 - 2 * ratio)
            return k * self.target_lat

        # Constant
        return self.target_lat


def merge_profiles(s_samples: np.ndarray, profiles: list[ShiftProfile]) -> tuple[np.ndarray, bool]:
    """Merge profiles.

    Returns:
        lat_target: Array of target lat values
        collision: Boolean, true if impossible
    """
    lat_target = np.zeros_like(s_samples)

    # We process point-wise
    for i, s in enumerate(s_samples):
        bound_min = -float("inf")  # Needs to be > this
        bound_max = float("inf")  # Needs to be < this

        active_min = False
        active_max = False

        for p in profiles:
            lat_req = p.get_lat(s)

            # Identify if this profile is active at this s
            if abs(lat_req) < 1e-6:
                continue

            if p.sign > 0:  # Left shift (Positive)
                bound_min = max(bound_min, lat_req)
                active_min = True
            else:  # Right shift (Negative)
                bound_max = min(bound_max, lat_req)
                active_max = True

        if active_min and active_max:
            if bound_min > bound_max:
                # Collision
                return lat_target, True  # Or handle partial
            lat_target[i] = (bound_min + bound_max) / 2.0

        elif active_min:
            lat_target[i] = bound_min

        elif active_max:
            lat_target[i] = bound_max

        else:
            lat_target[i] = 0.0

    return lat_target, False
