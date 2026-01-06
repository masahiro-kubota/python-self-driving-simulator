"""Initial state sampling for data collection.

This module provides functionality to sample random initial states
from a centerline track, with Lanelet validation.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simulator.map import LaneletMap

logger = logging.getLogger(__name__)


class InitialStateSampler:
    """Sample random initial states from centerline track."""

    def __init__(self, track_path: Path, map_path: Path) -> None:
        """Initialize sampler.

        Args:
            track_path: Path to centerline CSV file
            map_path: Path to Lanelet2 OSM map file
        """
        from experiment.engine.pose_sampler import PoseSampler

        # We initialize PoseSampler with no seed, as we will use the rng passed 
        # to sample_initial_state (by injecting it or using it).
        # PoseSampler loads the map and track.
        self.pose_sampler = PoseSampler(map_path=map_path, track_path=track_path, seed=None)

    def sample_initial_state(
        self,
        rng: np.random.Generator,
        lateral_offset_range: tuple[float, float],
        yaw_offset_range: tuple[float, float],
        velocity_range: tuple[float, float],
        max_retries: int = 10,
        vehicle_width: float | None = None,
        vehicle_length: float | None = None,
    ) -> dict[str, float]:
        """Sample a random initial state from centerline.

        Args:
            rng: Random number generator
            lateral_offset_range: Range for lateral offset from centerline [m]
            yaw_offset_range: Range for yaw offset from centerline direction [rad]
            velocity_range: Range for initial velocity [m/s]
            max_retries: Maximum number of retries for Lanelet validation
            vehicle_width: Vehicle width [m] for footprint check
            vehicle_length: Vehicle length [m] for footprint check

        Returns:
            Dictionary with keys: x, y, yaw, velocity

        Raises:
            RuntimeError: If failed to find valid position within max_retries
        """
        # Inject the provided RNG into PoseSampler (since Collector manages seeding)
        self.pose_sampler.rng = rng

        for attempt in range(max_retries):
            # Sample using PoseSampler
            result = self.pose_sampler.sample_track_pose(
                lateral_offset_range=lateral_offset_range,
                yaw_mode="aligned",
                yaw_offset_range=yaw_offset_range
            )

            if not result:
                continue

            x, y, yaw, _ = result

            shape = None
            if vehicle_width is not None and vehicle_length is not None:
                shape = {"width": vehicle_width, "length": vehicle_length}

            # Initial State Sampling implies we MUST be in drivable area.
            if self.pose_sampler.validate_pose(
                (x, y, yaw),
                shape=shape,
                require_fully_contained=True  # Requirement: "self-vehicle cannot start if not within Lanelet"
            ):
                # Sample velocity
                final_velocity = rng.uniform(velocity_range[0], velocity_range[1])

                logger.info(
                    f"Sampled initial state: x={x:.2f}, y={y:.2f}, "
                    f"yaw={yaw:.3f}, velocity={final_velocity:.2f} "
                    f"(attempt {attempt + 1})"
                )

                return {
                    "x": float(x),
                    "y": float(y),
                    "yaw": float(yaw),
                    "velocity": float(final_velocity),
                }
            else:
                logger.debug(f"Attempt {attempt + 1}: Pose invalid")

        # Failed to find valid position
        msg = f"Failed to sample valid initial state within {max_retries} retries"
        raise RuntimeError(msg)
