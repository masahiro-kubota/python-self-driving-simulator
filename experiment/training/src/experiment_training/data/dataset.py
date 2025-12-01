"""Dataset implementation for training."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from core.data import SimulationLog, Trajectory
from core.utils.geometry import distance, normalize_angle


class TrajectoryDataset(Dataset):
    """Dataset for trajectory following task."""

    def __init__(
        self,
        data_paths: list[str | Path],
        reference_trajectory: Trajectory | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            data_paths: List of paths to MCAP or JSON log files
            reference_trajectory: Reference trajectory for error calculation
        """
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        self.reference_trajectory = reference_trajectory

        for path in data_paths:
            self._load_data(Path(path))

    def _load_data(self, path: Path) -> None:
        """Load data from file."""
        if path.suffix == ".mcap":
            self._load_mcap(path)
        elif path.suffix == ".json":
            self._load_json(path)
        else:
            print(f"Warning: Unsupported file format {path.suffix}")

    def _load_mcap(self, path: Path) -> None:
        """Load data from MCAP file."""
        # Note: This requires the MCAPLogger to support reading, or we use mcap library directly
        # For now, assuming we can iterate over messages using a helper or the logger
        # Since MCAPLogger in experiment_runner is write-only context manager,
        # we might need to implement a reader or use mcap-ros2/mcap-protobuf if schema is known.
        # However, our MCAPLogger uses JSON serialization for simplicity.

        # TODO: Implement MCAP reading. For now, we'll skip or use a placeholder.
        # Real implementation would use mcap library to read the serialized JSON/Bytes.
        print(f"Loading MCAP from {path} (Not implemented yet, please use JSON logs for now)")
        pass

    def _load_json(self, path: Path) -> None:
        """Load data from JSON log file."""
        try:
            log = SimulationLog.load(path)
            self._process_log(log)
        except Exception as e:
            print(f"Error loading {path}: {e}")

    def _process_log(self, log: SimulationLog) -> None:
        """Process simulation log into training samples."""
        if not self.reference_trajectory:
            return

        for step in log.steps:
            # Calculate errors (features)
            # This logic duplicates NeuralController's error calculation
            # Ideally, we should refactor this into a shared utility or use the Controller class

            state = step.vehicle_state

            # Find nearest point
            min_dist = float("inf")
            nearest_idx = 0
            for i, point in enumerate(self.reference_trajectory):
                d = distance(state.x, state.y, point.x, point.y)
                if d < min_dist:
                    min_dist = d
                    nearest_idx = i

            nearest_point = self.reference_trajectory[nearest_idx]

            # Calculate lateral error
            dx = state.x - nearest_point.x
            dy = state.y - nearest_point.y
            path_yaw = nearest_point.yaw
            path_dx = np.cos(path_yaw)
            path_dy = np.sin(path_yaw)
            e_lat = dx * path_dy - dy * path_dx

            # Calculate yaw error
            e_yaw = normalize_angle(state.yaw - path_yaw)

            # Features: [e_lat, e_yaw, v, v_ref]
            features = np.array(
                [e_lat, e_yaw, state.velocity, nearest_point.velocity], dtype=np.float32
            )

            # Targets: [steering, acceleration]
            targets = np.array([step.action.steering, step.action.acceleration], dtype=np.float32)

            self.samples.append((features, targets))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        features, targets = self.samples[idx]
        return torch.from_numpy(features), torch.from_numpy(targets)
