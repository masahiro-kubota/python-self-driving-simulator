"""Data loading utilities for Tiny LiDAR Net training."""

import logging
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ScanControlDataset(Dataset):
    """PyTorch Dataset for LiDAR scans and control commands.

    Loads synchronized .npy files (scans, steers, accelerations) from a directory.
    The LiDAR scans are normalized by the specified maximum range.
    """

    def __init__(self, data_dir: Path | str, max_range: float = 30.0):
        """Initialize the dataset.

        Args:
            data_dir: Path to the directory containing .npy files
            max_range: Maximum range for LiDAR normalization
        """
        self.data_dir = Path(data_dir)
        self.max_range = max_range

        try:
            # Load raw data
            self.scans = np.load(self.data_dir / "scans.npy")
            self.steers = np.load(self.data_dir / "steers.npy")
            self.accels = np.load(self.data_dir / "accelerations.npy")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing required .npy files in {self.data_dir}: {e}")

        # Validate data consistency
        n_samples = len(self.scans)
        if not (len(self.steers) == n_samples and len(self.accels) == n_samples):
            raise ValueError(
                f"Data length mismatch in {self.data_dir}: "
                f"Scans={len(self.scans)}, Steers={len(self.steers)}, Accels={len(self.accels)}"
            )

        # Preprocessing: Clip and Normalize
        self.scans = np.clip(self.scans, 0.0, self.max_range) / self.max_range

        logger.info(f"Loaded {n_samples} samples from {self.data_dir}")

    def __len__(self) -> int:
        return len(self.scans)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (scan, target) where:
                scan: Normalized LiDAR scan data (float32)
                target: Control command vector [acceleration, steering] (float32)
        """
        # Ensure data is float32 for PyTorch compatibility
        scan = self.scans[idx].astype(np.float32)

        accel = np.float32(self.accels[idx])
        steer = np.float32(self.steers[idx])

        # Target vector construction: [Acceleration, Steering]
        target = np.array([accel, steer], dtype=np.float32)

        return scan, target
