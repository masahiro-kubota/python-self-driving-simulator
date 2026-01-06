"""Data loading utilities for Tiny LiDAR Net training."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import bisect
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ScanControlDataset(Dataset):
    """PyTorch Dataset for LiDAR scans and control commands.

    Loads synchronized .npy files (scans, steers, accelerations) from a directory.
    Uses Lazy Loading (files kept open with mmap) to support large datasets without OOM.
    Optionally allows loading entire dataset into RAM for performance.
    """

    def __init__(
        self,
        data_dir: Path | str,
        max_range: float = 30.0,
        stats: dict[str, Any] | None = None,
        cache_to_ram: bool = False,
    ):
        """Initialize the dataset.

        Args:
            data_dir: Path to the directory containing .npy files
            max_range: Maximum range (fallback if stats is not provided)
            stats: Dictionary containing normalization statistics (mean, std, etc.)
            cache_to_ram: If True, load all data into RAM instead of using mmap.
        """
        import bisect  # Import locally to avoid top-level clutter if preferred, but usually top-level is better. Kept local here for simplicity in replacement.

        self.data_dir = Path(data_dir)
        self.max_range = max_range
        self.stats = stats

        self.batches = []  # List of (scans, steers, accels) mmapped arrays
        self.cumulative_sizes = []
        self.total_size = 0

        self.scan_files = []
        self.steer_files = []
        self.accel_files = []

        # In-memory storage
        self.cache_to_ram = cache_to_ram
        self.scans_cache = []
        self.steers_cache = []
        self.accels_cache = []

        try:
            # Try loading single files first (legacy format)
            single_scans = self.data_dir / "scans.npy"
            if single_scans.exists():
                # For single file, just store it as one batch
                self.scan_files.append(single_scans)
                self.steer_files.append(self.data_dir / "steers.npy")
                self.accel_files.append(self.data_dir / "accelerations.npy")
                
                # We still need size for indexing
                scans = np.load(single_scans, mmap_mode="r")
                self.total_size += len(scans)
                self.cumulative_sizes.append(self.total_size)
                
                logger.info(f"Loaded single-file dataset from {self.data_dir} (mmap)")
            else:
                # Load batched files (new format: batch_XXXX_*.npy)
                scans_files = sorted(self.data_dir.glob("batch_*_scans.npy"))
                if not scans_files:
                    raise FileNotFoundError(f"No scans.npy or batch_*_scans.npy files found in {self.data_dir}")
                
                logger.info(f"Found {len(scans_files)} batch files in {self.data_dir}")
                
                for scan_file in scans_files:
                    batch_name = scan_file.stem.replace("_scans", "")
                    steer_file = self.data_dir / f"{batch_name}_steers.npy"
                    accel_file = self.data_dir / f"{batch_name}_accelerations.npy"
                    
                    if not steer_file.exists() or not accel_file.exists():
                        logger.warning(f"Skipping incomplete batch: {batch_name}")
                        continue
                    
                    # Store paths
                    self.scan_files.append(scan_file)
                    self.steer_files.append(steer_file)
                    self.accel_files.append(accel_file)

                    # Get size (cheap read with mmap)
                    scans = np.load(scan_file, mmap_mode="r")
                    batch_len = len(scans)
                    
                    # Basic validation check (length) - usually fast on mmap
                    # We need to load steers and accels to check their length
                    steers = np.load(steer_file, mmap_mode="r")
                    accels = np.load(accel_file, mmap_mode="r")
                    if not (batch_len == len(steers) == len(accels)):
                         logger.warning(f"Length mismatch in batch {batch_name}, skipping.")
                         # Remove the last added paths if validation fails
                         self.scan_files.pop()
                         self.steer_files.pop()
                         self.accel_files.pop()
                         continue

                    self.total_size += batch_len
                    self.cumulative_sizes.append(self.total_size)
                
                if not self.scan_files: # Check if any files were successfully loaded
                    raise FileNotFoundError(f"No valid batch files found in {self.data_dir}")
                
                logger.info(f"Initialized lazy loading for {len(self.scan_files)} batches. Total samples: {self.total_size}")
                
            # If caching is enabled, load everything now
            if self.cache_to_ram:
                logger.info(f"Loading {self.total_size} samples into RAM... (cache_to_ram=True)")
                for i in range(len(self.scan_files)):
                    # Load and append to cache lists. 
                    # Note: We already have file paths.
                    # We use mmap_mode=None to load into memory.
                    s = np.load(self.scan_files[i], mmap_mode=None)
                    st = np.load(self.steer_files[i], mmap_mode=None)
                    ac = np.load(self.accel_files[i], mmap_mode=None)
                    self.scans_cache.append(s)
                    self.steers_cache.append(st)
                    self.accels_cache.append(ac)
                logger.info("Finished loading dataset into RAM.")
                
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing required .npy files in {self.data_dir}: {e}")

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (scan, target) where:
                scan: Normalized LiDAR scan data (float32)
                target: Control command vector [acceleration, steering] (float32)
        """
        # Initialize variables for logging in case of error
        batch_idx = -1
        local_idx = -1
        scan_path = Path("")
        steer_path = Path("")
        accel_path = Path("")

        try:
            if idx < 0:
                if -idx > self.total_size:
                    raise IndexError(f"Index {idx} out of bounds for size {self.total_size}")
                idx += self.total_size
            if idx >= self.total_size:
                raise IndexError(f"Index {idx} out of bounds for size {self.total_size}")

            # Binary search to find which batch file contains the index
            # bisect_right returns an insertion point which is 1-based for finding the batch.
            # So, if idx is in the first batch, it returns 1. If in the second, 2, etc.
            # We need 0-based index for self.scan_files, so subtract 1.
            batch_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            
            # Calculate local index within that batch
            if batch_idx == 0:
                local_idx = idx
            else:
                local_idx = idx - self.cumulative_sizes[batch_idx - 1]
                
            # Load specific batch data
            if self.cache_to_ram:
                # Access from RAM cache
                raw_scan = self.scans_cache[batch_idx][local_idx].astype(np.float32)
                steer = self.steers_cache[batch_idx][local_idx].astype(np.float32)
                accel = self.accels_cache[batch_idx][local_idx].astype(np.float32)
            else:
                # Access via mmap
                scan_path = self.scan_files[batch_idx]
                steer_path = self.steer_files[batch_idx]
                accel_path = self.accel_files[batch_idx]
                
                # Use mmap_mode='r' to avoid loading full file to RAM
                # Only access the specific row
                try:
                    scans_mmap = np.load(scan_path, mmap_mode='r')
                    steers_mmap = np.load(steer_path, mmap_mode='r')
                    accels_mmap = np.load(accel_path, mmap_mode='r')
                except Exception as e:
                    logger.error(f"Error loading batch file: index={idx}, batch_idx={batch_idx}")
                    logger.error(f"Scan path: {scan_path}")
                    logger.error(f"Steer path: {steer_path}")
                    logger.error(f"Accel path: {accel_path}")
                    raise e
                
                raw_scan = scans_mmap[local_idx].astype(np.float32)
                steer = steers_mmap[local_idx].astype(np.float32)
                accel = accels_mmap[local_idx].astype(np.float32)

            # Let's clean NaN just in case.
            scan = np.nan_to_num(raw_scan, nan=0.0)

            # Preprocessing: Normalization (Applied on-the-fly)
            if self.stats and "scans" in self.stats:
                s_stats = self.stats["scans"]
                # Apply statistical normalization
                # Note: s_stats["mean"] and s_stats["std"] should be scalars or matching shape
                scan = (scan - s_stats["mean"]) / (s_stats["std"] + 1e-6)
            else:
                # Apply range-based normalization
                scan = np.clip(scan, 0.0, self.max_range) / self.max_range

            # Target vector construction: [Acceleration, Steering]
            target = np.array([accel, steer], dtype=np.float32)
            
            return scan, target

        except Exception as e:
            logger.error(f"Error loading sample at global_index={idx}: {e}")
            logger.error(f"  Mapped to batch_idx={batch_idx}, local_idx={local_idx}")
            logger.error(f"  Files: {scan_path}, {steer_path}, {accel_path}")
            raise e
