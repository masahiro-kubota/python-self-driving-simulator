"""Simple dataset for function approximation."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FunctionDataset(Dataset):
    """Dataset for function approximation."""

    def __init__(self, data_path: str | Path) -> None:
        """Initialize dataset.

        Args:
            data_path: Path to JSON data file
        """
        with open(data_path) as f:
            data = json.load(f)

        self.x = np.array(data["x"], dtype=np.float32).reshape(-1, 1)
        self.y = np.array(data["y"], dtype=np.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])
