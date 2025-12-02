"""Trajectory data structures."""

from dataclasses import dataclass

import numpy as np


@dataclass
class TrajectoryPoint:
    """軌道上の1点を表すデータクラス."""

    x: float  # X座標 [m]
    y: float  # Y座標 [m]
    yaw: float  # ヨー角 [rad]
    velocity: float  # 速度 [m/s]


@dataclass
class Trajectory:
    """軌道を表すデータクラス."""

    points: list[TrajectoryPoint]  # 軌道点のリスト

    def __len__(self) -> int:
        """軌道点の数を返す."""
        return len(self.points)

    def __getitem__(self, idx: int) -> TrajectoryPoint:
        """インデックスで軌道点を取得."""
        return self.points[idx]

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """numpy配列に変換 (x, y, yaw, velocity)."""
        x = np.array([p.x for p in self.points])
        y = np.array([p.y for p in self.points])
        yaw = np.array([p.yaw for p in self.points])
        velocity = np.array([p.velocity for p in self.points])
        return x, y, yaw, velocity
