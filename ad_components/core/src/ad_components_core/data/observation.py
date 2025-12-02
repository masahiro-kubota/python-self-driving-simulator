"""Observation data structure."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Observation:
    """センサー観測データを表すデータクラス."""

    lateral_error: float  # 横方向偏差 [m]
    heading_error: float  # ヨー角偏差 [rad]
    velocity: float  # 現在速度 [m/s]
    target_velocity: float  # 目標速度 [m/s]
    distance_to_goal: float | None = None  # ゴールまでの距離 [m]
    timestamp: float | None = None  # タイムスタンプ [s]

    def to_array(self) -> np.ndarray:
        """numpy配列に変換."""
        return np.array(
            [
                self.lateral_error,
                self.heading_error,
                self.velocity,
                self.target_velocity,
            ]
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Observation":
        """numpy配列から生成."""
        return cls(
            lateral_error=arr[0],
            heading_error=arr[1],
            velocity=arr[2],
            target_velocity=arr[3],
        )
