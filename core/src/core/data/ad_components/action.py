"""Action data structure."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Action:
    """制御指令を表すデータクラス."""

    steering: float  # ステアリング角 [rad]
    acceleration: float  # 加速度 [m/s^2]
    timestamp: float | None = None  # タイムスタンプ [s]

    def to_array(self) -> np.ndarray:
        """numpy配列に変換."""
        return np.array([self.steering, self.acceleration])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Action":
        """numpy配列から生成."""
        return cls(steering=arr[0], acceleration=arr[1])
