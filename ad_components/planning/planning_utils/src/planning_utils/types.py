"""Planning internal data types."""

import numpy as np
from pydantic import BaseModel


class ReferencePathPoint(BaseModel):
    """参照経路上の1点を表すデータクラス."""

    x: float  # X座標 [m]
    y: float  # Y座標 [m]
    yaw: float  # ヨー角 [rad]
    velocity: float  # 速度 [m/s]


class ReferencePath(BaseModel):
    """参照経路を表すデータクラス."""

    points: list[ReferencePathPoint]  # 経路点のリスト

    def __len__(self) -> int:
        """経路点の数を返す."""
        return len(self.points)

    def __getitem__(self, idx: int) -> ReferencePathPoint:
        """インデックスで経路点を取得."""
        return self.points[idx]

    def __iter__(self):
        """反復子を返す."""
        return iter(self.points)

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """numpy配列に変換 (x, y, yaw, velocity)."""
        x = np.array([p.x for p in self.points])
        y = np.array([p.y for p in self.points])
        yaw = np.array([p.yaw for p in self.points])
        velocity = np.array([p.velocity for p in self.points])
        return x, y, yaw, velocity
