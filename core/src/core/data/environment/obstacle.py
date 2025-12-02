"""Obstacle definition for future extension."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ObstacleType(Enum):
    """障害物タイプ."""

    STATIC = "static"  # 静的障害物
    DYNAMIC = "dynamic"  # 動的障害物


@dataclass
class Obstacle:
    """障害物定義（将来の拡張用）.

    現時点では最小限の実装。実際の障害物検出ロジックは未実装。
    """

    id: str  # 障害物ID
    type: ObstacleType  # 障害物タイプ
    x: float  # X座標 [m]
    y: float  # Y座標 [m]
    width: float  # 幅 [m]
    height: float  # 高さ [m]
    yaw: float = 0.0  # ヨー角 [rad]

    # NOTE: 以下のフィールドは動的障害物の将来の拡張で使用予定
    velocity: float = 0.0  # 速度 [m/s]
    trajectory: list[tuple[float, float]] | None = None  # 軌道 [(x, y), ...]
