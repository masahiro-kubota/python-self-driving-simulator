"""Scene and track boundary data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from core.data.environment.obstacle import Obstacle, ObstacleType

if TYPE_CHECKING:
    from core.data import Trajectory


@dataclass
class TrackBoundary:
    """トラック境界定義.

    トラックの内側境界、外側境界、中心線を保持。
    """

    centerline: list[tuple[float, float]] = field(default_factory=list)  # 中心線 [(x, y), ...]
    inner_boundary: list[tuple[float, float]] = field(
        default_factory=list
    )  # 内側境界 [(x, y), ...]
    outer_boundary: list[tuple[float, float]] = field(
        default_factory=list
    )  # 外側境界 [(x, y), ...]

    @classmethod
    def from_trajectory(cls, trajectory: Trajectory) -> TrackBoundary:
        """Trajectoryから中心線を作成.

        Args:
            trajectory: 参照軌道

        Returns:
            TrackBoundary: トラック境界(中心線のみ)
        """
        centerline = [(point.x, point.y) for point in trajectory]
        return cls(centerline=centerline)


@dataclass
class Scene:
    """シミュレーション環境全体.

    トラック情報と障害物を管理。
    """

    track: TrackBoundary | None = None  # トラック境界
    obstacles: list[Obstacle] = field(default_factory=list)  # 障害物リスト

    def add_obstacle(self, obstacle: Obstacle) -> None:
        """障害物を追加.

        Args:
            obstacle: 追加する障害物
        """
        self.obstacles.append(obstacle)

    def get_obstacles_in_range(self, x: float, y: float, radius: float) -> list[Obstacle]:
        """指定範囲内の障害物を取得(将来の拡張用).

        Args:
            x: X座標 [m]
            y: Y座標 [m]
            radius: 検索半径 [m]

        Returns:
            list[Obstacle]: 範囲内の障害物リスト
        """
        # 現時点では簡易実装
        result = []
        for obs in self.obstacles:
            dist = ((obs.x - x) ** 2 + (obs.y - y) ** 2) ** 0.5
            if dist <= radius:
                result.append(obs)
        return result

    @classmethod
    def from_yaml(cls, path: Path) -> Scene:
        """YAMLファイルからシーン設定を読み込む.

        Args:
            path: YAMLファイルのパス

        Returns:
            Scene: シーン
        """
        with path.open() as f:
            data = yaml.safe_load(f)

        scene = cls()

        # NOTE: トラック情報の読み込みは将来の拡張で実装予定
        if "track" in data:
            # TODO: トラックファイルから読み込み
            pass

        # NOTE: 障害物の読み込みは将来の拡張で実装予定
        if "obstacles" in data:
            for obs_data in data["obstacles"]:
                obstacle = Obstacle(
                    id=obs_data["id"],
                    type=ObstacleType(obs_data["type"]),
                    x=obs_data["x"],
                    y=obs_data["y"],
                    width=obs_data["width"],
                    height=obs_data["height"],
                    yaw=obs_data.get("yaw", 0.0),
                    velocity=obs_data.get("velocity", 0.0),
                    trajectory=obs_data.get("trajectory"),
                )
                scene.add_obstacle(obstacle)

        return scene
