"""Sensing data structure."""

from dataclasses import dataclass, field

from core.data.ad_components.state import VehicleState
from core.data.environment.obstacle import Obstacle


@dataclass
class Sensing:
    """自動運転コンポーネントへの入力データ（センシング情報）.

    Attributes:
        vehicle_state: 車両状態（位置、速度など）
        obstacles: 検知された障害物のリスト
    """

    vehicle_state: VehicleState
    obstacles: list[Obstacle] = field(default_factory=list)
