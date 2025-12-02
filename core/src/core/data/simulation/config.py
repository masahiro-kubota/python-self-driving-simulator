"""Simulation configuration data structure."""

from dataclasses import dataclass

from core.data.ad_components.config import ADComponentConfig
from core.data.ad_components.state import VehicleState
from core.data.environment.scene import Scene
from core.data.vehicle.params import VehicleParameters


@dataclass
class SimulationConfig:
    """1回のシミュレーション実行に必要な設定.

    Attributes:
        scene: 環境データ（トラック境界、障害物）
        vehicle_params: 車両諸元
        initial_state: 初期車両状態
        ad_component_config: 自動運転コンポーネント設定
        dt: タイムステップ（秒）
        simulator_type: シミュレータタイプ（kinematic / dynamic）
    """

    scene: Scene
    vehicle_params: VehicleParameters
    initial_state: VehicleState
    ad_component_config: ADComponentConfig
    dt: float = 0.1
    simulator_type: str = "kinematic"
