"""Kinematic bicycle model simulator implementation."""

from simulator_core.base import BaseSimulator
from simulator_core.data.environment import LaneletMap

from core.data import Action, VehicleParameters, VehicleState
from simulator_kinematic.vehicle import KinematicVehicleModel


class KinematicSimulator(BaseSimulator):
    """キネマティック自転車モデルに基づく軽量2Dシミュレータ."""

    def __init__(
        self,
        vehicle_params: "VehicleParameters | None" = None,
        initial_state: VehicleState | None = None,
        dt: float = 0.1,
        wheelbase: float | None = None,  # 後方互換性のため
        map_path: str | None = None,
    ) -> None:
        """初期化.

        Args:
            vehicle_params: 車両パラメータ（Noneの場合はデフォルト値を使用）
            initial_state: 初期車両状態
            dt: シミュレーション時間刻み [s]
            wheelbase: ホイールベース [m]（後方互換性のため、vehicle_paramsより優先）
            map_path: Lanelet2マップファイルへのパス
        """
        # 後方互換性: wheelbaseが指定されている場合はVehicleParametersを作成
        if wheelbase is not None and vehicle_params is None:
            vehicle_params = VehicleParameters(wheelbase=wheelbase)

        super().__init__(vehicle_params=vehicle_params, initial_state=initial_state, dt=dt)
        # self.vehicle_params will be populated by super().__init__ default if None
        self.vehicle_model = KinematicVehicleModel(wheelbase=self.vehicle_params.wheelbase)

        # マップの読み込み
        self.map: LaneletMap | None = None
        if map_path:
            import pathlib

            self.map = LaneletMap(pathlib.Path(map_path))

    def _update_state(self, action: Action) -> VehicleState:
        """Update vehicle state.

        Args:
            action: Control action

        Returns:
            Updated vehicle state
        """
        # 車両モデルによる更新
        next_state = self.vehicle_model.step(
            state=self._current_state,
            steering=action.steering,
            acceleration=action.acceleration,
            dt=self.dt,
        )

        # マップ外判定
        # マップ外判定
        # 簡略化のため、車両の中心位置のみで判定
        if self.map is not None and not self.map.is_drivable(next_state.x, next_state.y):
            next_state.off_track = True

        return next_state
