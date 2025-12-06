"""Kinematic bicycle model simulator implementation."""

from simulator_core.data import SimulationVehicleState
from simulator_core.simulator import BaseSimulator

from core.data import Action, VehicleParameters, VehicleState
from simulator_kinematic.vehicle import KinematicVehicleModel


class KinematicSimulator(BaseSimulator):
    """キネマティック自転車モデルに基づく軽量2Dシミュレータ."""

    def __init__(
        self,
        vehicle_params: "VehicleParameters | None" = None,
        initial_state: VehicleState | None = None,
        dt: float = 0.1,
        map_path: str | None = None,
    ) -> None:
        """初期化.

        Args:
            vehicle_params: 車両パラメータ(Noneの場合はデフォルト値を使用)
            initial_state: 初期車両状態
            dt: シミュレーション時間刻み [s]
            map_path: Lanelet2マップファイルへのパス
        """

        super().__init__(
            vehicle_params=vehicle_params, initial_state=initial_state, dt=dt, map_path=map_path
        )
        # self.vehicle_params will be populated by super().__init__ default if None
        self.vehicle_model = KinematicVehicleModel(wheelbase=self.vehicle_params.wheelbase)

    def _update_state(self, action: Action) -> SimulationVehicleState:
        """Update vehicle state.

        Args:
            action: Control action

        Returns:
            Updated vehicle state (SimulationVehicleState)
        """
        # 車両モデルによる更新
        next_state = self.vehicle_model.step(
            state=self._current_state,
            steering=action.steering,
            acceleration=action.acceleration,
            dt=self.dt,
        )

        return next_state
