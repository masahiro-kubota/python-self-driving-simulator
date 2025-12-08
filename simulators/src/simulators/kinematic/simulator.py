"""Kinematic bicycle model simulator implementation."""

from typing import TYPE_CHECKING

from core.data import Action, VehicleParameters, VehicleState
from simulators.core.data import SimulationVehicleState
from simulators.core.simulator import BaseSimulator
from simulators.kinematic.vehicle import KinematicVehicleModel

if TYPE_CHECKING:
    from shapely.geometry import Polygon


class KinematicSimulator(BaseSimulator):
    """キネマティック自転車モデルに基づく軽量2Dシミュレータ."""

    def __init__(
        self,
        dt: float,
        vehicle_params: "VehicleParameters | None" = None,
        initial_state: VehicleState | None = None,
        map_path: str | None = None,
        goal_x: float | None = None,
        goal_y: float | None = None,
    ) -> None:
        """初期化.

        Args:
            vehicle_params: 車両パラメータ(Noneの場合はデフォルト値を使用)
            initial_state: 初期車両状態
            dt: シミュレーション時間刻み [s]
            map_path: Lanelet2マップファイルへのパス
            goal_x: ゴール位置のX座標 [m]
            goal_y: ゴール位置のY座標 [m]
        """

        super().__init__(
            vehicle_params=vehicle_params,
            initial_state=initial_state,
            dt=dt,
            map_path=map_path,
            goal_x=goal_x,
            goal_y=goal_y,
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

    def _get_vehicle_polygon(self, state: VehicleState) -> "Polygon":
        """車両のポリゴンを取得 (Kinematic: 後輪中心基準)."""

        p = self.vehicle_params

        # キネマティクスモデルは後輪中心が基準 (x, y)
        # 前端: wheelbase + front_overhang
        # 後端: -rear_overhang

        front_edge = p.wheelbase + p.front_overhang
        rear_edge = -p.rear_overhang

        return self._create_vehicle_polygon(
            x=state.x,
            y=state.y,
            yaw=state.yaw,
            front_edge_dist=front_edge,
            rear_edge_dist=rear_edge,
            half_width=p.width / 2.0,
        )
