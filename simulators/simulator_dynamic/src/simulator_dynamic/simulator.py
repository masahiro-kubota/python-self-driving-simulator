"""Dynamic bicycle model simulator implementation."""

from typing import TYPE_CHECKING

from simulator_core.data import SimulationVehicleState
from simulator_core.simulator import BaseSimulator
from simulator_core.solver import rk4_step

from core.data import Action, VehicleParameters, VehicleState
from simulator_dynamic.vehicle import DynamicVehicleModel

if TYPE_CHECKING:
    from shapely.geometry import Polygon


class DynamicSimulator(BaseSimulator):
    """ダイナミック自転車モデルに基づく2Dシミュレータ."""

    def __init__(
        self,
        vehicle_params: "VehicleParameters | None" = None,
        initial_state: VehicleState | None = None,
        dt: float = 0.01,  # Smaller dt for RK4 stability
        map_path: str | None = None,
        goal_x: float | None = None,
        goal_y: float | None = None,
    ) -> None:
        """初期化.

        Args:
            vehicle_params: 車両パラメータ
            initial_state: 初期車両状態(キネマティクス形式)
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

        self.vehicle_model = DynamicVehicleModel(params=self.vehicle_params)

    def _update_state(self, action: Action) -> SimulationVehicleState:
        """Update vehicle state using RK4 integration.

        Args:
            action: Control action

        Returns:
            Updated vehicle state (SimulationVehicleState)
        """
        # Convert acceleration to throttle (simplified)
        throttle = action.acceleration / 5.0  # Normalize to [-1, 1] range
        throttle = max(-1.0, min(1.0, throttle))

        # Dynamic update using RK4
        def derivative_func(state: SimulationVehicleState) -> SimulationVehicleState:
            return self.vehicle_model.calculate_derivative(state, action.steering, throttle)

        next_state = rk4_step(
            state=self._current_state,
            derivative_func=derivative_func,
            dt=self.dt,
            add_func=self.vehicle_model.add_state,
        )

        return next_state

    def _get_vehicle_polygon(self, state: VehicleState) -> "Polygon":
        """車両のポリゴンを取得 (Dynamic: 重心基準)."""

        p = self.vehicle_params

        # ダイナミクスモデルは重心(CG)が基準 (x, y)
        # 前端: lf + front_overhang
        # 後端: -(lr + rear_overhang)

        front_edge = p.lf + p.front_overhang
        rear_edge = -(p.lr + p.rear_overhang)

        return self._create_vehicle_polygon(
            x=state.x,
            y=state.y,
            yaw=state.yaw,
            front_edge_dist=front_edge,
            rear_edge_dist=rear_edge,
            half_width=p.width / 2.0,
        )
