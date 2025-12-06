"""Dynamic bicycle model simulator implementation."""

from simulator_core.data import SimulationVehicleState
from simulator_core.simulator import BaseSimulator
from simulator_core.solver import rk4_step

from core.data import Action, VehicleParameters, VehicleState
from simulator_dynamic.vehicle import DynamicVehicleModel


class DynamicSimulator(BaseSimulator):
    """ダイナミック自転車モデルに基づく2Dシミュレータ."""

    def __init__(
        self,
        vehicle_params: "VehicleParameters | None" = None,
        initial_state: VehicleState | None = None,
        dt: float = 0.01,  # Smaller dt for RK4 stability
        map_path: str | None = None,
    ) -> None:
        """初期化.

        Args:
            vehicle_params: 車両パラメータ
            initial_state: 初期車両状態(キネマティクス形式)
            dt: シミュレーション時間刻み [s]
            map_path: Lanelet2マップファイルへのパス
        """

        super().__init__(
            vehicle_params=vehicle_params, initial_state=initial_state, dt=dt, map_path=map_path
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
