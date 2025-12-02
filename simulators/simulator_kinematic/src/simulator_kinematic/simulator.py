"""Kinematic bicycle model simulator implementation."""

from typing import TYPE_CHECKING, Any

from simulators_core.base import BaseSimulator

from core.data import Action, SimulationStep, VehicleParameters, VehicleState
from simulator_kinematic.vehicle import KinematicVehicleModel

if TYPE_CHECKING:
    from core.data import Scene


class KinematicSimulator(BaseSimulator):
    """キネマティック自転車モデルに基づく軽量2Dシミュレータ."""

    def __init__(
        self,
        vehicle_params: "VehicleParameters | None" = None,
        scene: "Scene | None" = None,
        initial_state: VehicleState | None = None,
        dt: float = 0.1,
        wheelbase: float | None = None,  # 後方互換性のため
    ) -> None:
        """初期化.

        Args:
            vehicle_params: 車両パラメータ（Noneの場合はデフォルト値を使用）
            scene: シミュレーション環境（Noneの場合は空のシーンを使用）
            initial_state: 初期車両状態
            dt: シミュレーション時間刻み [s]
            wheelbase: ホイールベース [m]（後方互換性のため、vehicle_paramsより優先）
        """
        # 後方互換性: wheelbaseが指定されている場合はVehicleParametersを作成
        if wheelbase is not None and vehicle_params is None:
            vehicle_params = VehicleParameters(wheelbase=wheelbase)

        super().__init__(
            vehicle_params=vehicle_params, scene=scene, initial_state=initial_state, dt=dt
        )
        self.vehicle_model = KinematicVehicleModel(wheelbase=self.vehicle_params.wheelbase)

    def step(self, action: Action) -> tuple[VehicleState, bool, dict[str, Any]]:
        """Execute one simulation step.

        Args:
            action: Control action

        Returns:
            tuple containing:
                - next_state: Updated vehicle state
                - done: Episode termination flag
                - info: Additional information
        """
        # Update vehicle state
        self._current_state = self.vehicle_model.step(
            state=self._current_state,
            steering=action.steering,
            acceleration=action.acceleration,
            dt=self.dt,
        )

        # Update simulation log
        step = SimulationStep(
            timestamp=self._current_state.timestamp,
            vehicle_state=self._current_state,
            action=action,
            ad_component_log=None,  # Will be populated by ExperimentRunner if needed
            info=self._create_info(),
        )
        self.log.add_step(step)

        # Check termination
        done = self._is_done()
        info = self._create_info()

        return self._current_state, done, info
