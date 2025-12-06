"""Dynamic bicycle model simulator implementation."""

import math
from typing import TYPE_CHECKING, Any

from simulator_core.base import BaseSimulator

from core.data import Action, VehicleParameters, VehicleState
from simulator_dynamic.state import DynamicVehicleState
from simulator_dynamic.vehicle import DynamicVehicleModel
from simulator_dynamic.vehicle_params import VehicleParameters as DynamicVehicleParams

if TYPE_CHECKING:
    from core.data import Scene


class DynamicSimulator(BaseSimulator):
    """ダイナミック自転車モデルに基づく2Dシミュレータ."""

    def __init__(
        self,
        vehicle_params: "VehicleParameters | None" = None,
        scene: "Scene | None" = None,
        initial_state: VehicleState | None = None,
        dt: float = 0.01,  # Smaller dt for RK4 stability
        params: DynamicVehicleParams | None = None,  # 後方互換性のため
    ) -> None:
        """初期化.

        Args:
            vehicle_params: 車両パラメータ（Noneの場合はデフォルト値を使用）
            scene: シミュレーション環境（Noneの場合は空のシーンを使用）
            initial_state: 初期車両状態(キネマティクス形式)
            dt: シミュレーション時間刻み [s]
            params: 動力学車両パラメータ（後方互換性のため、vehicle_paramsより優先）
        """
        super().__init__(
            vehicle_params=vehicle_params, scene=scene, initial_state=initial_state, dt=dt
        )

        # 後方互換性: paramsが指定されている場合はそれを使用
        if params is not None:
            dynamic_params = params
        else:
            # simulator_core.VehicleParametersをDynamicVehicleParamsに変換
            dynamic_params = DynamicVehicleParams(
                mass=self.vehicle_params.mass or 1500.0,
                iz=self.vehicle_params.inertia or 2500.0,
                wheelbase=self.vehicle_params.wheelbase,
                lf=self.vehicle_params.lf or 1.2,
                lr=self.vehicle_params.lr or 1.3,
                cf=self.vehicle_params.cf or 80000.0,
                cr=self.vehicle_params.cr or 80000.0,
                c_drag=self.vehicle_params.c_drag or 0.3,
                c_roll=self.vehicle_params.c_roll or 0.015,
                max_drive_force=self.vehicle_params.max_drive_force or 5000.0,
                max_brake_force=self.vehicle_params.max_brake_force or 8000.0,
            )

        self.vehicle_model = DynamicVehicleModel(params=dynamic_params)

        # Convert kinematic state to dynamic state
        self._dynamic_state = self._kinematic_to_dynamic(self.initial_state)

    def reset(self) -> VehicleState:
        """シミュレーションをリセット.

        Returns:
            初期車両状態
        """
        self._current_state = self.initial_state
        self._dynamic_state = self._kinematic_to_dynamic(self.initial_state)
        return self._current_state

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
        # Convert acceleration to throttle (simplified)
        throttle = action.acceleration / 5.0  # Normalize to [-1, 1] range
        throttle = max(-1.0, min(1.0, throttle))

        self._dynamic_state = self.vehicle_model.step(
            state=self._dynamic_state,
            steering=action.steering,
            throttle=throttle,
            dt=self.dt,
        )

        # Convert dynamic state back to kinematic state
        self._current_state = self._dynamic_to_kinematic(
            self._dynamic_state, action.steering, action.acceleration
        )

        done = self._is_done()
        info = self._create_info()

        return self._current_state, done, info

    def _kinematic_to_dynamic(self, state: VehicleState) -> DynamicVehicleState:
        """キネマティクス状態をダイナミクス状態に変換.

        Args:
            state: キネマティクス状態

        Returns:
            ダイナミクス状態
        """

        # Assume no lateral velocity initially
        vx = state.velocity * math.cos(0.0)  # beta = 0
        vy = state.velocity * math.sin(0.0)

        return DynamicVehicleState(
            x=state.x,
            y=state.y,
            yaw=state.yaw,
            vx=vx,
            vy=vy,
            yaw_rate=0.0,
            steering=state.steering,
            throttle=0.0,
            timestamp=state.timestamp,
        )

    def _dynamic_to_kinematic(
        self, state: DynamicVehicleState, steering: float, acceleration: float
    ) -> VehicleState:
        """ダイナミクス状態をキネマティクス状態に変換.

        Args:
            state: ダイナミクス状態
            steering: ステアリング角
            acceleration: 加速度

        Returns:
            キネマティクス状態
        """
        return VehicleState(
            x=state.x,
            y=state.y,
            yaw=state.yaw,
            velocity=state.velocity,
            acceleration=acceleration,
            steering=steering,
            timestamp=state.timestamp,
        )
