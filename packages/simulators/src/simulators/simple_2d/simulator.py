"""Simple 2D simulator implementation."""

from typing import Any, Optional

from core.data import Action, Observation, VehicleState
from core.interfaces import Simulator
from simulators.simple_2d.vehicle import VehicleDynamics


class Simple2DSimulator(Simulator):
    """軽量2Dシミュレータ."""

    def __init__(
        self,
        initial_state: Optional[VehicleState] = None,
        dt: float = 0.1,
        wheelbase: float = 2.5,
    ) -> None:
        """初期化.

        Args:
            initial_state: 初期車両状態
            dt: シミュレーション時間刻み [s]
            wheelbase: ホイールベース [m]
        """
        self.dt = dt
        self.dynamics = VehicleDynamics(wheelbase=wheelbase)
        self.initial_state = initial_state or VehicleState(
            x=0.0, y=0.0, yaw=0.0, velocity=0.0, timestamp=0.0
        )
        self.current_state = self.initial_state

    def reset(self) -> VehicleState:
        """シミュレーションをリセット.

        Returns:
            初期車両状態
        """
        self.current_state = self.initial_state
        return self.current_state

    def step(self, action: Action) -> tuple[VehicleState, Observation, bool, dict[str, Any]]:
        """1ステップ実行.

        Args:
            action: 制御指令

        Returns:
            vehicle_state: 更新された車両状態
            observation: 観測データ（現在はダミー）
            done: エピソード終了フラグ（現在は常にFalse）
            info: 追加情報
        """
        self.current_state = self.dynamics.step(
            state=self.current_state,
            steering=action.steering,
            acceleration=action.acceleration,
            dt=self.dt,
        )

        # TODO: Implement proper observation generation based on track/obstacles
        observation = Observation(
            lateral_error=0.0,
            heading_error=0.0,
            velocity=self.current_state.velocity,
            target_velocity=0.0,
            timestamp=self.current_state.timestamp,
        )

        done = False
        info: dict[str, Any] = {}

        return self.current_state, observation, done, info

    def close(self) -> None:
        """シミュレータを終了."""
        pass

    def render(self) -> None:
        """シミュレーションを描画（未実装）."""
        pass
