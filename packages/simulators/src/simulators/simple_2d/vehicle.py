"""Vehicle dynamics model for simple 2D simulator."""

import math
from typing import Optional

from core.data import VehicleState
from core.utils.geometry import normalize_angle


class VehicleDynamics:
    """キネマティック自転車モデルに基づく車両ダイナミクス."""

    def __init__(self, wheelbase: float = 2.5) -> None:
        """初期化.

        Args:
            wheelbase: ホイールベース [m]
        """
        self.wheelbase = wheelbase

    def step(
        self,
        state: VehicleState,
        steering: float,
        acceleration: float,
        dt: float,
    ) -> VehicleState:
        """1ステップ更新.

        Args:
            state: 現在の車両状態
            steering: ステアリング角 [rad]
            acceleration: 加速度 [m/s^2]
            dt: 時間刻み [s]

        Returns:
            更新された車両状態（新しいインスタンス）
        """
        # Kinematic bicycle model equations
        # x_dot = v * cos(yaw)
        # y_dot = v * sin(yaw)
        # yaw_dot = v / L * tan(delta)
        # v_dot = a

        x_next = state.x + state.velocity * math.cos(state.yaw) * dt
        y_next = state.y + state.velocity * math.sin(state.yaw) * dt
        yaw_next = state.yaw + state.velocity / self.wheelbase * math.tan(steering) * dt
        velocity_next = state.velocity + acceleration * dt

        # Normalize yaw
        yaw_next = normalize_angle(yaw_next)
        
        # Update timestamp if present
        timestamp_next: Optional[float] = None
        if state.timestamp is not None:
            timestamp_next = state.timestamp + dt

        return VehicleState(
            x=x_next,
            y=y_next,
            yaw=yaw_next,
            velocity=velocity_next,
            acceleration=acceleration,
            steering=steering,
            timestamp=timestamp_next,
        )
