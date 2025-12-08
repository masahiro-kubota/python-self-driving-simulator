"""Vehicle dynamics model for kinematic simulator."""

import math

from core.utils.geometry import normalize_angle
from simulators.core.data import SimulationVehicleState


class KinematicVehicleModel:
    """キネマティック自転車モデルに基づく車両ダイナミクス."""

    def __init__(self, wheelbase: float = 2.5) -> None:
        """初期化.

        Args:
            wheelbase: ホイールベース [m]
        """
        self.wheelbase = wheelbase

    def step(
        self,
        state: SimulationVehicleState,
        steering: float,
        acceleration: float,
        dt: float,
    ) -> SimulationVehicleState:
        """1ステップ更新.

        Args:
            state: 現在の車両状態
            steering: ステアリング角 [rad]
            acceleration: 加速度 [m/s^2]
            dt: 時間刻み [s]

        Returns:
            更新された車両状態(新しいインスタンス)
        """
        # Kinematic bicycle model equations
        # x_dot = vx * cos(yaw) - vy * sin(yaw)
        # y_dot = vx * sin(yaw) + vy * cos(yaw)
        # yaw_dot = vx / L * tan(delta)
        # vx_dot = ax

        # キネマティクスモデルでは vy = 0 を維持
        vx_next = state.vx + acceleration * dt

        # 平均速度を使用して位置を更新(より正確な積分)
        vx_avg = (state.vx + vx_next) / 2.0

        # ヨーレートを計算(平均速度を使用)
        if abs(vx_avg) < 0.01:
            yaw_rate_next = 0.0
        else:
            yaw_rate_next = vx_avg / self.wheelbase * math.tan(steering)

        # 位置・姿勢の更新(平均速度を使用)
        x_next = state.x + vx_avg * math.cos(state.yaw) * dt
        y_next = state.y + vx_avg * math.sin(state.yaw) * dt
        yaw_next = normalize_angle(state.yaw + yaw_rate_next * dt)

        # Update timestamp if present
        timestamp_next: float | None = None
        if state.timestamp is not None:
            timestamp_next = state.timestamp + dt

        return SimulationVehicleState(
            # 位置 (2D更新、z=0維持)
            x=x_next,
            y=y_next,
            z=0.0,
            # 姿勢 (yawのみ更新、roll=pitch=0維持)
            roll=0.0,
            pitch=0.0,
            yaw=yaw_next,
            # 速度 (vxのみ更新、vy=vz=0維持)
            vx=vx_next,
            vy=0.0,
            vz=0.0,
            # 角速度 (yaw_rateのみ計算、他は0維持)
            roll_rate=0.0,
            pitch_rate=0.0,
            yaw_rate=yaw_rate_next,
            # 加速度
            ax=acceleration,
            ay=0.0,
            az=0.0,
            # 入力
            steering=steering,
            throttle=0.0,
            # タイムスタンプ
            timestamp=timestamp_next,
        )
