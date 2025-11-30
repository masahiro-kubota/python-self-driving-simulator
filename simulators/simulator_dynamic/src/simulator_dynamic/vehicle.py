"""Vehicle dynamics model for dynamic simulator."""

import math

from core.utils.geometry import normalize_angle
from simulator_dynamic.state import DynamicVehicleState
from simulator_dynamic.tire_model import LinearTireModel
from simulator_dynamic.vehicle_params import VehicleParameters


class DynamicVehicleModel:
    """ダイナミック自転車モデルに基づく車両ダイナミクス."""

    def __init__(self, params: VehicleParameters | None = None) -> None:
        """初期化.

        Args:
            params: 車両パラメータ
        """
        self.params = params or VehicleParameters()
        self.tire_front = LinearTireModel(self.params.cf)
        self.tire_rear = LinearTireModel(self.params.cr)

    def step(
        self,
        state: DynamicVehicleState,
        steering: float,
        throttle: float,
        dt: float,
    ) -> DynamicVehicleState:
        """1ステップ更新(Runge-Kutta 4次).

        Args:
            state: 現在の車両状態
            steering: ステアリング角 [rad]
            throttle: スロットル入力 [-1.0 to 1.0]
            dt: 時間刻み [s]

        Returns:
            更新された車両状態
        """
        # RK4積分
        k1 = self._derivatives(state, steering, throttle)

        state2 = self._add_derivatives(state, k1, dt / 2)
        k2 = self._derivatives(state2, steering, throttle)

        state3 = self._add_derivatives(state, k2, dt / 2)
        k3 = self._derivatives(state3, steering, throttle)

        state4 = self._add_derivatives(state, k3, dt)
        k4 = self._derivatives(state4, steering, throttle)

        # 状態更新
        x_next = state.x + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) * dt / 6
        y_next = state.y + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) * dt / 6
        yaw_next = state.yaw + (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) * dt / 6
        vx_next = state.vx + (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) * dt / 6
        vy_next = state.vy + (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4]) * dt / 6
        yaw_rate_next = state.yaw_rate + (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5]) * dt / 6

        yaw_next = normalize_angle(yaw_next)

        timestamp_next = None
        if state.timestamp is not None:
            timestamp_next = state.timestamp + dt

        return DynamicVehicleState(
            x=x_next,
            y=y_next,
            yaw=yaw_next,
            vx=vx_next,
            vy=vy_next,
            yaw_rate=yaw_rate_next,
            steering=steering,
            throttle=throttle,
            timestamp=timestamp_next,
        )

    def _derivatives(
        self,
        state: DynamicVehicleState,
        steering: float,
        throttle: float,
    ) -> tuple[float, float, float, float, float, float]:
        """状態の微分を計算.

        Returns:
            (x_dot, y_dot, yaw_dot, vx_dot, vy_dot, yaw_rate_dot)
        """
        p = self.params

        # 前輪・後輪の横滑り角
        if abs(state.vx) < 0.1:
            alpha_f = 0.0
            alpha_r = 0.0
        else:
            alpha_f = steering - math.atan2(state.vy + p.lf * state.yaw_rate, state.vx)
            alpha_r = -math.atan2(state.vy - p.lr * state.yaw_rate, state.vx)

        # タイヤの横方向力(線形タイヤモデル)
        # 垂直荷重は簡略化のため考慮しない
        fyf = self.tire_front.lateral_force(alpha_f, 0.0)
        fyr = self.tire_rear.lateral_force(alpha_r, 0.0)

        # 縦方向力(簡易モデル: スロットルに比例)
        fx = throttle * p.max_drive_force if throttle >= 0 else throttle * p.max_brake_force

        # 抵抗力
        f_drag = p.c_drag * state.vx**2  # 空気抵抗
        f_roll = p.c_roll * p.mass * 9.81  # 転がり抵抗

        # 運動方程式
        x_dot = state.vx * math.cos(state.yaw) - state.vy * math.sin(state.yaw)
        y_dot = state.vx * math.sin(state.yaw) + state.vy * math.cos(state.yaw)
        yaw_dot = state.yaw_rate

        vx_dot = (
            fx - f_drag - f_roll - fyf * math.sin(steering)
        ) / p.mass + state.vy * state.yaw_rate
        vy_dot = (fyf * math.cos(steering) + fyr) / p.mass - state.vx * state.yaw_rate
        yaw_rate_dot = (fyf * p.lf * math.cos(steering) - fyr * p.lr) / p.iz

        return (x_dot, y_dot, yaw_dot, vx_dot, vy_dot, yaw_rate_dot)

    def _add_derivatives(
        self,
        state: DynamicVehicleState,
        derivatives: tuple[float, float, float, float, float, float],
        dt: float,
    ) -> DynamicVehicleState:
        """状態に微分を加算(RK4用)."""
        return DynamicVehicleState(
            x=state.x + derivatives[0] * dt,
            y=state.y + derivatives[1] * dt,
            yaw=state.yaw + derivatives[2] * dt,
            vx=state.vx + derivatives[3] * dt,
            vy=state.vy + derivatives[4] * dt,
            yaw_rate=state.yaw_rate + derivatives[5] * dt,
            steering=state.steering,
            throttle=state.throttle,
            timestamp=state.timestamp,
        )
