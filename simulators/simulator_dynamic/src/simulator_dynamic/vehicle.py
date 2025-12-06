"""Vehicle dynamics model for dynamic simulator."""

import math

from simulator_core.data import DynamicVehicleState

from core.data import VehicleParameters
from core.utils.geometry import normalize_angle
from simulator_dynamic.tire_model import LinearTireModel


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

    def calculate_derivative(
        self,
        state: DynamicVehicleState,
        steering: float,
        throttle: float,
    ) -> DynamicVehicleState:
        """状態の微分を計算.

        Args:
            state: 現在の車両状態
            steering: ステアリング角 [rad]
            throttle: スロットル入力 [-1.0 to 1.0]

        Returns:
            状態の微分値 (DynamicVehicleState形式)
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
        fyf = self.tire_front.lateral_force(alpha_f, 0.0)
        fyr = self.tire_rear.lateral_force(alpha_r, 0.0)

        # 縦方向力(簡易モデル: スロットルに比例)
        fx = throttle * p.max_drive_force if throttle >= 0 else throttle * p.max_brake_force

        # 抵抗力
        f_drag = p.c_drag * state.vx**2  # 空気抵抗
        f_roll = p.c_roll * p.mass * 9.81  # 転がり抵抗

        # 運動方程式
        # x_dot, y_dot, yaw_dot
        x_dot = state.vx * math.cos(state.yaw) - state.vy * math.sin(state.yaw)
        y_dot = state.vx * math.sin(state.yaw) + state.vy * math.cos(state.yaw)
        yaw_dot = state.yaw_rate

        # vx_dot, vy_dot, yaw_rate_dot
        vx_dot = (
            fx - f_drag - f_roll - fyf * math.sin(steering)
        ) / p.mass + state.vy * state.yaw_rate
        vy_dot = (fyf * math.cos(steering) + fyr) / p.mass - state.vx * state.yaw_rate
        yaw_rate_dot = (fyf * p.lf * math.cos(steering) - fyr * p.lr) / p.inertia

        return DynamicVehicleState(
            # 位置の微分 (2D, z_dot=0)
            x=x_dot,
            y=y_dot,
            z=0.0,
            # 姿勢の微分 (yawのみ, roll_dot=pitch_dot=0)
            roll=0.0,
            pitch=0.0,
            yaw=yaw_dot,
            # 速度の微分 (2D, vz_dot=0)
            vx=vx_dot,
            vy=vy_dot,
            vz=0.0,
            # 角速度の微分 (yaw_rateのみ)
            roll_rate=0.0,
            pitch_rate=0.0,
            yaw_rate=yaw_rate_dot,
            # 加速度の微分 (0)
            ax=0.0,
            ay=0.0,
            az=0.0,
            # 入力の微分 (0)
            steering=0.0,
            throttle=0.0,
            # タイムスタンプの微分
            timestamp=0.0,
        )

    @staticmethod
    def add_state(
        state: DynamicVehicleState, derivative: DynamicVehicleState, dt: float
    ) -> DynamicVehicleState:
        """状態への微分の加算 (integration用).

        Args:
            state: ベースとなる状態
            derivative: 微分値
            dt: 時間刻み

        Returns:
            更新された状態
        """
        # タイムスタンプの更新
        next_timestamp: float | None = None
        if state.timestamp is not None:
            next_timestamp = state.timestamp + dt

        # 角度の更新と正規化
        next_yaw = normalize_angle(state.yaw + derivative.yaw * dt)

        return DynamicVehicleState(
            # 位置
            x=state.x + derivative.x * dt,
            y=state.y + derivative.y * dt,
            z=state.z + derivative.z * dt,
            # 姿勢
            roll=state.roll + derivative.roll * dt,
            pitch=state.pitch + derivative.pitch * dt,
            yaw=next_yaw,
            # 速度
            vx=state.vx + derivative.vx * dt,
            vy=state.vy + derivative.vy * dt,
            vz=state.vz + derivative.vz * dt,
            # 角速度
            roll_rate=state.roll_rate + derivative.roll_rate * dt,
            pitch_rate=state.pitch_rate + derivative.pitch_rate * dt,
            yaw_rate=state.yaw_rate + derivative.yaw_rate * dt,
            # 加速度
            ax=state.ax + derivative.ax * dt,
            ay=state.ay + derivative.ay * dt,
            az=state.az + derivative.az * dt,
            # 入力は維持
            steering=state.steering,
            throttle=state.throttle,
            # タイムスタンプ
            timestamp=next_timestamp,
        )
