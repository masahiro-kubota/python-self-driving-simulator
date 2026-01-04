"""Vehicle parameters management."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from core.data.node import ComponentConfig


class LidarConfig(ComponentConfig):
    """Configuration for LiDAR sensor."""

    num_beams: int = Field(..., gt=0)
    fov: float = Field(..., gt=0)
    range_min: float = Field(...)
    range_max: float = Field(...)
    angle_increment: float = Field(...)
    # Mounting position relative to vehicle center
    x: float = Field(...)
    y: float = Field(...)
    z: float = Field(...)
    yaw: float = Field(...)
    publish_rate_hz: float = Field(
        ..., description="LiDAR data publish rate [Hz]. Must be <= simulator rate_hz"
    )


class VehicleParameters(BaseModel):
    """統一車両パラメータ定義.

    運動学・動力学シミュレータの両方で使用可能な車両パラメータ。
    """

    # 基本形状
    wheelbase: float  # ホイールベース [m]
    width: float  # 車幅 [m]
    vehicle_height: float  # 車高 [m] (default to 2.2 if missing)

    @property
    def length(self) -> float:
        """車長 (computed) [m]."""
        return self.wheelbase + self.front_overhang + self.rear_overhang

    # 運動学パラメータ
    max_steering_angle: float  # 最大操舵角 [rad]
    max_velocity: float  # 最大速度 [m/s]
    max_acceleration: float  # 最大加速度 [m/s^2]

    # 寸法詳細パラメータ (オプション、未指定時は length, wheelbase から推定)
    front_overhang: float  # フロントオーバーハング [m]
    rear_overhang: float  # リアオーバーハング [m]

    # ステアリング応答パラメータ
    steer_delay_time: float = Field(description="純粋な時間遅れ [秒]")

    max_steer_rate: float = Field(description="ステア角の最大変化率 [rad/s]")

    steer_gain: float = Field(description="ステアリングゲイン (DCゲイン)")

    steer_zeta: float = Field(default=0.7, description="SOPDTモデルの減衰比 (damping ratio)")

    steer_omega_n: float = Field(
        default=5.0, description="SOPDTモデルの固有角周波数 (natural frequency) [rad/s]"
    )

    steer_tau: float = Field(default=0.0, description="FOPDTモデルの時定数 (time constant) [秒]")

    # Longitudinal Dynamics Parameters
    accel_gain: float = Field(default=1.0, description="Longitudinal Acceleration Gain")
    accel_offset: float = Field(default=0.0, description="Longitudinal Acceleration Offset [m/s^2]")
    drag_coefficient: float = Field(default=0.0, description="Air Drag term C*v*|v|")
    cornering_drag_coefficient: float = Field(default=0.0, description="Cornering Drag term C*|s|*v^2")

    # センサー設定
    lidar: LidarConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換.

        Returns:
            dict: 車両パラメータの辞書
        """
        return self.model_dump()
