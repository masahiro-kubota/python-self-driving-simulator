"""Vehicle parameters management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass
class VehicleParameters:
    """統一車両パラメータ定義.

    運動学・動力学シミュレータの両方で使用可能な車両パラメータ。
    """

    # 基本形状
    wheelbase: float  # ホイールベース [m]
    width: float  # 車幅 [m]

    @property
    def length(self) -> float:
        """車長 (computed) [m]."""
        return self.wheelbase + self.front_overhang + self.rear_overhang

    # 運動学パラメータ
    max_steering_angle: float  # 最大操舵角 [rad]
    max_velocity: float  # 最大速度 [m/s]
    max_acceleration: float  # 最大加速度 [m/s^2]

    # 動力学パラメータ(オプション)
    mass: float  # 質量 [kg]
    inertia: float  # ヨー慣性モーメント [kg*m^2]

    # NOTE: 以下は重心位置パラメータ
    lf: float  # 重心から前軸までの距離 [m]
    lr: float  # 重心から後軸までの距離 [m]

    # NOTE: 以下はタイヤ特性パラメータ
    cf: float  # 前輪コーナリング剛性 [N/rad]
    cr: float  # 後輪コーナリング剛性 [N/rad]

    # NOTE: 以下は抵抗係数パラメータ
    c_drag: float  # 空気抵抗係数
    c_roll: float  # 転がり抵抗係数

    # NOTE: 以下は駆動力パラメータ
    max_drive_force: float  # 最大駆動力 [N]
    max_brake_force: float  # 最大制動力 [N]

    # 寸法詳細パラメータ (オプション、未指定時は length, wheelbase から推定)
    front_overhang: float  # フロントオーバーハング [m]
    rear_overhang: float  # リアオーバーハング [m]

    tire_params: dict[str, Any] = field(default_factory=dict)  # タイヤパラメータ(将来の拡張用)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換.

        Returns:
            dict: 車両パラメータの辞書
        """
        return {
            "wheelbase": self.wheelbase,
            "width": self.width,
            "length": self.length,
            "max_steering_angle": self.max_steering_angle,
            "max_velocity": self.max_velocity,
            "max_acceleration": self.max_acceleration,
            "mass": self.mass,
            "inertia": self.inertia,
            "lf": self.lf,
            "lr": self.lr,
            "cf": self.cf,
            "cr": self.cr,
            "c_drag": self.c_drag,
            "c_roll": self.c_roll,
            "max_drive_force": self.max_drive_force,
            "max_brake_force": self.max_brake_force,
            "front_overhang": self.front_overhang,
            "rear_overhang": self.rear_overhang,
            "tire_params": self.tire_params,
        }
