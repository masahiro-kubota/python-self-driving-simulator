"""Vehicle parameters management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class VehicleParameters:
    """統一車両パラメータ定義.

    運動学・動力学シミュレータの両方で使用可能な車両パラメータ。
    """

    # 基本形状
    wheelbase: float = 2.5  # ホイールベース [m]
    width: float = 1.8  # 車幅 [m]
    length: float = 4.5  # 車長 [m]

    # 運動学パラメータ
    max_steering_angle: float = 0.6  # 最大操舵角 [rad]
    max_velocity: float = 20.0  # 最大速度 [m/s]
    max_acceleration: float = 3.0  # 最大加速度 [m/s^2]

    # 動力学パラメータ(DynamicSimulator用、オプション)
    mass: float | None = None  # 質量 [kg]
    inertia: float | None = None  # ヨー慣性モーメント [kg*m^2]

    # NOTE: 以下は重心位置パラメータ (DynamicSimulator用)
    lf: float | None = None  # 重心から前軸までの距離 [m]
    lr: float | None = None  # 重心から後軸までの距離 [m]

    # NOTE: 以下はタイヤ特性パラメータ (DynamicSimulator用)
    cf: float | None = None  # 前輪コーナリング剛性 [N/rad]
    cr: float | None = None  # 後輪コーナリング剛性 [N/rad]

    # NOTE: 以下は抵抗係数パラメータ (DynamicSimulator用)
    c_drag: float | None = None  # 空気抵抗係数
    c_roll: float | None = None  # 転がり抵抗係数

    # NOTE: 以下は駆動力パラメータ (DynamicSimulator用)
    max_drive_force: float | None = None  # 最大駆動力 [N]
    max_brake_force: float | None = None  # 最大制動力 [N]

    tire_params: dict[str, Any] = field(default_factory=dict)  # タイヤパラメータ(将来の拡張用)

    @classmethod
    def from_yaml(cls, path: Path) -> VehicleParameters:
        """YAMLファイルから車両パラメータを読み込む.

        Args:
            path: YAMLファイルのパス

        Returns:
            VehicleParameters: 車両パラメータ
        """
        with path.open() as f:
            data = yaml.safe_load(f)
        return cls(**data)

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
            "tire_params": self.tire_params,
        }
