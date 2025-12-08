"""Internal vehicle state for simulators."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.data import Action, VehicleState


@dataclass
class SimulationVehicleState:
    """シミュレーター内部で使用する完全な車両状態.

    センサーから取得できるVehicleStateよりも詳細な情報を含む。
    3次元位置・姿勢、速度、加速度などを管理。
    """

    # 位置 (3D)
    x: float  # X座標 [m]
    y: float  # Y座標 [m]
    z: float = 0.0  # Z座標 [m]

    # 姿勢 (3D)
    roll: float = 0.0  # ロール角 [rad]
    pitch: float = 0.0  # ピッチ角 [rad]
    yaw: float = 0.0  # ヨー角 [rad]

    # 速度 (3D)
    vx: float = 0.0  # X方向速度 [m/s]
    vy: float = 0.0  # Y方向速度 [m/s]
    vz: float = 0.0  # Z方向速度 [m/s]

    # 角速度 (3D)
    roll_rate: float = 0.0  # ロールレート [rad/s]
    pitch_rate: float = 0.0  # ピッチレート [rad/s]
    yaw_rate: float = 0.0  # ヨーレート [rad/s]

    # 加速度 (3D)
    ax: float = 0.0  # X方向加速度 [m/s^2]
    ay: float = 0.0  # Y方向加速度 [m/s^2]
    az: float = 0.0  # Z方向加速度 [m/s^2]

    # 入力
    steering: float = 0.0  # ステアリング角 [rad]
    throttle: float = 0.0  # スロットル [-1.0 to 1.0]

    # タイムスタンプ
    timestamp: float | None = None

    @property
    def velocity(self) -> float:
        """合成速度 [m/s]."""
        return (self.vx**2 + self.vy**2 + self.vz**2) ** 0.5

    @property
    def velocity_2d(self) -> float:
        """2D平面での合成速度 [m/s]."""
        return (self.vx**2 + self.vy**2) ** 0.5

    @property
    def beta(self) -> float:
        """車体横滑り角 [rad]."""
        if abs(self.vx) < 0.1:
            return 0.0
        import math

        return math.atan2(self.vy, self.vx)

    @classmethod
    def from_vehicle_state(cls, state: "VehicleState") -> "SimulationVehicleState":
        """VehicleStateからインスタンスを生成.

        Args:
            state: VehicleState

        Returns:
            SimulationVehicleState
        """
        import math

        # 横滑り角を0と仮定してvx, vyを計算
        vx = state.velocity * math.cos(0.0)
        vy = state.velocity * math.sin(0.0)

        return cls(
            # 位置 (2D -> 3D, z=0)
            x=state.x,
            y=state.y,
            z=0.0,
            # 姿勢 (yawのみ -> 3D, roll=pitch=0)
            roll=0.0,
            pitch=0.0,
            yaw=state.yaw,
            # 速度 (スカラー -> 3Dベクトル, vz=0)
            vx=vx,
            vy=vy,
            vz=0.0,
            # 角速度 (全て0)
            roll_rate=0.0,
            pitch_rate=0.0,
            yaw_rate=0.0,
            # 加速度
            ax=state.acceleration or 0.0,
            ay=0.0,
            az=0.0,
            # 入力
            steering=state.steering or 0.0,
            throttle=0.0,
            # タイムスタンプ
            timestamp=state.timestamp,
        )

    def to_vehicle_state(self, action: "Action | None" = None) -> "VehicleState":
        """VehicleStateに変換.

        Args:
            action: 実行されたアクション（指定された場合、加速度とステアリングを上書き）

        Returns:
            VehicleState
        """
        from core.data import VehicleState

        acceleration = self.ax
        steering = self.steering

        if action is not None:
            acceleration = action.acceleration
            steering = action.steering

        return VehicleState(
            x=self.x,
            y=self.y,
            yaw=self.yaw,
            velocity=self.velocity_2d,
            acceleration=acceleration,
            steering=steering,
            timestamp=self.timestamp,
        )
