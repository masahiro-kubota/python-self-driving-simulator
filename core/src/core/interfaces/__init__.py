"""Abstract interfaces for autonomous driving components."""

from abc import ABC, abstractmethod
from typing import Any

from core.data import Action, Observation, Trajectory, VehicleState
from core.interfaces.dashboard import DashboardGenerator


class PerceptionComponent(ABC):
    """認識コンポーネントの抽象基底クラス."""

    @abstractmethod
    def perceive(self, sensor_data: Any) -> Observation:
        """センサーデータから観測を生成.

        Args:
            sensor_data: センサーデータ（カメラ画像、LiDARなど）

        Returns:
            Observation: 観測データ
        """

    @abstractmethod
    def reset(self) -> None:
        """認識コンポーネントをリセット."""


class PlanningComponent(ABC):
    """計画コンポーネントの抽象基底クラス."""

    @abstractmethod
    def plan(
        self,
        observation: Observation,
        vehicle_state: VehicleState,
    ) -> Trajectory:
        """観測と車両状態から軌道を生成.

        Args:
            observation: 観測データ
            vehicle_state: 車両状態

        Returns:
            Trajectory: 計画された軌道
        """

    @abstractmethod
    def reset(self) -> None:
        """計画コンポーネントをリセット."""


class ControlComponent(ABC):
    """制御コンポーネントの抽象基底クラス."""

    @abstractmethod
    def control(
        self,
        trajectory: Trajectory,
        vehicle_state: VehicleState,
        observation: Observation | None = None,
    ) -> Action:
        """軌道と車両状態から制御指令を生成.

        Args:
            trajectory: 目標軌道
            vehicle_state: 車両状態
            observation: 観測データ（オプション）

        Returns:
            Action: 制御指令
        """

    @abstractmethod
    def reset(self) -> None:
        """制御コンポーネントをリセット."""


class Simulator(ABC):
    """シミュレータの抽象基底クラス."""

    @abstractmethod
    def reset(self) -> VehicleState:
        """シミュレーションをリセット.

        Returns:
            VehicleState: 初期車両状態
        """

    @abstractmethod
    def step(self, action: Action) -> tuple[VehicleState, Observation, bool, dict[str, Any]]:
        """1ステップ実行.

        Args:
            action: 制御指令

        Returns:
            vehicle_state: 更新された車両状態
            observation: 観測データ
            done: エピソード終了フラグ
            info: 追加情報
        """

    @abstractmethod
    def close(self) -> None:
        """シミュレータを終了."""

    @abstractmethod
    def render(self) -> Any | None:
        """シミュレーションを描画（オプション）.

        Returns:
            描画結果（画像など）、またはNone
        """


__all__ = [
    "ControlComponent",
    "DashboardGenerator",
    "PerceptionComponent",
    "PlanningComponent",
    "Simulator",
]
