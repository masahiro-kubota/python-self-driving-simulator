"""Autonomous driving component interfaces."""

from abc import ABC, abstractmethod
from typing import Any

from ad_components_core.data import Observation

from core.data import Action, VehicleState
from core.data.ad_components import Trajectory


class Perception(ABC):
    """認識コンポーネントの抽象基底クラス."""

    @abstractmethod
    def perceive(self, sensor_data: Any) -> Observation:
        """センサーデータから観測を生成.

        Args:
            sensor_data: センサーデータ(カメラ画像、LiDARなど)

        Returns:
            Observation: 観測データ
        """

    @abstractmethod
    def reset(self) -> None:
        """認識コンポーネントをリセット."""


class Planner(ABC):
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


class Controller(ABC):
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
            observation: 観測データ(オプション)

        Returns:
            Action: 制御指令
        """

    @abstractmethod
    def reset(self) -> None:
        """制御コンポーネントをリセット."""
