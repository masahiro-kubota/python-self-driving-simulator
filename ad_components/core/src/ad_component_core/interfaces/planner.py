"""Planner interface."""

from abc import ABC, abstractmethod

# Import from ad_components_core.data instead from abc import ABC, abstractmethod
from ad_component_core.data import Observation
from core.data import VehicleState
from core.data.ad_components import Trajectory


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
