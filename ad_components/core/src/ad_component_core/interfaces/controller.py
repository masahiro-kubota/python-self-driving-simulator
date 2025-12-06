"""Controller interface."""

from abc import ABC, abstractmethod

# Import from ad_components_core.data instead from abc import ABC, abstractmethod
from ad_component_core.data import Observation
from core.data import Action, VehicleState
from core.data.ad_components import Trajectory


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
