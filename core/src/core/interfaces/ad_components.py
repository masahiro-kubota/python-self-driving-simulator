"""Autonomous driving component interfaces."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from core.data import Action, Observation, VehicleParameters, VehicleState
from core.data.ad_components import Trajectory
from core.data.ad_components.sensing import Sensing

if TYPE_CHECKING:
    from core.interfaces.node import Node


class ADComponent(ABC):
    """自動運転コンポーネントの統合インターフェース.

    PlannerとControllerを内部で管理し、統合的な実行を提供する。
    """

    def __init__(self, vehicle_params: VehicleParameters, **kwargs: Any) -> None:
        """初期化.

        Args:
            vehicle_params: 車両パラメータ
            **kwargs: コンポーネント固有のパラメータ
        """
        self.vehicle_params = vehicle_params
        self.planner = self._create_planner(**kwargs)
        self.controller = self._create_controller(**kwargs)

    @abstractmethod
    def _create_planner(self, **kwargs: Any) -> "Planner":
        """Plannerを作成（サブクラスで実装）.

        Args:
            **kwargs: Planner固有のパラメータ

        Returns:
            Planner: 作成されたPlanner
        """

    @abstractmethod
    def _create_controller(self, **kwargs: Any) -> "Controller":
        """Controllerを作成（サブクラスで実装）.

        Args:
            **kwargs: Controller固有のパラメータ

        Returns:
            Controller: 作成されたController
        """

    def run(self, sensing: Sensing) -> Action:
        """コンポーネントを実行して制御指令を生成する.

        Args:
            sensing: センシングデータ（車両状態、障害物など）

        Returns:
            Action: 制御指令
        """
        # 1. 認識 (現在は未実装、将来の拡張用)
        observation = Observation()

        # 2. 計画
        trajectory = self.planner.plan(observation, sensing.vehicle_state)

        # 3. 制御
        action = self.controller.control(trajectory, sensing.vehicle_state, observation)

        return action

    def reset(self) -> None:
        """コンポーネントをリセット."""
        self.planner.reset()
        self.controller.reset()

    @abstractmethod
    def get_schedulable_nodes(self) -> list["Node"]:
        """Get list of schedulable nodes for this component.

        Returns:
            List of Nodes to be executed by the executor.
        """


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
