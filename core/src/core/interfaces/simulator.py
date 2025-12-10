"""Simulator interface for autonomous driving simulation."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.data import Action, SimulationLog, VehicleState

if TYPE_CHECKING:
    from core.data import SimulationResult


class Simulator(ABC):
    """シミュレータの抽象基底クラス."""

    @abstractmethod
    def reset(self) -> VehicleState:
        """シミュレーションをリセット.

        Returns:
            VehicleState: 初期車両状態
        """

    @abstractmethod
    def step(self, action: Action) -> tuple[VehicleState, bool, dict[str, Any]]:
        """シミュレーションを1ステップ進める.

        Args:
            action: 実行するアクション

        Returns:
            tuple[VehicleState, bool, dict[str, Any]]:
                - next_state: 次の車両状態
                - done: エピソード終了フラグ
                - info: 追加情報
        """

    @abstractmethod
    def run(
        self,
        ad_component: Any,
        max_steps: int = 1000,
    ) -> "SimulationResult":
        """シミュレーションを実行.

        Args:
            ad_component: AD component instance (planner + controller)
            max_steps: 最大ステップ数

        Returns:
            SimulationResult: シミュレーション結果
        """

    @abstractmethod
    def close(self) -> bool:
        """シミュレータを終了.

        Returns:
            bool: 終了処理が成功した場合True
        """

    @abstractmethod
    def get_log(self) -> SimulationLog:
        """シミュレーションログを取得.

        Returns:
            SimulationLog: シミュレーションログ
        """


class SimulationLogRepository(ABC):
    """Interface for simulation log persistence.

    This interface abstracts the storage and retrieval of simulation logs,
    allowing different implementations (JSON, MCAP, database, etc.) without
    affecting the code that uses simulation logs.
    """

    @abstractmethod
    def save(self, log: SimulationLog, file_path: Path) -> bool:
        """Save simulation log to file.

        Args:
            log: Simulation log to save
            file_path: Output file path

        Returns:
            bool: True if save was successful
        """

    @abstractmethod
    def load(self, file_path: Path) -> SimulationLog:
        """Load simulation log from file.

        Args:
            file_path: Input file path

        Returns:
            SimulationLog object
        """
