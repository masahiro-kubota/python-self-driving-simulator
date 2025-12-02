"""Simulator interface for autonomous driving simulation."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from core.data import Action, Observation, SimulationLog, VehicleState

if TYPE_CHECKING:
    from core.data import SimulationResult, Trajectory
    from core.interfaces import Controller, Planner


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
    def run(
        self,
        planner: "Planner",
        controller: "Controller",
        max_steps: int = 1000,
        reference_trajectory: "Trajectory | None" = None,
    ) -> "SimulationResult":
        """シミュレーションを実行.

        Args:
            planner: プランナー
            controller: コントローラー
            max_steps: 最大ステップ数
            reference_trajectory: 参照軌道（ゴール判定用）

        Returns:
            SimulationResult: シミュレーション結果
        """

    @abstractmethod
    def close(self) -> None:
        """シミュレータを終了."""

    @abstractmethod
    def render(self) -> Any | None:
        """シミュレーションを描画(オプション).

        Returns:
            描画結果(画像など)、またはNone
        """

    @abstractmethod
    def get_log(self) -> SimulationLog:
        """シミュレーションログを取得.

        Returns:
            SimulationLog: シミュレーションログ
        """
