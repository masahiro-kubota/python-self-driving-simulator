"""AD Component interface."""

from abc import ABC, abstractmethod

from core.data.ad_components.action import Action
from core.data.ad_components.sensing import Sensing


class ADComponent(ABC):
    """自動運転コンポーネントのインターフェース."""

    @abstractmethod
    def run(self, sensing: Sensing) -> Action:
        """コンポーネントを実行して制御指令を生成する.

        Args:
            sensing: センシングデータ（車両状態、障害物など）

        Returns:
            Action: 制御指令
        """
