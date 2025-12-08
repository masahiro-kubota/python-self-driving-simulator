"""Autonomous driving component interfaces."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from core.data import VehicleParameters

if TYPE_CHECKING:
    from core.interfaces.node import Node


class ADComponent(ABC):
    """自動運転コンポーネントの統合インターフェース.

    ノードプロバイダーとして機能し、実行可能なノードのリストを提供する。
    具体的なノード構成は各実装クラスで定義する。
    """

    def __init__(self, vehicle_params: VehicleParameters, **_kwargs: Any) -> None:
        """初期化.

        Args:
            vehicle_params: 車両パラメータ
            **_kwargs: コンポーネント固有のパラメータ
        """
        self.vehicle_params = vehicle_params

    @abstractmethod
    def get_schedulable_nodes(self) -> list["Node"]:
        """スケジュール可能なノードのリストを返す.

        Returns:
            List of Nodes to be executed by the executor.
        """

    @abstractmethod
    def reset(self) -> bool:
        """Reset the component.

        Returns:
            bool: True if reset was successful
        """
