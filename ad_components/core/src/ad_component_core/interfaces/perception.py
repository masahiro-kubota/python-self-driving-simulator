"""Perception interface."""

from abc import ABC, abstractmethod
from typing import Any

from ad_component_core.data import Observation
from core.data import VehicleState


class Perception(ABC):
    """認識コンポーネントの抽象基底クラス."""

    @abstractmethod
    def perceive(
        self,
        vehicle_state: VehicleState,
        sensor_data: Any,
    ) -> Observation:
        """センサーデータから環境を認識.

        Args:
            vehicle_state: 車両状態
            sensor_data: センサーデータ

        Returns:
            Observation: 観測結果
        """

    @abstractmethod
    def reset(self) -> None:
        """認識コンポーネントをリセット."""
