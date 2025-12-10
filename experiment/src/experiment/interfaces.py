from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from experiment.structures import Experiment

TConfig = TypeVar("TConfig")
TResult = TypeVar("TResult")
TProcessed = TypeVar("TProcessed")


class ExperimentRunner(ABC, Generic[TConfig, TResult]):
    """実験実行インターフェース（実験タイプごとに実装）"""

    @abstractmethod
    def run(self, experiment: Experiment) -> TResult:
        """実験を実行

        Args:
            experiment: 実行する実験定義 (Experiment)
        """
        pass

    @abstractmethod
    def get_type(self) -> str:
        """実験タイプを返す（"evaluation", "training"等）"""
        pass


class ExperimentRunnerFactory(ABC, Generic[TConfig]):
    """Runner生成ファクトリ（Factory Pattern）"""

    @abstractmethod
    def create(self, experiment_type: str) -> ExperimentRunner:
        """実験タイプに応じたRunnerを生成"""
        pass


class ExperimentPostprocessor(ABC, Generic[TResult, TProcessed]):
    """後処理インターフェース（実験タイプごとに実装）"""

    @abstractmethod
    def process(self, result: TResult, config: Any) -> TProcessed:
        """結果の保存、分析、可視化

        Args:
            result: 実験結果
            config: 実験設定（ログ記録に必要）

        Returns:
            処理済みの結果
        """
        pass


class ExperimentPostprocessorFactory(ABC):
    """Postprocessor生成ファクトリ（Factory Pattern）"""

    @abstractmethod
    def create(self, experiment_type: str) -> ExperimentPostprocessor:
        """実験タイプに応じたPostprocessorを生成"""
        pass
