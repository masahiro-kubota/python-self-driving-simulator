"""Experiment interfaces."""

from abc import ABC, abstractmethod

from core.data.experiment import ExperimentResult


class ExperimentLogger(ABC):
    """実験実行のインターフェース."""

    @abstractmethod
    def run(self) -> ExperimentResult:
        """実験を実行する.

        Returns:
            ExperimentResult: 実験結果
        """

    @abstractmethod
    def log_result(self, result: ExperimentResult) -> bool:
        """実験結果をログに記録する.

        Args:
            result: 記録する実験結果

        Returns:
            bool: ログ記録が成功した場合True
        """
