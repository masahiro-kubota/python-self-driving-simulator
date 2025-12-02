"""Experiment Runner interface."""

from abc import ABC, abstractmethod

from core.data.experiment.result import ExperimentResult


class ExperimentRunner(ABC):
    """実験実行のインターフェース."""

    @abstractmethod
    def run(self) -> ExperimentResult:
        """実験を実行する.

        Returns:
            ExperimentResult: 実験結果
        """
