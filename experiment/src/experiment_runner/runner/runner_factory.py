from experiment_runner.interfaces import ExperimentRunner, ExperimentRunnerFactory
from experiment_runner.runner.evaluation import EvaluationRunner


class DefaultRunnerFactory(ExperimentRunnerFactory):
    """デフォルトのRunnerファクトリ"""

    def create(self, experiment_type: str) -> ExperimentRunner:
        """実験タイプに応じたRunnerを生成"""
        if experiment_type == "evaluation":
            return EvaluationRunner()
        elif experiment_type == "training":
            # return TrainingRunner()  # 将来実装
            raise NotImplementedError("Training mode not implemented yet")
        elif experiment_type == "data_collection":
            raise NotImplementedError("Data collection mode not implemented yet")
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
