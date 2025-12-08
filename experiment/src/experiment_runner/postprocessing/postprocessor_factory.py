from experiment_runner.interfaces import ExperimentPostprocessor, ExperimentPostprocessorFactory
from experiment_runner.postprocessing.evaluation_postprocessor import EvaluationPostprocessor


class DefaultPostprocessorFactory(ExperimentPostprocessorFactory):
    """デフォルトのPostprocessorファクトリ"""

    def create(self, experiment_type: str) -> ExperimentPostprocessor:
        """実験タイプに応じたPostprocessorを生成"""
        if experiment_type == "evaluation":
            return EvaluationPostprocessor()
        elif experiment_type == "training":
            # return TrainingPostprocessor()  # 将来実装
            raise NotImplementedError("Training mode not implemented yet")
        elif experiment_type == "data_collection":
            raise NotImplementedError("Data collection mode not implemented yet")
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
