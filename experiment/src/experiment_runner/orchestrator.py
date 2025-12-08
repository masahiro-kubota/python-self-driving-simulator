from pathlib import Path
from typing import Any

from experiment_runner.interfaces import ExperimentPostprocessor, ExperimentRunner
from experiment_runner.postprocessing.postprocessor_factory import (
    DefaultPostprocessorFactory,
)
from experiment_runner.preprocessing.loader import DefaultPreprocessor
from experiment_runner.runner.runner_factory import DefaultRunnerFactory


class ExperimentOrchestrator:
    """実験全体のオーケストレーター

    - Preprocessorは具体的な実装を直接持つ（動的に変わらない）
    - RunnerとPostprocessorはFactoryで動的生成（実験タイプに応じて変わる）
    """

    def __init__(self) -> None:
        # Preprocessorは具体的な実装を直接組み込み
        self.preprocessor = DefaultPreprocessor()

        # RunnerとPostprocessorはFactoryで動的生成
        self.runner_factory = DefaultRunnerFactory()
        self.postprocessor_factory = DefaultPostprocessorFactory()

    def run(self, config_path: Path) -> Any:
        """実験を実行

        1. 前処理: 設定読み込み、コンポーネント生成（固定）
        2. 実行: 設定から実験タイプを判定してRunnerを生成・実行（動的）
        3. 後処理: 実験タイプに応じたPostprocessorを生成・実行(動的)
        """
        # 1. 前処理(固定)
        config = self.preprocessor.load_config(config_path)
        components = self.preprocessor.setup_components(config)

        # 2. 実験タイプに応じたRunnerを動的生成
        experiment_type = config.experiment.type
        runner: ExperimentRunner = self.runner_factory.create(experiment_type)

        # 3. 実行
        result = runner.run(config, components)

        # 4. 実験タイプに応じたPostprocessorを動的生成
        postprocessor: ExperimentPostprocessor = self.postprocessor_factory.create(experiment_type)

        # 5. 後処理
        # Note: postprocessor.process expects (result, config)
        processed_result = postprocessor.process(result, config)

        return processed_result
