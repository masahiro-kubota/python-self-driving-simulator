from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from experiment.interfaces import Experiment, ExperimentPostprocessor, ExperimentRunner
from experiment.postprocessing.evaluation_postprocessor import EvaluationPostprocessor
from experiment.preprocessing.loader import DefaultPreprocessor
from experiment.runner.runner_factory import DefaultRunnerFactory


class ExperimentOrchestrator:
    """実験全体のオーケストレーター

    - Preprocessorは具体的な実装を直接持つ（動的に変わらない）
    - RunnerとPostprocessorはFactoryで動的生成（実験タイプに応じて変わる）
    """

    def __init__(self) -> None:
        # Preprocessorは具体的な実装を直接組み込み
        self.preprocessor = DefaultPreprocessor()

        # RunnerとPostprocessorはFactoryで動的生成
        # TODO: RunnerもFactoryをやめて直接生成にするか検討
        self.runner_factory = DefaultRunnerFactory()

    def run(self, config_path: Path) -> Any:
        """実験を実行

        1. 前処理: 設定読み込み、コンポーネント生成（固定）
        2. 実行: 設定から実験タイプを判定してRunnerを生成・実行（動的）
        3. 後処理: 実験タイプに応じたPostprocessorを生成・実行(動的)
        """
        # 1. 前処理(実験の生成)
        # Preprocessorが単一の実験インスタンスを生成
        experiment: Experiment = self.preprocessor.create_experiment(config_path)

        # 2. 実験タイプに応じたPostprocessorを動的生成 (Fail Fastのために実行前に生成)
        if experiment.type == "evaluation":
            postprocessor: ExperimentPostprocessor = EvaluationPostprocessor(experiment.config)
        else:
            raise ValueError(f"Unknown experiment type: {experiment.type}")

        # 3. 実験タイプに応じたRunnerを動的生成
        runner: ExperimentRunner = self.runner_factory.create(experiment.type)

        # 4. 実行
        import time

        start_time = time.perf_counter()
        result = runner.run(experiment)
        simulation_time = time.perf_counter() - start_time
        print(f"Simulation execution time: {simulation_time:.2f} seconds")

        # 5. 後処理
        start_time = time.perf_counter()
        processed_result = postprocessor.process(result, experiment.config)
        postprocess_time = time.perf_counter() - start_time
        print(f"Postprocessing (Dashboard) execution time: {postprocess_time:.2f} seconds")

        return processed_result

    def run_from_hydra(self, cfg: DictConfig) -> Any:
        """Run experiment from Hydra configuration.

        Args:
            cfg: Hydra DictConfig object

        Returns:
            Processed experiment result
        """
        # Import here to avoid circular dependency
        import uuid

        from core.data.experiment.config import (
            ExecutionConfig,
            ExperimentMetadata,
            NodeConfig,
            ResolvedExperimentConfig,
        )
        from core.utils.node_factory import NodeFactory
        from experiment.structures import Experiment as ExperimentStructure

        # Convert DictConfig to dict and resolve variables
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        # Extract configuration sections
        experiment_data = cfg_dict["experiment"]
        execution_data = cfg_dict["execution"]
        postprocess_data = cfg_dict["postprocess"]
        system_data = cfg_dict["system"]

        # Merge agent nodes into system nodes (same logic as collect_data.py)
        nodes_data = system_data["nodes"]

        agent_nodes_data = []
        if "agent" in cfg_dict and "nodes" in cfg_dict["agent"]:
            agent_nodes_data = cfg_dict["agent"]["nodes"]
        elif "nodes" in cfg_dict:
            agent_nodes_data = cfg_dict["nodes"]

        if agent_nodes_data:
            insert_idx = len(nodes_data)
            for i, node in enumerate(nodes_data):
                if node["name"] == "Supervisor":
                    insert_idx = i
                    break

            for node in reversed(agent_nodes_data):
                nodes_data.insert(insert_idx, node)

        # Create NodeConfig objects
        resolved_nodes = [NodeConfig(**node) for node in nodes_data]

        # Create ResolvedExperimentConfig
        resolved_config = ResolvedExperimentConfig(
            experiment=ExperimentMetadata(**experiment_data),
            nodes=resolved_nodes,
            execution=ExecutionConfig(**execution_data),
            postprocess=postprocess_data,
        )

        # Instantiate Nodes
        factory = NodeFactory()
        nodes = []

        for node_config in resolved_nodes:
            node = factory.create(
                node_type=node_config.type,
                rate_hz=node_config.rate_hz,
                params=node_config.params,
            )
            nodes.append(node)

        # Create Experiment Structure
        experiment_id = str(uuid.uuid4())

        experiment = ExperimentStructure(
            id=experiment_id,
            type=resolved_config.experiment.type,
            config=resolved_config,
            nodes=nodes,
        )

        # Create postprocessor (Fail Fast)
        if experiment.type == "evaluation":
            postprocessor: ExperimentPostprocessor = EvaluationPostprocessor(experiment.config)
        else:
            raise ValueError(f"Unknown experiment type: {experiment.type}")

        # Create and run via runner
        runner: ExperimentRunner = self.runner_factory.create(experiment.type)

        import time

        start_time = time.perf_counter()
        result = runner.run(experiment)
        simulation_time = time.perf_counter() - start_time
        print(f"Simulation execution time: {simulation_time:.2f} seconds")

        # Postprocess
        start_time = time.perf_counter()
        processed_result = postprocessor.process(result, experiment.config)
        postprocess_time = time.perf_counter() - start_time
        print(f"Postprocessing (Dashboard) execution time: {postprocess_time:.2f} seconds")

        return processed_result
