from typing import Any

from omegaconf import DictConfig


class ExperimentOrchestrator:
    """実験フェーズのオーケストレーター"""

    def run_from_hydra(self, cfg: DictConfig) -> Any:
        """Hydra設定から実験フェーズを実行"""
        from experiment.core.config import ExperimentConfig
        from omegaconf import OmegaConf

        # Validate configuration using Pydantic
        # Resolve OmegaConf to primitive types for Pydantic validation
        cfg_container = OmegaConf.to_container(cfg, resolve=True)
        validated_config = ExperimentConfig(**cfg_container)

        # Convert back to OmegaConf to support dot access in existing codebase (e.g. cfg.system.nodes)
        # and ensure we are using the validated/sanitized configuration.
        strict_cfg = OmegaConf.create(validated_config.model_dump())

        phase = strict_cfg.experiment.type

        if phase == "data_collection":
            from experiment.engine.collector import CollectorEngine

            engine = CollectorEngine()
        elif phase == "extraction":
            from experiment.engine.extractor import ExtractorEngine

            engine = ExtractorEngine()
        elif phase == "training":
            from experiment.engine.trainer import TrainerEngine

            engine = TrainerEngine()
        elif phase == "evaluation":
            from experiment.engine.evaluator import EvaluatorEngine

            engine = EvaluatorEngine()
        else:
            raise ValueError(f"Unknown experiment phase: {phase}")

        return engine.run(strict_cfg)
