from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.interfaces.node import Node


@dataclass
class Experiment:
    """実験インスタンスの基本構造体"""

    id: str
    type: str
    config: Any
    nodes: list[Node]


@dataclass
class Metrics:
    """実験結果の集計メトリクス"""

    success_rate: float = 0.0
    collision_count: int = 0
    goal_count: int = 0
    checkpoint_count: int = 0
    termination_code: int = 0


@dataclass
class Artifact:
    """実験の成果物（ダッシュボード、成果物等）"""

    local_path: Path


@dataclass
class ExperimentResult:
    """実験フェーズの実行結果"""

    experiment: Experiment
    simulation_results: list[Any]
    metrics: Metrics
    execution_time: datetime
    artifacts: list[Artifact]

    @property
    def experiment_name(self) -> str:
        return self.experiment.config.experiment.name

    @property
    def experiment_type(self) -> str:
        return self.experiment.type
