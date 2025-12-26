"""Experiment result data structures."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from core.data.experiment.config import ResolvedExperimentConfig
from core.data.simulator.result import SimulationResult


@dataclass
class Artifact:
    """実験アーティファクト.

    Attributes:
        local_path: ローカルファイルパス
        remote_path: 保存先のパス（ディレクトリ名など）
        description: 説明
    """

    local_path: Path | str
    remote_path: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        """パスの型変換."""
        if isinstance(self.local_path, str):
            self.local_path = Path(self.local_path)


@dataclass
class EvaluationMetrics:
    """Standard metrics for experiment evaluation."""

    lap_time_sec: float
    collision_count: int
    success: int
    termination_code: int  # 0: unknown, 1: goal, 2: off_track, 3: timeout, 4: sim_comp, 5: collision
    goal_count: int = 0
    checkpoint_count: int = 0

    def to_dict(self) -> dict[str, float | int]:
        """Convert to dictionary for MLflow."""
        return asdict(self)


@dataclass
class ExperimentResult:
    """実験全体の結果.

    Attributes:
        experiment_name: 実験名
        experiment_type: 実験タイプ
        execution_time: 実行時刻
        simulation_results: シミュレーション結果のリスト
        config: 実験設定
        params: 実験パラメータ（追加の動的パラメータ）
        artifacts: アーティファクトのリスト
        metrics: 評価メトリクス
        mlflow_run_id: MLflow Run ID
        dataset_path: データ収集時のS3パス
        model_path: トレーニング時のモデルパス
    """

    experiment_name: str
    experiment_type: str
    execution_time: datetime
    simulation_results: list[SimulationResult]
    config: ResolvedExperimentConfig | None = None
    params: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    metrics: EvaluationMetrics | None = None
    mlflow_run_id: str | None = None
    dataset_path: str | None = None
    model_path: str | None = None
