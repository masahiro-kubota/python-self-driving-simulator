"""Experiment result data structure."""

from dataclasses import dataclass, field
from datetime import datetime

from core.data.simulation.result import SimulationResult


@dataclass
class ExperimentResult:
    """実験全体の結果.

    Attributes:
        experiment_name: 実験名
        experiment_type: 実験タイプ
        execution_time: 実行時刻
        simulation_results: シミュレーション結果のリスト
        metrics: 集計メトリクス（平均成功率、平均ラップタイムなど）
        mlflow_run_id: MLflow Run ID
        dataset_path: データ収集時のS3パス
        model_path: トレーニング時のモデルパス
    """

    experiment_name: str
    experiment_type: str
    execution_time: datetime
    simulation_results: list[SimulationResult]
    metrics: dict[str, float] = field(default_factory=dict)
    mlflow_run_id: str | None = None
    dataset_path: str | None = None
    model_path: str | None = None
