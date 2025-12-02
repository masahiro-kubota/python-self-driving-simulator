"""Experiment configuration data structure."""

from dataclasses import dataclass
from enum import Enum

from core.data.simulation.config import SimulationConfig


class ExperimentType(Enum):
    """実験タイプ."""

    EVALUATION = "evaluation"
    DATA_COLLECTION = "data_collection"
    TRAINING = "training"


@dataclass
class ExperimentConfig:
    """実験全体の設定.

    Attributes:
        name: 実験名
        description: 実験の説明
        type: 実験タイプ
        simulation: シミュレーション設定
        num_episodes: 実行エピソード数
        max_steps_per_episode: 1エピソードの最大ステップ数
    """

    name: str
    description: str
    type: ExperimentType
    simulation: SimulationConfig
    num_episodes: int = 1
    max_steps_per_episode: int = 2000
