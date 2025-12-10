from dataclasses import dataclass

from core.interfaces import Node
from experiment.preprocessing.schemas import ResolvedExperimentConfig


@dataclass
class Experiment:
    """An executable experiment definition containing configuration and initialized nodes."""

    id: str
    type: str
    config: ResolvedExperimentConfig
    nodes: list[Node]
