"""Core data structures."""

from core.data.ad_components import (
    Action,
    ADComponentConfig,
    ADComponentLog,
    ADComponentSpec,
    ADComponentType,
    Trajectory,
    TrajectoryPoint,
    VehicleState,
)
from core.data.environment import (
    Obstacle,
    ObstacleShape,
    ObstacleState,
    ObstacleTrajectory,
    ObstacleType,
    Scene,
    SimulatorObstacle,
    StaticObstaclePosition,
    TrajectoryWaypoint,
)
from core.data.experiment import Artifact, EvaluationMetrics, ExperimentResult
from core.data.observation import Observation
from core.data.simulator import SimulationConfig, SimulationLog, SimulationResult, SimulationStep
from core.data.vehicle import VehicleParameters

__all__ = [
    "ADComponentConfig",
    "ADComponentLog",
    "ADComponentSpec",
    "ADComponentType",
    "Action",
    "Artifact",
    "EvaluationMetrics",
    "ExperimentResult",
    "Observation",
    "Obstacle",
    "ObstacleShape",
    "ObstacleState",
    "ObstacleTrajectory",
    "ObstacleType",
    "Scene",
    "SimulationConfig",
    "SimulationLog",
    "SimulationResult",
    "SimulationStep",
    "SimulatorObstacle",
    "StaticObstaclePosition",
    "Trajectory",
    "TrajectoryPoint",
    "TrajectoryWaypoint",
    "VehicleParameters",
    "VehicleState",
]
