"""Environment data structures."""

from core.data.environment.obstacle import (
    Obstacle,
    ObstacleShape,
    ObstacleState,
    ObstacleTrajectory,
    ObstacleType,
    SimulatorObstacle,
    StaticObstaclePosition,
    TrajectoryWaypoint,
)
from core.data.environment.scene import Scene

__all__ = [
    "Obstacle",
    "ObstacleShape",
    "ObstacleState",
    "ObstacleTrajectory",
    "ObstacleType",
    "Scene",
    "SimulatorObstacle",
    "StaticObstaclePosition",
    "TrajectoryWaypoint",
]
