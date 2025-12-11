"""Obstacle definition for future extension."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ObstacleType(Enum):
    """障害物タイプ."""

    STATIC = "static"  # 静的障害物
    DYNAMIC = "dynamic"  # 動的障害物


@dataclass
class Obstacle:
    """障害物定義（将来の拡張用）.

    現時点では最小限の実装。実際の障害物検出ロジックは未実装。
    """

    id: str  # 障害物ID
    type: ObstacleType  # 障害物タイプ
    x: float  # X座標 [m]
    y: float  # Y座標 [m]
    width: float  # 幅 [m]
    height: float  # 高さ [m]
    yaw: float = 0.0  # ヨー角 [rad]

    # NOTE: 以下のフィールドは動的障害物の将来の拡張で使用予定
    velocity: float = 0.0  # 速度 [m/s]
    trajectory: list[tuple[float, float]] | None = None  # 軌道 [(x, y), ...]


# New Pydantic-based obstacle definitions for simulator


class ObstacleShape(BaseModel):
    """Obstacle shape definition."""

    type: Literal["rectangle", "circle"] = Field(description="Shape type")
    # Rectangle parameters
    width: float | None = Field(None, description="Width [m] (for rectangle)")
    length: float | None = Field(None, description="Length [m] (for rectangle)")
    # Circle parameters
    radius: float | None = Field(None, description="Radius [m] (for circle)")


class TrajectoryWaypoint(BaseModel):
    """Single waypoint in a trajectory."""

    time: float = Field(description="Time [s]")
    x: float = Field(description="X coordinate [m]")
    y: float = Field(description="Y coordinate [m]")
    yaw: float = Field(default=0.0, description="Yaw angle [rad]")


class ObstacleTrajectory(BaseModel):
    """Obstacle trajectory definition."""

    type: Literal["waypoint"] = Field(description="Trajectory type")
    interpolation: Literal["linear", "cubic_spline"] = Field(
        default="linear", description="Interpolation method"
    )
    waypoints: list[TrajectoryWaypoint] = Field(description="Waypoints")
    loop: bool = Field(default=False, description="Loop trajectory")


class StaticObstaclePosition(BaseModel):
    """Static obstacle position."""

    x: float = Field(description="X coordinate [m]")
    y: float = Field(description="Y coordinate [m]")
    yaw: float = Field(default=0.0, description="Yaw angle [rad]")


class SimulatorObstacle(BaseModel):
    """Complete obstacle definition for simulator."""

    type: Literal["static", "dynamic"] = Field(description="Obstacle type")
    shape: ObstacleShape = Field(description="Obstacle shape")
    # Static obstacle parameters
    position: StaticObstaclePosition | None = Field(
        None, description="Position (for static obstacle)"
    )
    # Dynamic obstacle parameters
    trajectory: ObstacleTrajectory | None = Field(
        None, description="Trajectory (for dynamic obstacle)"
    )


@dataclass
class ObstacleState:
    """Obstacle state at a specific time."""

    x: float  # X coordinate [m]
    y: float  # Y coordinate [m]
    yaw: float  # Yaw angle [rad]
    timestamp: float  # Time [s]
