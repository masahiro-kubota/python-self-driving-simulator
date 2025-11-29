"""Data structures for autonomous driving."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class VehicleState:
    """車両の状態を表すデータクラス."""

    x: float  # X座標 [m]
    y: float  # Y座標 [m]
    yaw: float  # ヨー角 [rad]
    velocity: float  # 速度 [m/s]
    acceleration: float | None = None  # 加速度 [m/s^2]
    steering: float | None = None  # ステアリング角 [rad]
    timestamp: float | None = None  # タイムスタンプ [s]

    def to_array(self) -> np.ndarray:
        """numpy配列に変換."""
        return np.array([self.x, self.y, self.yaw, self.velocity])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "VehicleState":
        """numpy配列から生成."""
        return cls(x=arr[0], y=arr[1], yaw=arr[2], velocity=arr[3])


@dataclass
class Observation:
    """センサー観測データを表すデータクラス."""

    lateral_error: float  # 横方向偏差 [m]
    heading_error: float  # ヨー角偏差 [rad]
    velocity: float  # 現在速度 [m/s]
    target_velocity: float  # 目標速度 [m/s]
    distance_to_goal: float | None = None  # ゴールまでの距離 [m]
    timestamp: float | None = None  # タイムスタンプ [s]

    def to_array(self) -> np.ndarray:
        """numpy配列に変換."""
        return np.array(
            [
                self.lateral_error,
                self.heading_error,
                self.velocity,
                self.target_velocity,
            ]
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Observation":
        """numpy配列から生成."""
        return cls(
            lateral_error=arr[0],
            heading_error=arr[1],
            velocity=arr[2],
            target_velocity=arr[3],
        )


@dataclass
class Action:
    """制御指令を表すデータクラス."""

    steering: float  # ステアリング角 [rad]
    acceleration: float  # 加速度 [m/s^2]
    timestamp: float | None = None  # タイムスタンプ [s]

    def to_array(self) -> np.ndarray:
        """numpy配列に変換."""
        return np.array([self.steering, self.acceleration])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Action":
        """numpy配列から生成."""
        return cls(steering=arr[0], acceleration=arr[1])


@dataclass
class TrajectoryPoint:
    """軌道上の1点を表すデータクラス."""

    x: float  # X座標 [m]
    y: float  # Y座標 [m]
    yaw: float  # ヨー角 [rad]
    velocity: float  # 速度 [m/s]
    curvature: float | None = None  # 曲率 [1/m]
    timestamp: float | None = None  # タイムスタンプ [s]


@dataclass
class Trajectory:
    """軌道を表すデータクラス."""

    points: list[TrajectoryPoint]  # 軌道点のリスト

    def __len__(self) -> int:
        """軌道点の数を返す."""
        return len(self.points)

    def __getitem__(self, idx: int) -> TrajectoryPoint:
        """インデックスで軌道点を取得."""
        return self.points[idx]

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """numpy配列に変換 (x, y, yaw, velocity)."""
        x = np.array([p.x for p in self.points])
        y = np.array([p.y for p in self.points])
        yaw = np.array([p.yaw for p in self.points])
        velocity = np.array([p.velocity for p in self.points])
        return x, y, yaw, velocity


@dataclass
class SimulationStep:
    """Single step in a simulation."""

    timestamp: float
    vehicle_state: VehicleState
    action: Action
    observation: Observation | None = None
    info: dict[str, Any] | None = None


class SimulationLog:
    """Log of a simulation run."""

    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        """Initialize simulation log.

        Args:
            metadata: Optional metadata about the simulation (e.g. track name, config)
        """
        self.metadata = metadata or {}
        self.steps: list[SimulationStep] = []

    def add_step(self, step: SimulationStep) -> None:
        """Add a step to the log.

        Args:
            step: Simulation step
        """
        self.steps.append(step)

    def save(self, file_path: str | Path) -> None:
        """Save log to file (JSON format).

        Args:
            file_path: Output file path
        """
        import json
        from dataclasses import asdict

        data = {
            "metadata": self.metadata,
            "steps": [
                {
                    "timestamp": s.timestamp,
                    "vehicle_state": asdict(s.vehicle_state),
                    "action": asdict(s.action),
                    # Handle observation and info if needed, skipping for simplicity in JSON
                }
                for s in self.steps
            ],
        }
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, file_path: str | Path) -> "SimulationLog":
        """Load log from file.

        Args:
            file_path: Input file path

        Returns:
            SimulationLog object
        """
        import json

        with open(file_path, "r") as f:
            data = json.load(f)

        log = cls(metadata=data.get("metadata"))
        
        for s in data.get("steps", []):
            step = SimulationStep(
                timestamp=s["timestamp"],
                vehicle_state=VehicleState(**s["vehicle_state"]),
                action=Action(**s["action"]),
                # Observation/info loading to be implemented if needed
            )
            log.add_step(step)
            
        return log
