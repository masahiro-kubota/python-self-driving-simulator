"""Centerline Planner Node."""

from pathlib import Path

from core.data import ComponentConfig, VehicleState
from core.data.ad_components import Trajectory
from core.data.node_io import NodeIO
from core.interfaces.node import Node, NodeExecutionResult
from core.utils.geometry import distance
from pydantic import Field


class CenterlinePlannerConfig(ComponentConfig):
    """Configuration for CenterlinePlannerNode."""

    track_path: Path = Field(..., description="Path to reference trajectory CSV")
    lookahead_points: int = Field(100, description="Number of points to output ahead of vehicle")


class CenterlinePlannerNode(Node[CenterlinePlannerConfig]):
    """Simple planner that outputs centerline trajectory from CSV."""

    def __init__(self, config: CenterlinePlannerConfig, rate_hz: float):
        super().__init__("CenterlinePlanner", rate_hz, config)

        # Load reference trajectory
        from core.utils import get_project_root
        from planning_utils import load_track_csv

        track_path = self.config.track_path
        if not track_path.is_absolute():
            track_path = get_project_root() / track_path

        self.reference_trajectory = load_track_csv(track_path)

    def get_node_io(self) -> NodeIO:
        return NodeIO(
            inputs={"vehicle_state": VehicleState},
            outputs={"trajectory": Trajectory},
        )

    def on_run(self, _current_time: float) -> NodeExecutionResult:
        if self.frame_data is None:
            return NodeExecutionResult.FAILED

        vehicle_state = getattr(self.frame_data, "vehicle_state", None)
        if vehicle_state is None:
            return NodeExecutionResult.SKIPPED

        # Find nearest point on reference trajectory
        min_dist = float("inf")
        nearest_idx = 0

        for i, point in enumerate(self.reference_trajectory):
            d = distance(vehicle_state.x, vehicle_state.y, point.x, point.y)
            if d < min_dist:
                min_dist = d
                nearest_idx = i

        # Output trajectory from nearest point forward
        end_idx = min(nearest_idx + self.config.lookahead_points, len(self.reference_trajectory))

        trajectory_points = self.reference_trajectory.points[nearest_idx:end_idx]

        # If we're near the end, wrap around or just use remaining points
        if len(trajectory_points) == 0:
            trajectory_points = self.reference_trajectory.points[-self.config.lookahead_points :]

        self.frame_data.trajectory = Trajectory(points=trajectory_points)
        return NodeExecutionResult.SUCCESS
