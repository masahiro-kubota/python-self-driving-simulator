from pydantic import Field

from core.data import VehicleParameters, VehicleState
from core.data.ad_components import Trajectory, TrajectoryPoint
from core.data.node_io import NodeIO
from core.interfaces.node import Node, NodeConfig, NodeExecutionResult
from core.utils.geometry import distance


class PurePursuitConfig(NodeConfig):
    """Configuration for PurePursuitNode."""

    track_path: str = Field(..., description="Path to reference trajectory CSV")
    lookahead_distance: float | None = Field(
        None, description="Detailed fixed lookahead (deprecated in favor of dynamic/min/max)"
    )
    min_lookahead_distance: float = Field(..., description="Minimum lookahead distance [m]")
    max_lookahead_distance: float = Field(..., description="Maximum lookahead distance [m]")
    lookahead_speed_ratio: float = Field(..., description="Lookahead distance speed ratio [s]")

    # If lookahead_distance is set, it overrides the dynamic logic (backward compatibility)


class PurePursuitNode(Node[PurePursuitConfig]):
    """Pure Pursuit path tracking node."""

    def __init__(
        self,
        config: PurePursuitConfig,
        rate_hz: float,
        vehicle_params: VehicleParameters | None = None,
    ):
        super().__init__("PurePursuit", rate_hz, config)
        if vehicle_params is None:
            raise ValueError("VehicleParameters must be provided to PurePursuitNode")
        self.vehicle_params = vehicle_params
        self.reference_trajectory: Trajectory | None = None
        # self.config is set by base class

        if self.config.track_path:
            # Note: We need to handle path resolution.
            # In old processor, FlexibleADComponent resolved paths.
            # Here, the 'config' dictionary passed in should already have resolved paths if Loader/Factory did its job.
            # However, FlexibleADComponent._create_processor did path resolution.
            # We might need to handle it or assume absolute path.
            # If instantiated via FlexibleADComponent (updated), it should resolve paths!
            from planning_utils import load_track_csv

            self.reference_trajectory = load_track_csv(self.config.track_path)

    def get_node_io(self) -> NodeIO:
        return NodeIO(inputs={"vehicle_state": VehicleState}, outputs={"trajectory": Trajectory})

    def on_run(self, _current_time: float) -> NodeExecutionResult:
        if self.frame_data is None:
            return NodeExecutionResult.FAILED

        # Get Input
        vehicle_state = getattr(self.frame_data, "vehicle_state", None)
        if vehicle_state is None:
            return NodeExecutionResult.SKIPPED

        # Process
        trajectory = self._plan(vehicle_state)

        # Set Output
        self.frame_data.trajectory = trajectory
        return NodeExecutionResult.SUCCESS

    def _plan(self, vehicle_state: VehicleState) -> Trajectory:
        if self.reference_trajectory is None or len(self.reference_trajectory) < 2:
            return Trajectory(points=[])

        # 1. Calculate dynamic lookahead
        current_speed = vehicle_state.velocity
        lookahead = self.config.lookahead_distance
        if lookahead is None:
            lookahead = max(
                self.config.min_lookahead_distance,
                min(
                    self.config.max_lookahead_distance,
                    current_speed * self.config.lookahead_speed_ratio,
                ),
            )

        # 2. Find nearest point
        min_dist = float("inf")
        nearest_idx = 0

        # Optimization: verify if we can start search from previous nearest index?
        # For now, keep it simple stateless or full search to be safe
        for i, point in enumerate(self.reference_trajectory):
            d = distance(vehicle_state.x, vehicle_state.y, point.x, point.y)
            if d < min_dist:
                min_dist = d
                nearest_idx = i

        # 3. Find lookahead point
        target_point = self.reference_trajectory[nearest_idx]
        accumulated_dist = 0.0
        current_idx = nearest_idx

        while accumulated_dist < lookahead:
            if current_idx >= len(self.reference_trajectory) - 1:
                target_point = self.reference_trajectory[-1]
                break

            p1 = self.reference_trajectory[current_idx]
            p2 = self.reference_trajectory[current_idx + 1]
            d = distance(p1.x, p1.y, p2.x, p2.y)

            if accumulated_dist + d >= lookahead:
                remaining = lookahead - accumulated_dist
                ratio = remaining / d
                target_x = p1.x + (p2.x - p1.x) * ratio
                target_y = p1.y + (p2.y - p1.y) * ratio
                target_v = p1.velocity + (p2.velocity - p1.velocity) * ratio
                target_point = TrajectoryPoint(x=target_x, y=target_y, yaw=0.0, velocity=target_v)
                break

            accumulated_dist += d
            current_idx += 1
            target_point = self.reference_trajectory[current_idx]

        return Trajectory(points=[target_point])
