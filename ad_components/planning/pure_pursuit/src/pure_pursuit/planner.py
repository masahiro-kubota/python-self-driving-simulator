"""Pure Pursuit Planner implementation."""

from ad_component_core.data import Observation

from core.data import VehicleParameters, VehicleState
from core.data.ad_components import Trajectory, TrajectoryPoint
from core.interfaces import Planner
from core.utils.geometry import distance


class PurePursuitPlanner(Planner):
    """Pure Pursuit path tracking algorithm."""

    def __init__(
        self, lookahead_distance: float = 5.0, vehicle_params: VehicleParameters | None = None
    ) -> None:
        """Initialize Pure Pursuit planner.

        Args:
            lookahead_distance: Distance to look ahead for target point [m]
            vehicle_params: Vehicle parameters (optional)
        """
        self.lookahead_distance = lookahead_distance
        self.vehicle_params = vehicle_params or VehicleParameters()
        self.reference_trajectory: Trajectory | None = None

    def set_reference_trajectory(self, trajectory: Trajectory) -> None:
        """Set the reference trajectory to track.

        Args:
            trajectory: Reference trajectory
        """
        self.reference_trajectory = trajectory

    def plan(self, _observation: Observation, vehicle_state: VehicleState) -> Trajectory:
        """Plan a trajectory using Pure Pursuit.

        Args:
            _observation: Current observation (unused in simple Pure Pursuit)
            vehicle_state: Current vehicle state

        Returns:
            Planned trajectory (single point with target steering)
        """
        if self.reference_trajectory is None or len(self.reference_trajectory) < 2:
            # Return empty trajectory or stop if no reference
            return Trajectory(points=[])

        # 1. Find nearest point on reference trajectory
        min_dist = float("inf")
        nearest_idx = 0

        # Optimization: Search around previous nearest point if available
        # For now, simple linear search
        for i, point in enumerate(self.reference_trajectory):
            d = distance(vehicle_state.x, vehicle_state.y, point.x, point.y)
            if d < min_dist:
                min_dist = d
                nearest_idx = i

        # 2. Find lookahead point
        # Search forward from nearest point
        target_point = self.reference_trajectory[nearest_idx]
        accumulated_dist = 0.0

        current_idx = nearest_idx
        while accumulated_dist < self.lookahead_distance:
            if current_idx >= len(self.reference_trajectory) - 1:
                # End of trajectory
                target_point = self.reference_trajectory[-1]
                break

            p1 = self.reference_trajectory[current_idx]
            p2 = self.reference_trajectory[current_idx + 1]
            d = distance(p1.x, p1.y, p2.x, p2.y)

            if accumulated_dist + d >= self.lookahead_distance:
                # Interpolate
                remaining = self.lookahead_distance - accumulated_dist
                ratio = remaining / d
                target_x = p1.x + (p2.x - p1.x) * ratio
                target_y = p1.y + (p2.y - p1.y) * ratio
                target_v = p1.velocity + (p2.velocity - p1.velocity) * ratio
                target_point = TrajectoryPoint(x=target_x, y=target_y, yaw=0.0, velocity=target_v)
                break

            accumulated_dist += d
            current_idx += 1
            target_point = self.reference_trajectory[current_idx]

        # 3. Calculate steering angle
        # alpha = angle between vehicle heading and direction to target
        # Note: alpha calculation is commented out as it's not currently used
        # target_angle = math.atan2(
        #     target_point.y - vehicle_state.y, target_point.x - vehicle_state.x
        # )
        # alpha = normalize_angle(target_angle - vehicle_state.yaw)

        # steering = atan(2 * L * sin(alpha) / Ld)
        # Assuming wheelbase L is handled by controller or vehicle model,
        # Pure Pursuit outputs curvature or steering for a specific wheelbase.
        # Here we output a trajectory point with the target steering.
        # Note: Ideally planner outputs a path, and controller follows it.
        # But Pure Pursuit is often used as a controller.
        # Here we treat it as a planner that generates a target trajectory point
        # that implicitly contains the control command (or we can return a trajectory to the target).

        # For this implementation, we will return a trajectory containing the target point
        # and let the controller (or simulator) use it.
        # However, to be useful for the "ControlComponent", we might want to compute the steering here
        # if this was a controller. Since it's a "Planner", we return the path to the target.

        # Let's return a trajectory from vehicle to target
        return Trajectory(points=[target_point])

    def reset(self) -> None:
        """Reset planner state."""
        pass
