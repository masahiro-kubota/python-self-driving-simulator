"""Pure Pursuit Controller Node."""

import math
from typing import Any

from core.data import ComponentConfig, VehicleParameters, VehicleState
from core.data.ad_components import Trajectory, TrajectoryPoint
from core.data.node_io import NodeIO
from core.data.ros import ColorRGBA, MarkerArray
from core.interfaces.node import Node, NodeExecutionResult
from core.utils.geometry import distance, normalize_angle
from planning_utils.visualization import create_trajectory_marker
from pydantic import Field


class LateralControlParams(ComponentConfig):
    """Lateral control parameters (Pure Pursuit)."""

    min_lookahead_distance: float = Field(..., description="Minimum lookahead distance [m]")
    max_lookahead_distance: float = Field(..., description="Maximum lookahead distance [m]")
    lookahead_speed_ratio: float = Field(..., description="Lookahead distance speed ratio [s]")


class LongitudinalControlParams(ComponentConfig):
    """Longitudinal control parameters (PID)."""

    kp: float = Field(..., description="Proportional gain for velocity control")
    ki: float = Field(..., description="Integral gain for velocity control")
    kd: float = Field(..., description="Derivative gain for velocity control")
    u_min: float = Field(..., description="Minimum acceleration [m/s^2]")
    u_max: float = Field(..., description="Maximum acceleration [m/s^2]")


class PurePursuitControllerConfig(ComponentConfig):
    """Configuration for PurePursuitControllerNode."""

    vehicle_params: VehicleParameters = Field(..., description="Vehicle parameters")
    lookahead_marker_color: str = Field("#FF00FFCC", description="Lookahead marker color")

    lateral: LateralControlParams = Field(..., description="Lateral control parameters")
    longitudinal: LongitudinalControlParams = Field(
        ..., description="Longitudinal control parameters"
    )


class PurePursuitControllerNode(Node[PurePursuitControllerConfig]):
    """Pure Pursuit controller for path tracking."""

    def __init__(self, config: PurePursuitControllerConfig, rate_hz: float, priority: int):
        super().__init__("PurePursuitController", rate_hz, config, priority)
        self.vehicle_params = config.vehicle_params
        self.wheelbase = self.vehicle_params.wheelbase

        # PID state
        self.integral_error = 0.0
        self.prev_error = 0.0

    def get_node_io(self) -> NodeIO:
        from core.data.ros import AckermannDriveStamped

        return NodeIO(
            inputs={"trajectory": Trajectory, "vehicle_state": VehicleState},
            outputs={
                "control_cmd": AckermannDriveStamped,
                "lookahead_marker": MarkerArray,
            },
        )

    def on_run(self, _current_time: float) -> NodeExecutionResult:
        if self.frame_data is None:
            return NodeExecutionResult.FAILED

        trajectory = self.subscribe("trajectory")
        vehicle_state = self.subscribe("vehicle_state")

        if trajectory is None or vehicle_state is None:
            return NodeExecutionResult.SKIPPED

        if not trajectory or len(trajectory) == 0:
            # Output zero control command
            from core.data.ros import AckermannDrive, AckermannDriveStamped, Header
            from core.utils.ros_message_builder import to_ros_time

            self.publish(
                "control_cmd",
                AckermannDriveStamped(
                    header=Header(stamp=to_ros_time(_current_time), frame_id="base_link"),
                    drive=AckermannDrive(steering_angle=0.0, acceleration=0.0),
                ),
            )
            return NodeExecutionResult.SUCCESS

        steering, acceleration, target_point = self._compute_control(trajectory, vehicle_state)

        # Output AckermannDriveStamped
        from core.data.ros import AckermannDrive, AckermannDriveStamped, Header
        from core.utils.ros_message_builder import to_ros_time

        self.publish(
            "control_cmd",
            AckermannDriveStamped(
                header=Header(stamp=to_ros_time(_current_time), frame_id="base_link"),
                drive=AckermannDrive(
                    steering_angle=steering,
                    acceleration=acceleration,
                ),
            ),
        )

        # Output Visualization Marker
        marker = create_trajectory_marker(
            trajectory=Trajectory(points=[target_point]),
            timestamp=_current_time,
            ns="pure_pursuit_lookahead",
            color=ColorRGBA.from_hex(self.config.lookahead_marker_color),
        )
        self.publish("lookahead_marker", MarkerArray(markers=[marker]))

        return NodeExecutionResult.SUCCESS

    def _compute_control(
        self, trajectory: Trajectory, vehicle_state: VehicleState
    ) -> tuple[float, float, Any]:
        """Compute steering and acceleration using Pure Pursuit + PID.

        Returns:
            tuple: (steering, acceleration, target_point)
        """

        # 1. Calculate dynamic lookahead distance
        current_speed = vehicle_state.velocity
        lookahead = max(
            self.config.lateral.min_lookahead_distance,
            min(
                self.config.lateral.max_lookahead_distance,
                current_speed * self.config.lateral.lookahead_speed_ratio,
            ),
        )

        # 2. Find lookahead point on trajectory
        target_point = self._find_lookahead_point(trajectory, vehicle_state, lookahead)

        # 3. Pure Pursuit steering control
        target_angle = math.atan2(
            target_point.y - vehicle_state.y, target_point.x - vehicle_state.x
        )
        alpha = normalize_angle(target_angle - vehicle_state.yaw)
        ld = distance(vehicle_state.x, vehicle_state.y, target_point.x, target_point.y)

        if ld < 1e-3:
            steering = 0.0
        else:
            steering = math.atan2(2 * self.wheelbase * math.sin(alpha), ld)

        # Clip steering to vehicle limits
        steering = max(
            -self.vehicle_params.max_steering_angle,
            min(self.vehicle_params.max_steering_angle, steering),
        )

        # 4. PID velocity control
        target_velocity = target_point.velocity
        current_velocity = vehicle_state.velocity

        error = target_velocity - current_velocity
        self.integral_error += error
        derivative_error = error - self.prev_error

        acceleration = (
            self.config.longitudinal.kp * error
            + self.config.longitudinal.ki * self.integral_error
            + self.config.longitudinal.kd * derivative_error
        )

        # Output saturation
        acceleration = max(
            self.config.longitudinal.u_min, min(self.config.longitudinal.u_max, acceleration)
        )

        self.prev_error = error

        return steering, acceleration, target_point

    def _find_lookahead_point(
        self, trajectory: Trajectory, vehicle_state: VehicleState, lookahead: float
    ):
        """Find the lookahead point on the trajectory."""

        # Find nearest point
        min_dist = float("inf")
        nearest_idx = 0

        for i, point in enumerate(trajectory):
            d = distance(vehicle_state.x, vehicle_state.y, point.x, point.y)
            if d < min_dist:
                min_dist = d
                nearest_idx = i

        # Search forward from nearest point
        accumulated_dist = 0.0
        current_idx = nearest_idx
        target_point = trajectory[nearest_idx]

        while accumulated_dist < lookahead:
            if current_idx >= len(trajectory) - 1:
                target_point = trajectory[-1]
                break

            p1 = trajectory[current_idx]
            p2 = trajectory[current_idx + 1]
            d = distance(p1.x, p1.y, p2.x, p2.y)

            if accumulated_dist + d >= lookahead:
                # Interpolate
                remaining = lookahead - accumulated_dist
                ratio = remaining / d if d > 1e-6 else 0.0

                target_point = TrajectoryPoint(
                    x=p1.x + (p2.x - p1.x) * ratio,
                    y=p1.y + (p2.y - p1.y) * ratio,
                    yaw=math.atan2(p2.y - p1.y, p2.x - p1.x),
                    velocity=p1.velocity + (p2.velocity - p1.velocity) * ratio,
                )
                break

            accumulated_dist += d
            current_idx += 1
            target_point = trajectory[current_idx]

        return target_point
