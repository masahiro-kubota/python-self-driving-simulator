import math

from core.data import ComponentConfig, VehicleParameters, VehicleState
from core.data.autoware import Trajectory
from core.data.node_io import NodeIO
from core.interfaces.node import Node, NodeExecutionResult
from core.utils.geometry import distance, normalize_angle
from pydantic import Field


class PIDConfig(ComponentConfig):
    """Configuration for PIDControllerNode."""

    kp: float = Field(..., description="Proportional gain")
    ki: float = Field(..., description="Integral gain")
    kd: float = Field(..., description="Derivative gain")
    u_min: float = Field(..., description="Minimum acceleration [m/s^2]")
    u_max: float = Field(..., description="Maximum acceleration [m/s^2]")
    vehicle_params: VehicleParameters = Field(..., description="Vehicle parameters")


class PIDControllerNode(Node[PIDConfig]):
    """PID Controller node for combined steering (Pure Pursuit logic legacy) and velocity control."""

    def __init__(self, config: PIDConfig, rate_hz: float, priority: int):
        super().__init__("PIDController", rate_hz, config, priority)
        self.vehicle_params = config.vehicle_params
        self.wheelbase = self.vehicle_params.wheelbase
        # self.config is set by base class

        self.integral_error = 0.0
        self.prev_error = 0.0

    def get_node_io(self) -> NodeIO:
        from core.data.autoware import AckermannControlCommand, Trajectory

        return NodeIO(
            inputs={"trajectory": Trajectory, "vehicle_state": VehicleState},
            outputs={"control_cmd": AckermannControlCommand},
        )

    def on_run(self, _current_time: float) -> NodeExecutionResult:
        if self.frame_data is None:
            return NodeExecutionResult.FAILED

        trajectory = self.subscribe("trajectory")
        vehicle_state = self.subscribe("vehicle_state")

        if trajectory is None or vehicle_state is None:
            return NodeExecutionResult.SKIPPED

        steering, acceleration = self._process(trajectory, vehicle_state)

        # Output AckermannControlCommand
        from core.data.autoware import (
            AckermannControlCommand,
            AckermannLateralCommand,
            LongitudinalCommand,
        )
        from core.utils.ros_message_builder import to_ros_time

        self.publish(
            "control_cmd",
            AckermannControlCommand(
                stamp=to_ros_time(_current_time),
                lateral=AckermannLateralCommand(
                    stamp=to_ros_time(_current_time), steering_tire_angle=steering
                ),
                longitudinal=LongitudinalCommand(
                    stamp=to_ros_time(_current_time), acceleration=acceleration, speed=0.0
                ),  # Speed tracking optional/TODO
            ),
        )

        return NodeExecutionResult.SUCCESS

    def _process(self, trajectory: Trajectory, vehicle_state: VehicleState) -> tuple[float, float]:
        """Process trajectory and vehicle state to compute control commands.

        Returns:
            tuple: (steering, acceleration)
        """
        if not trajectory or len(trajectory) == 0:
            # Stop
            return 0.0, 0.0

        # 1. Steering Control (Pure Pursuit logic repeated here as per original controller.py)
        # Note: Ideally this logic should be distinct? But original PIDController did steering too.
        # Assuming the first point is the lookahead target provided by Planner.
        target_point = trajectory[0]

        # Handle Autoware TrajectoryPoint or Internal
        if hasattr(target_point, "pose"):
            target_x = target_point.pose.position.x
            target_y = target_point.pose.position.y
            target_velocity = target_point.longitudinal_velocity_mps
        else:
            target_x = target_point.x
            target_y = target_point.y
            target_velocity = target_point.velocity

        target_angle = math.atan2(target_y - vehicle_state.y, target_x - vehicle_state.x)
        alpha = normalize_angle(target_angle - vehicle_state.yaw)
        ld = distance(vehicle_state.x, vehicle_state.y, target_x, target_y)

        if ld < 1e-3:
            steering = 0.0
        else:
            steering = math.atan2(2 * self.wheelbase * math.sin(alpha), ld)

        # 2. Velocity Control (PID)
        current_velocity = vehicle_state.velocity

        error = target_velocity - current_velocity
        # Integral with anti-windup (clamping integral term?)
        # For now, standard accumulation, applied output clamping.
        self.integral_error += error
        derivative_error = error - self.prev_error

        acceleration = (
            self.config.kp * error
            + self.config.ki * self.integral_error
            + self.config.kd * derivative_error
        )

        # Output Saturation
        acceleration = max(self.config.u_min, min(self.config.u_max, acceleration))

        self.prev_error = error

        return steering, acceleration
