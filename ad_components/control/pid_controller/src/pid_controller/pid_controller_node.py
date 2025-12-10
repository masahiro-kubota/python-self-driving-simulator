import math

from pydantic import BaseModel, Field

from core.data import Action, VehicleParameters, VehicleState
from core.data.ad_components import Trajectory
from core.data.node_io import NodeIO
from core.interfaces.node import Node
from core.utils.geometry import distance, normalize_angle


class PIDConfig(BaseModel):
    """Configuration for PIDControllerNode."""

    kp: float = Field(..., description="Proportional gain")
    ki: float = Field(..., description="Integral gain")
    kd: float = Field(..., description="Derivative gain")
    u_min: float = Field(-10.0, description="Minimum acceleration [m/s^2]")
    u_max: float = Field(10.0, description="Maximum acceleration [m/s^2]")


class PIDControllerNode(Node):
    """PID Controller node for combined steering (Pure Pursuit logic legacy) and velocity control."""

    def __init__(
        self, config: dict, rate_hz: float, vehicle_params: VehicleParameters | None = None
    ):
        super().__init__("PIDController", rate_hz, config, config_model=PIDConfig)
        self.vehicle_params = vehicle_params or VehicleParameters()
        self.wheelbase = self.vehicle_params.wheelbase
        self.config: PIDConfig = self.validated_config

        self.integral_error = 0.0
        self.prev_error = 0.0

    def get_node_io(self) -> NodeIO:
        return NodeIO(
            inputs={"trajectory": Trajectory, "vehicle_state": VehicleState},
            outputs={"action": Action},
        )

    def on_run(self, _current_time: float) -> bool:
        if self.frame_data is None:
            return False

        trajectory = getattr(self.frame_data, "trajectory", None)
        vehicle_state = getattr(self.frame_data, "vehicle_state", None)

        if trajectory is None or vehicle_state is None:
            # Maybe output safe action? Or nothing?
            return True

        action = self._process(trajectory, vehicle_state)
        self.frame_data.action = action
        return True

    def _process(self, trajectory: Trajectory, vehicle_state: VehicleState) -> Action:
        if not trajectory:
            return Action(steering=0.0, acceleration=0.0)

        # 1. Steering Control (Pure Pursuit logic repeated here as per original controller.py)
        # Note: Ideally this logic should be distinct? But original PIDController did steering too.
        # Assuming the first point is the lookahead target provided by Planner.
        target_point = trajectory[0]

        target_angle = math.atan2(
            target_point.y - vehicle_state.y, target_point.x - vehicle_state.x
        )
        alpha = normalize_angle(target_angle - vehicle_state.yaw)
        ld = distance(vehicle_state.x, vehicle_state.y, target_point.x, target_point.y)

        if ld < 1e-3:
            steering = 0.0
        else:
            steering = math.atan2(2 * self.wheelbase * math.sin(alpha), ld)

        # 2. Velocity Control (PID)
        target_velocity = target_point.velocity
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

        return Action(steering=steering, acceleration=acceleration)
