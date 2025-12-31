"""MPC Lateral Controller Node with PID Longitudinal Control."""

import logging
import math

import numpy as np
from core.data import ComponentConfig, VehicleParameters, VehicleState
from core.data.autoware import (
    AckermannControlCommand,
    AckermannLateralCommand,
    LongitudinalCommand,
    Trajectory,
)
from core.data.node_io import NodeIO
from core.data.ros import ColorRGBA, Marker, MarkerArray, Point
from core.interfaces.node import Node, NodeExecutionResult
from core.utils.geometry import distance, normalize_angle
from core.utils.ros_message_builder import to_ros_time
from pydantic import Field

from mpc_lateral_controller.mpc_solver import LinearMPCLateralSolver, MPCConfig

logger = logging.getLogger(__name__)


class MPCLateralParams(ComponentConfig):
    """MPC lateral control parameters."""

    prediction_horizon: int = Field(..., description="Prediction horizon [steps]")
    control_horizon: int = Field(..., description="Control horizon [steps]")
    dt: float = Field(..., description="Discretization time step [s]")
    weight_lateral_error: float = Field(..., description="Weight for lateral error")
    weight_heading_error: float = Field(..., description="Weight for heading error")
    weight_steering: float = Field(..., description="Weight for steering input")
    weight_steering_rate: float = Field(..., description="Weight for steering rate")
    max_steering_angle: float = Field(..., description="Maximum steering angle [rad]")
    max_steering_rate: float = Field(..., description="Maximum steering rate [rad/s]")


class LongitudinalControlParams(ComponentConfig):
    """Longitudinal control parameters (PID)."""

    kp: float = Field(..., description="Proportional gain for velocity control")
    ki: float = Field(..., description="Integral gain for velocity control")
    kd: float = Field(..., description="Derivative gain for velocity control")
    u_min: float = Field(..., description="Minimum acceleration [m/s^2]")
    u_max: float = Field(..., description="Maximum acceleration [m/s^2]")


class MPCLateralControllerConfig(ComponentConfig):
    """Configuration for MPCLateralControllerNode."""

    vehicle_params: VehicleParameters = Field(..., description="Vehicle parameters")
    mpc_lateral: MPCLateralParams = Field(..., description="MPC lateral control parameters")
    longitudinal: LongitudinalControlParams = Field(
        ..., description="Longitudinal control parameters"
    )


class MPCLateralControllerNode(Node[MPCLateralControllerConfig]):
    """MPC-based lateral controller with PID longitudinal control.

    This controller uses Model Predictive Control (MPC) for lateral path tracking
    and a simple PID controller for longitudinal speed control.
    """

    def __init__(
        self, config: MPCLateralControllerConfig, rate_hz: float, priority: int
    ) -> None:
        super().__init__("MPCLateralController", rate_hz, config, priority)

        # Initialize MPC solver
        mpc_config = MPCConfig(
            prediction_horizon=self.config.mpc_lateral.prediction_horizon,
            control_horizon=self.config.mpc_lateral.control_horizon,
            dt=self.config.mpc_lateral.dt,
            weight_lateral_error=self.config.mpc_lateral.weight_lateral_error,
            weight_heading_error=self.config.mpc_lateral.weight_heading_error,
            weight_steering=self.config.mpc_lateral.weight_steering,
            weight_steering_rate=self.config.mpc_lateral.weight_steering_rate,
            max_steering_angle=self.config.mpc_lateral.max_steering_angle,
            max_steering_rate=self.config.mpc_lateral.max_steering_rate,
        )
        self.mpc_solver = LinearMPCLateralSolver(
            config=mpc_config, wheelbase=self.config.vehicle_params.wheelbase
        )

        # PID state for longitudinal control
        self.velocity_error_integral = 0.0
        self.previous_velocity_error = 0.0
        self.previous_time = None

    def get_node_io(self) -> NodeIO:
        return NodeIO(
            inputs={"trajectory": Trajectory, "vehicle_state": VehicleState},
            outputs={
                "control_cmd": AckermannControlCommand,
                "debug_predicted_trajectory": MarkerArray,
            },
        )

    def on_run(self, current_time: float) -> NodeExecutionResult:
        trajectory = self.subscribe("trajectory")
        vehicle_state = self.subscribe("vehicle_state")

        if trajectory is None or vehicle_state is None:
            return NodeExecutionResult.SKIPPED

        if not trajectory or len(trajectory) == 0:
            # Output zero control command
            self.publish(
                "control_cmd",
                AckermannControlCommand(
                    stamp=to_ros_time(current_time),
                    lateral=AckermannLateralCommand(
                        stamp=to_ros_time(current_time), steering_tire_angle=0.0
                    ),
                    longitudinal=LongitudinalCommand(
                        stamp=to_ros_time(current_time), acceleration=0.0, speed=0.0
                    ),
                ),
            )
            return NodeExecutionResult.SUCCESS

        # Compute control commands
        steering_angle, acceleration = self._compute_control(
            trajectory, vehicle_state, current_time
        )

        # Output control command
        self.publish(
            "control_cmd",
            AckermannControlCommand(
                stamp=to_ros_time(current_time),
                lateral=AckermannLateralCommand(
                    stamp=to_ros_time(current_time), steering_tire_angle=steering_angle
                ),
                longitudinal=LongitudinalCommand(
                    stamp=to_ros_time(current_time), acceleration=acceleration, speed=0.0
                ),
            ),
        )

        return NodeExecutionResult.SUCCESS

    def _quaternion_to_yaw(self, quat) -> float:
        """Convert quaternion to yaw angle.
        
        Args:
            quat: Quaternion object with x, y, z, w fields
            
        Returns:
            Yaw angle in radians
        """
        # Extract yaw from quaternion using simplified formula
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        return math.atan2(
            2.0 * (quat.w * quat.z + quat.x * quat.y),
            1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        )

    def _compute_control(
        self, trajectory: Trajectory, vehicle_state: VehicleState, current_time: float
    ) -> tuple[float, float]:
        """Compute steering and acceleration using MPC + PID.

        Args:
            trajectory: Reference trajectory
            vehicle_state: Current vehicle state
            current_time: Current simulation time

        Returns:
            tuple: (steering_angle, acceleration)
        """
        # Find closest point on trajectory
        closest_idx, min_dist = self._find_closest_point(trajectory, vehicle_state)

        logger.info(f"[MPC] Vehicle: pos=({vehicle_state.x:.2f}, {vehicle_state.y:.2f}), "
                   f"yaw={math.degrees(vehicle_state.yaw):.1f}°, v={vehicle_state.velocity:.2f}m/s, "
                   f"steering={math.degrees(vehicle_state.steering):.2f}°")
        logger.info(f"[MPC] Closest point: idx={closest_idx}, dist={min_dist:.3f}m")

        # Calculate lateral error (signed distance to path)
        lateral_error = self._calculate_lateral_error(
            trajectory.points[closest_idx], vehicle_state
        )

        # Calculate heading error
        ref_heading = self._quaternion_to_yaw(trajectory.points[closest_idx].pose.orientation)
        heading_error = normalize_angle(vehicle_state.yaw - ref_heading)

        logger.info(f"[MPC] Errors: lateral={lateral_error:.3f}m, "
                   f"heading={math.degrees(heading_error):.2f}° "
                   f"(ref_yaw={math.degrees(ref_heading):.1f}°)")

        # Extract reference curvature for prediction horizon
        reference_curvature = self._extract_reference_curvature(trajectory, closest_idx)
        logger.info(f"[MPC] Reference curvature[0:3]: {reference_curvature[:3]}")

        # Solve MPC for lateral control
        current_velocity = vehicle_state.velocity
        steering_angle, predicted_states, predicted_controls, success = self.mpc_solver.solve(
            lateral_error=lateral_error,
            heading_error=heading_error,
            current_steering=vehicle_state.steering,
            reference_curvature=reference_curvature,
            current_velocity=current_velocity,
        )

        if success:
            logger.info(f"[MPC] ✅ Optimization success")
            # Publish predicted trajectory for debugging
            self._publish_predicted_trajectory(
                predicted_states, trajectory, closest_idx, current_time
            )
        else:
            logger.warning("[MPC] ❌ Optimization failed, using current steering")
            steering_angle = vehicle_state.steering

        logger.info(f"[MPC] Steering angle: {math.degrees(steering_angle):.3f}°")
        
        # Clamp steering angle to limits
        steering_angle = np.clip(
            steering_angle,
            -self.config.mpc_lateral.max_steering_angle,
            self.config.mpc_lateral.max_steering_angle,
        )
        
        logger.info(f"[MPC] Final steering command: {math.degrees(steering_angle):.3f}° "
                   f"(limits: ±{math.degrees(self.config.mpc_lateral.max_steering_angle):.1f}°)")

        # PID longitudinal control
        target_velocity = trajectory.points[closest_idx].longitudinal_velocity_mps
        acceleration = self._compute_longitudinal_control(
            target_velocity, current_velocity, current_time
        )

        logger.info(f"[MPC] Longitudinal: target_v={target_velocity:.2f}m/s, accel={acceleration:.2f}m/s²")
        logger.info(f"[MPC] " + "="*80)

        return float(steering_angle), float(acceleration)

    def _find_closest_point(
        self, trajectory: Trajectory, vehicle_state: VehicleState
    ) -> tuple[int, float]:
        """Find the closest point on the trajectory.

        Returns:
            tuple: (index of closest point, distance to closest point)
        """
        min_dist = float("inf")
        closest_idx = 0

        # Find closest point
        for i, point in enumerate(trajectory.points):
            dist = distance(
                vehicle_state.x,
                vehicle_state.y,
                point.pose.position.x,
                point.pose.position.y,
            )
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        return closest_idx, min_dist

    def _calculate_lateral_error(
        self, target_point, vehicle_state: VehicleState
    ) -> float:
        """Calculate signed lateral error to target point.

        Positive error means vehicle is to the left of the path.
        """
        # Vector from vehicle to target point
        dx = target_point.pose.position.x - vehicle_state.x
        dy = target_point.pose.position.y - vehicle_state.y

        # Vehicle heading
        yaw = vehicle_state.yaw

        # Lateral error is the cross-track error (perpendicular distance)
        # Rotate the error vector to vehicle frame
        # Positive error = left of path → should steer right (negative)
        lateral_error = dx * math.sin(yaw) - dy * math.cos(yaw)

        return lateral_error

    def _extract_reference_curvature(
        self, trajectory: Trajectory, start_idx: int
    ) -> np.ndarray:
        """Extract reference path curvature for prediction horizon based on distance.

        Args:
            trajectory: Reference trajectory
            start_idx: Starting index (closest point)

        Returns:
            Array of curvatures
        """
        N = self.config.mpc_lateral.prediction_horizon
        dt = self.config.mpc_lateral.dt
        v = 3.0  # Assumed target velocity as per user suggestion
        
        curvatures = []
        accumulated_dist = 0.0
        current_idx = start_idx

        for i in range(N):
            target_dist = i * v * dt
            
            # Find point at target_dist along trajectory
            while current_idx < len(trajectory.points) - 1:
                dx = trajectory.points[current_idx + 1].pose.position.x - trajectory.points[current_idx].pose.position.x
                dy = trajectory.points[current_idx + 1].pose.position.y - trajectory.points[current_idx].pose.position.y
                ds = math.sqrt(dx**2 + dy**2)
                
                if accumulated_dist + ds >= target_dist:
                    # Point is in this segment
                    break
                
                accumulated_dist += ds
                current_idx += 1
            
            # Calculate curvature at current_idx
            idx = current_idx
            if idx < len(trajectory.points) - 1:
                current_yaw = self._quaternion_to_yaw(trajectory.points[idx].pose.orientation)
                next_yaw = self._quaternion_to_yaw(trajectory.points[idx + 1].pose.orientation)
                dx = trajectory.points[idx + 1].pose.position.x - trajectory.points[idx].pose.position.x
                dy = trajectory.points[idx + 1].pose.position.y - trajectory.points[idx].pose.position.y
                seg_ds = math.sqrt(dx**2 + dy**2)
                
                if seg_ds > 1e-6:
                    curvature = normalize_angle(next_yaw - current_yaw) / seg_ds
                else:
                    curvature = 0.0
            else:
                curvature = 0.0
            
            curvatures.append(curvature)

        return np.array(curvatures)

    def _compute_longitudinal_control(
        self, target_velocity: float, current_velocity: float, current_time: float
    ) -> float:
        """Compute longitudinal acceleration using PID control.

        Args:
            target_velocity: Target velocity [m/s]
            current_velocity: Current velocity [m/s]
            current_time: Current time [s]

        Returns:
            Acceleration command [m/s^2]
        """
        # Calculate velocity error
        velocity_error = target_velocity - current_velocity

        # Calculate time step
        if self.previous_time is None:
            dt = 0.02  # Default dt
        else:
            dt = current_time - self.previous_time
            dt = max(dt, 1e-6)  # Avoid division by zero

        # Update integral term
        self.velocity_error_integral += velocity_error * dt

        # Calculate derivative term
        if self.previous_velocity_error is not None:
            velocity_error_derivative = (velocity_error - self.previous_velocity_error) / dt
        else:
            velocity_error_derivative = 0.0

        # PID control law
        acceleration = (
            self.config.longitudinal.kp * velocity_error
            + self.config.longitudinal.ki * self.velocity_error_integral
            + self.config.longitudinal.kd * velocity_error_derivative
        )

        # Clamp acceleration
        acceleration = np.clip(
            acceleration, self.config.longitudinal.u_min, self.config.longitudinal.u_max
        )

        # Update state
        self.previous_velocity_error = velocity_error
        self.previous_time = current_time

        return acceleration

    def _publish_predicted_trajectory(
        self,
        predicted_states: np.ndarray,
        reference_trajectory: Trajectory,
        start_idx: int,
        current_time: float,
    ) -> None:
        """Publish predicted trajectory as MarkerArray.

        Args:
            predicted_states: [2, N+1] array of [lateral_error, heading_error]
            reference_trajectory: Reference trajectory
            start_idx: Current closest/lookahead index on reference trajectory
            current_time: Current simulation time
        """
        if predicted_states is None:
            return

        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = to_ros_time(current_time)
        marker.ns = "predicted_trajectory"
        marker.id = 0
        marker.type = 4  # LINE_STRIP
        marker.action = 0  # ADD
        marker.scale.x = 0.2  # Line width
        marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.8)  # Orange

        N = predicted_states.shape[1]
        dt = self.config.mpc_lateral.dt
        v = 3.0  # Assumed target velocity
        accumulated_dist = 0.0
        current_idx = start_idx

        for i in range(N):
            e_y = predicted_states[0, i]
            target_dist = i * v * dt

            # Find reference point at target_dist
            while current_idx < len(reference_trajectory.points) - 1:
                dx = reference_trajectory.points[current_idx + 1].pose.position.x - reference_trajectory.points[current_idx].pose.position.x
                dy = reference_trajectory.points[current_idx + 1].pose.position.y - reference_trajectory.points[current_idx].pose.position.y
                ds = math.sqrt(dx**2 + dy**2)
                
                if accumulated_dist + ds >= target_dist:
                    break
                
                accumulated_dist += ds
                current_idx += 1

            ref_point = reference_trajectory.points[current_idx]
            ref_x = ref_point.pose.position.x
            ref_y = ref_point.pose.position.y
            ref_yaw = self._quaternion_to_yaw(ref_point.pose.orientation)

            # Reconstruct predicted absolute position
            pred_x = ref_x - e_y * math.sin(ref_yaw)
            pred_y = ref_y + e_y * math.cos(ref_yaw)

            marker.points.append(Point(x=pred_x, y=pred_y, z=0.0))

        marker_array.markers.append(marker)
        self.publish("debug_predicted_trajectory", marker_array)
