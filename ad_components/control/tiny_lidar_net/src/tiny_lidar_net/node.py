"""Tiny LiDAR Net node implementation."""

import logging

import numpy as np
from core.data.node_io import NodeIO
from core.data.ros import LaserScan
from core.interfaces.node import Node, NodeExecutionResult

from tiny_lidar_net.config import TinyLidarNetConfig
from tiny_lidar_net.core import TinyLidarNetCore


class TinyLidarNetNode(Node[TinyLidarNetConfig]):
    """Tiny LiDAR Net node for end-to-end autonomous driving control.

    This node subscribes to LiDAR scan data, processes it using the
    TinyLidarNetCore logic, and publishes control commands (AckermannControlCommand).
    """

    def __init__(self, config: TinyLidarNetConfig, rate_hz: float, priority: int) -> None:
        """Initialize Tiny LiDAR Net node.

        Args:
            config: Validated configuration
            rate_hz: Node execution rate [Hz]
            priority: Execution priority
        """
        super().__init__("TinyLidarNet", rate_hz, config, priority)

        self.logger = logging.getLogger(__name__)

        self.target_velocity = config.target_velocity
        self.control_cmd_topic = config.control_cmd_topic

        # Initialize core inference engine
        try:
            self.core = TinyLidarNetCore(
                input_dim=config.input_dim,
                output_dim=config.output_dim,
                architecture=config.architecture,
                ckpt_path=config.model_path,
                acceleration=0.0,  # Unused in new logic
                control_mode="fixed",  # We only use steering from core
                max_range=config.max_range,
            )
            self.logger.info(
                f"TinyLidarNetCore initialized. Architecture: {config.architecture}, "
                f"MaxRange: {config.max_range}, TargetVelocity: {self.target_velocity} m/s"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize TinyLidarNetCore: {e}")
            raise

    def get_node_io(self) -> NodeIO:
        """Define node IO.

        Returns:
            NodeIO specification
        """
        from core.data import VehicleState
        from core.data.autoware import AckermannControlCommand

        return NodeIO(
            inputs={"perception_lidar_scan": LaserScan, "vehicle_state": VehicleState},
            outputs={self.control_cmd_topic: AckermannControlCommand},
        )

    def on_run(self, _current_time: float) -> NodeExecutionResult:
        """Execute inference step."""

        # Get LiDAR scan from frame_data (now a LaserScan message)
        lidar_scan = self.subscribe("perception_lidar_scan")
        vehicle_state = self.subscribe("vehicle_state")

        if lidar_scan is None or vehicle_state is None:
            return NodeExecutionResult.SKIPPED

        # Extract ranges from LidarScan
        ranges = np.array(lidar_scan.ranges, dtype=np.float32)

        # Process via Core Logic (returns accel, steer but we only use steer)
        _, steer = self.core.process(ranges)

        # Velocity Control (Simple P-Control)
        current_velocity = vehicle_state.velocity
        velocity_error = self.target_velocity - current_velocity
        kp_velocity = 1.0  # Simple P gain
        acceleration = kp_velocity * velocity_error
        acceleration = max(-3.0, min(3.0, acceleration))  # Clip acceleration

        # Output AckermannControlCommand
        from core.data.autoware import (
            AckermannControlCommand,
            AckermannLateralCommand,
            LongitudinalCommand,
        )
        from core.utils.ros_message_builder import to_ros_time

        self.publish(
            self.control_cmd_topic,
            AckermannControlCommand(
                stamp=to_ros_time(_current_time),
                lateral=AckermannLateralCommand(
                    stamp=to_ros_time(_current_time), steering_tire_angle=steer
                ),
                longitudinal=LongitudinalCommand(
                    stamp=to_ros_time(_current_time),
                    acceleration=acceleration,
                    speed=self.target_velocity,
                ),
            ),
        )

        return NodeExecutionResult.SUCCESS
