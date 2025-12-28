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
    TinyLidarNetCore logic, and publishes control commands (AckermannDriveStamped).
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

        # Initialize core inference engine
        try:
            self.core = TinyLidarNetCore(
                input_dim=config.input_dim,
                output_dim=config.output_dim,
                architecture=config.architecture,
                ckpt_path=config.model_path,
                acceleration=config.fixed_acceleration,
                control_mode=config.control_mode,
                max_range=config.max_range,
            )
            self.logger.info(
                f"TinyLidarNetCore initialized. Architecture: {config.architecture}, "
                f"MaxRange: {config.max_range}"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize TinyLidarNetCore: {e}")
            raise

    def get_node_io(self) -> NodeIO:
        """Define node IO.

        Returns:
            NodeIO specification
        """
        from core.data.ros import AckermannDriveStamped

        return NodeIO(
            inputs={"perception_lidar_scan": LaserScan},
            outputs={"control_cmd": AckermannDriveStamped},
        )

    def on_run(self, _current_time: float) -> NodeExecutionResult:
        """Execute inference step.

        Args:
            _current_time: Current simulation time

        Returns:
            NodeExecutionResult indicating execution status
        """
        if self.frame_data is None:
            return NodeExecutionResult.FAILED

        # Get LiDAR scan from frame_data (now a LaserScan message)
        lidar_scan = self.subscribe("perception_lidar_scan")

        if lidar_scan is None:
            return NodeExecutionResult.SKIPPED

        # Extract ranges from LidarScan
        ranges = np.array(lidar_scan.ranges, dtype=np.float32)

        # Process via Core Logic
        accel, steer = self.core.process(ranges)

        # Create and publish AckermannDriveStamped
        from core.data.ros import AckermannDrive, AckermannDriveStamped, Header
        from core.utils.ros_message_builder import to_ros_time

        self.publish(
            "control_cmd",
            AckermannDriveStamped(
                header=Header(stamp=to_ros_time(_current_time), frame_id="base_link"),
                drive=AckermannDrive(
                    steering_angle=steer,
                    acceleration=accel,
                ),
            ),
        )

        return NodeExecutionResult.SUCCESS
