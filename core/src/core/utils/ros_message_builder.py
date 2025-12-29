"""ROS 2 message builder utilities for logger."""

import math

from core.data import VehicleState
from core.data.ros import (
    Header,
    LaserScan,
    Odometry,
    Point,
    Pose,
    PoseWithCovariance,
    Quaternion,
    TFMessage,
    Time,
    Transform,
    TransformStamped,
    Twist,
    TwistWithCovariance,
    Vector3,
)
from core.data.vehicle.params import LidarConfig


def to_ros_time(t: float) -> Time:
    """Convert float timestamp to ROS Time.

    Args:
        t: Timestamp in seconds.

    Returns:
        ROS Time message.
    """
    sec = int(t)
    nanosec = int((t - sec) * 1e9)

    # Ensure strict constraints for valid ROS time
    if nanosec < 0:
        nanosec = 0
    if nanosec >= 1_000_000_000:
        nanosec = 999_999_999

    return Time(sec=sec, nanosec=nanosec, nsec=nanosec)


def quaternion_from_yaw(yaw: float) -> Quaternion:
    """Convert yaw angle to quaternion.

    Args:
        yaw: Yaw angle in radians.

    Returns:
        Quaternion representing the rotation.
    """
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


def build_tf_message(vehicle_state: VehicleState, timestamp: float) -> TFMessage:
    """Build TF message for map -> base_link transform.

    Args:
        vehicle_state: Current vehicle state.
        timestamp: Current timestamp.

    Returns:
        TF message containing the transform.
    """
    ros_time = to_ros_time(timestamp)
    q = quaternion_from_yaw(vehicle_state.yaw)

    return TFMessage(
        transforms=[
            TransformStamped(
                header=Header(stamp=ros_time, frame_id="map"),
                child_frame_id="base_link",
                transform=Transform(
                    translation=Vector3(x=vehicle_state.x, y=vehicle_state.y, z=0.0),
                    rotation=q,
                ),
            )
        ]
    )


def build_odometry_message(vehicle_state: VehicleState, timestamp: float) -> Odometry:
    """Build Odometry message from vehicle state.

    Args:
        vehicle_state: Current vehicle state.
        timestamp: Current timestamp.

    Returns:
        Odometry message.
    """
    ros_time = to_ros_time(timestamp)
    q = quaternion_from_yaw(vehicle_state.yaw)

    return Odometry(
        header=Header(stamp=ros_time, frame_id="map"),
        child_frame_id="base_link",
        pose=PoseWithCovariance(
            pose=Pose(
                position=Point(x=vehicle_state.x, y=vehicle_state.y, z=0.0),
                orientation=q,
            )
        ),
        twist=TwistWithCovariance(
            twist=Twist(
                linear=Vector3(x=vehicle_state.velocity, y=0.0, z=0.0),
                angular=Vector3(x=0.0, y=0.0, z=0.0),
            )
        ),
    )


def build_lidar_tf_message(config: LidarConfig, timestamp: float) -> TFMessage:
    """Build TF message for base_link -> lidar_link transform.

    Args:
        config: LiDAR configuration.
        timestamp: Current timestamp.

    Returns:
        TF message containing the transform.
    """
    ros_time = to_ros_time(timestamp)
    q_lidar = quaternion_from_yaw(config.yaw)

    return TFMessage(
        transforms=[
            TransformStamped(
                header=Header(stamp=ros_time, frame_id="base_link"),
                child_frame_id="lidar_link",
                transform=Transform(
                    translation=Vector3(x=config.x, y=config.y, z=config.z),
                    rotation=q_lidar,
                ),
            )
        ]
    )


def build_laser_scan_message(
    config: LidarConfig, ranges: list[float], timestamp: float
) -> LaserScan:
    """Build LaserScan message from LiDAR scan data.

    Args:
        config: LiDAR configuration.
        ranges: List of ranges.
        timestamp: Current timestamp.

    Returns:
        LaserScan message.
    """
    return LaserScan(
        header=Header(stamp=to_ros_time(timestamp), frame_id="lidar_link"),
        angle_min=-math.radians(config.fov) / 2,
        angle_max=math.radians(config.fov) / 2,
        angle_increment=math.radians(config.fov) / config.num_beams
        if config.num_beams > 0
        else 0.0,
        range_min=config.range_min,
        range_max=config.range_max,
        ranges=[r if r != float("inf") and not math.isnan(r) else float("inf") for r in ranges],
        intensities=[],
    )
