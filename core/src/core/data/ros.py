"""ROS 2 standard message definitions using Pydantic."""

import struct

from pydantic import BaseModel, Field


class Time(BaseModel):
    """builtin_interfaces/Time."""

    sec: int = 0
    nanosec: int = 0
    nsec: int = 0  # Compatibility for Foxglove/jsonschema which might check for legacy field


class Header(BaseModel):
    """std_msgs/Header."""

    stamp: Time = Field(default_factory=Time)
    frame_id: str = ""


class Point(BaseModel):
    """geometry_msgs/Point."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class Vector3(BaseModel):
    """geometry_msgs/Vector3."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class Quaternion(BaseModel):
    """geometry_msgs/Quaternion."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


class Pose(BaseModel):
    """geometry_msgs/Pose."""

    position: Point = Field(default_factory=Point)
    orientation: Quaternion = Field(default_factory=Quaternion)


class Twist(BaseModel):
    """geometry_msgs/Twist."""

    linear: Vector3 = Field(default_factory=Vector3)
    angular: Vector3 = Field(default_factory=Vector3)


class PoseWithCovariance(BaseModel):
    """geometry_msgs/PoseWithCovariance."""

    pose: Pose = Field(default_factory=Pose)
    covariance: list[float] = Field(default_factory=lambda: [0.0] * 36)


class TwistWithCovariance(BaseModel):
    """geometry_msgs/TwistWithCovariance."""

    twist: Twist = Field(default_factory=Twist)
    covariance: list[float] = Field(default_factory=lambda: [0.0] * 36)


class Odometry(BaseModel):
    """nav_msgs/Odometry."""

    header: Header = Field(default_factory=Header)
    child_frame_id: str = ""
    pose: PoseWithCovariance = Field(default_factory=PoseWithCovariance)
    twist: TwistWithCovariance = Field(default_factory=TwistWithCovariance)


class LaserScan(BaseModel):
    """sensor_msgs/LaserScan."""

    header: Header = Field(default_factory=Header)
    angle_min: float = 0.0
    angle_max: float = 0.0
    angle_increment: float = 0.0
    time_increment: float = 0.0
    scan_time: float = 0.0
    range_min: float = 0.0
    range_max: float = 0.0
    ranges: list[float] = Field(default_factory=list)
    intensities: list[float] = Field(default_factory=list)


class Transform(BaseModel):
    """geometry_msgs/Transform."""

    translation: Vector3 = Field(default_factory=Vector3)
    rotation: Quaternion = Field(default_factory=Quaternion)


class TransformStamped(BaseModel):
    """geometry_msgs/TransformStamped."""

    header: Header = Field(default_factory=Header)
    child_frame_id: str = ""
    transform: Transform = Field(default_factory=Transform)


class TFMessage(BaseModel):
    """tf2_msgs/TFMessage."""

    transforms: list[TransformStamped] = Field(default_factory=list)


class Float32(BaseModel):
    """std_msgs/Float32."""

    data: float = 0.0


class Bool(BaseModel):
    """std_msgs/Bool."""

    data: bool = False


class String(BaseModel):
    """std_msgs/String."""

    data: str = ""


class ColorRGBA(BaseModel):
    """std_msgs/ColorRGBA."""

    r: float = 0.0
    g: float = 0.0
    b: float = 0.0
    a: float = 1.0

    @classmethod
    def from_hex(cls, hex_str: str) -> "ColorRGBA":
        """Create ColorRGBA from hex string (e.g. #RRGGBB or #RRGGBBAA)."""
        hex_str = hex_str.lstrip("#")
        if len(hex_str) == 6:
            r, g, b = struct.unpack("BBB", bytes.fromhex(hex_str))
            return cls(r=r / 255.0, g=g / 255.0, b=b / 255.0, a=1.0)
        if len(hex_str) == 8:
            r, g, b, a = struct.unpack("BBBB", bytes.fromhex(hex_str))
            return cls(r=r / 255.0, g=g / 255.0, b=b / 255.0, a=a / 255.0)
        raise ValueError(f"Invalid hex color string: {hex_str}")


class Marker(BaseModel):
    """visualization_msgs/Marker."""

    header: Header = Field(default_factory=Header)
    ns: str = ""
    id: int = 0
    type: int = 0
    action: int = 0
    pose: Pose = Field(default_factory=Pose)
    scale: Vector3 = Field(default_factory=Vector3)
    color: ColorRGBA = Field(default_factory=ColorRGBA)
    lifetime: Time = Field(default_factory=Time)
    frame_locked: bool = False
    points: list[Point] = Field(default_factory=list)
    colors: list[ColorRGBA] = Field(default_factory=list)
    text: str = ""
    mesh_resource: str = ""
    mesh_use_embedded_materials: bool = False


class MarkerArray(BaseModel):
    """visualization_msgs/MarkerArray."""

    markers: list[Marker] = Field(default_factory=list)
