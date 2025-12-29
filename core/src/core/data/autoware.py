"""Autoware Auto compliant message definitions."""

from pydantic import BaseModel, Field

from core.data.ros import Header, Pose, Time


class Duration(BaseModel):
    """builtin_interfaces/Duration."""

    sec: int = 0
    nanosec: int = 0


# --- Control Messages ---


class AckermannLateralCommand(BaseModel):
    """autoware_auto_control_msgs/AckermannLateralCommand."""

    stamp: Time = Field(default_factory=Time)
    steering_tire_angle: float = 0.0
    steering_tire_rotation_rate: float = 0.0


class LongitudinalCommand(BaseModel):
    """autoware_auto_control_msgs/LongitudinalCommand."""

    stamp: Time = Field(default_factory=Time)
    speed: float = 0.0
    acceleration: float = 0.0
    jerk: float = 0.0


class AckermannControlCommand(BaseModel):
    """autoware_auto_control_msgs/AckermannControlCommand."""

    stamp: Time = Field(default_factory=Time)
    lateral: AckermannLateralCommand = Field(default_factory=AckermannLateralCommand)
    longitudinal: LongitudinalCommand = Field(default_factory=LongitudinalCommand)


# --- Planning Messages ---


class TrajectoryPoint(BaseModel):
    """autoware_auto_planning_msgs/TrajectoryPoint."""

    time_from_start: Duration = Field(default_factory=Duration)
    pose: Pose = Field(default_factory=Pose)
    longitudinal_velocity_mps: float = 0.0
    lateral_velocity_mps: float = 0.0
    acceleration_mps2: float = 0.0
    heading_rate_rps: float = 0.0
    front_wheel_angle_rad: float = 0.0
    rear_wheel_angle_rad: float = 0.0


class Trajectory(BaseModel):
    """autoware_auto_planning_msgs/Trajectory."""

    header: Header = Field(default_factory=Header)
    points: list[TrajectoryPoint] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> TrajectoryPoint:
        return self.points[idx]

    def __iter__(self):
        return iter(self.points)


# --- Vehicle Messages ---


class VelocityReport(BaseModel):
    """autoware_auto_vehicle_msgs/VelocityReport."""

    header: Header = Field(default_factory=Header)
    longitudinal_velocity: float = 0.0
    lateral_velocity: float = 0.0
    heading_rate: float = 0.0


class SteeringReport(BaseModel):
    """autoware_auto_vehicle_msgs/SteeringReport."""

    stamp: Time = Field(default_factory=Time)
    steering_tire_angle: float = 0.0
